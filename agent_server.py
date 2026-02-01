# agent_server.py (patched to use bind_tools + convert_to_openai_tool)
import os, json
from typing import Dict
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import requests

from langchain_google_genai import ChatGoogleGenerativeAI  # Gemini via LangChain [uses bind_tools]
from langchain_core.utils.function_calling import convert_to_openai_tool  # modern helper

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.5-flash")
if not GOOGLE_API_KEY:
    raise ValueError("FATAL ERROR: GOOGLE_API_KEY not found in .env file.")

TOOL_SERVER_URL = os.getenv("TOOL_SERVER_URL", "http://127.0.0.1:8001")

# Friend-style dictionary router that targets the HTTP Tool Gateway
class MCPClient:
    def __init__(self, tool_server_map: Dict[str, str]):
        self.tool_map = tool_server_map
        print(f"MCP Client initialized. Known tools: {list(self.tool_map.keys())}")

    def execute_tool(self, tool_name: str, tool_args: Dict) -> Dict:
        if tool_name not in self.tool_map:
            raise HTTPException(status_code=400, detail=f"Agent chose an unknown tool: {tool_name}")
        target_url = f"{TOOL_SERVER_URL}{self.tool_map[tool_name]}"
        try:
            resp = requests.post(target_url, json=tool_args, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=503, detail=f"Tool Server error for '{tool_name}': {e}")

TOOL_SERVER_MAP = {
    "run_query": "/tools/run-query",
    "get_tables": "/tools/get-tables",
}

# Pydantic input schemas
class RunQueryInput(BaseModel):
    sql: str = Field(description="PostgreSQL SQL to execute.")
    unrestricted: bool = Field(default=False, description="Allow writes when true.")

class GetTablesInput(BaseModel):
    pass

# Convert to OpenAI tool specs (dicts) and set names/descriptions
run_query_tool = convert_to_openai_tool(RunQueryInput, strict=False)
run_query_tool["function"]["name"] = "run_query"
run_query_tool["function"]["description"] = "Execute SQL via the consolidated Tool Server (routes to MCP DB servers)."

get_tables_tool = convert_to_openai_tool(GetTablesInput, strict=False)
get_tables_tool["function"]["name"] = "get_tables"
get_tables_tool["function"]["description"] = "List schema tables via the consolidated Tool Server."

# FastAPI setup
app = FastAPI(title="MCP Agent Server (Friend-style)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

mcp_client = MCPClient(tool_server_map=TOOL_SERVER_MAP)

# Gemini LLM with tool calling; pass OpenAI tool dicts to bind_tools
llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL_NAME, temperature=0, convert_system_message_to_human=True)
agent = llm.bind_tools(tools=[run_query_tool, get_tables_tool])

class AgentRequest(BaseModel):
    goal: str

@app.post("/agent/invoke")
async def invoke_agent(request: AgentRequest):
    print(f"\n--- Agent Brain received new goal: '{request.goal}' ---")
    decision = agent.invoke(request.goal)

    # Prefer tool_calls (OpenAI-style), but handle function_call fallback if present
    tool_calls = decision.additional_kwargs.get("tool_calls")
    if not tool_calls:
        fn = decision.additional_kwargs.get("function_call")
        if not fn:
            print("Agent Brain: Decided no tool was necessary.")
            return {"final_answer": decision.content, "trace": {"decision": "No tool needed."}}
        tool_name = fn["name"]
        args_raw = fn.get("arguments") or "{}"
    else:
        first = tool_calls[0]
        tool_name = first["function"]["name"]
        args_raw = first["function"].get("arguments") or "{}"

    try:
        tool_args = json.loads(args_raw)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Malformed function/tool arguments from model: {e}")

    print(f"Agent Brain: Decided to use tool '{tool_name}' with arguments: {tool_args}")
    tool_result = mcp_client.execute_tool(tool_name=tool_name, tool_args=tool_args)
    print(f"Agent Brain: MCP Client returned result: {tool_result}")

    return {
        "final_answer": tool_result,
        "trace": {
            "agent_decision": {"tool_name": tool_name, "arguments": tool_args},
            "mcp_client_routing_result": tool_result,
        },
    }
