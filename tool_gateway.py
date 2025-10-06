# tool_gateway.py
# Consolidated Tool Server (FastAPI) exposing REST endpoints that forward to MCP servers over SSE.
# pip install fastapi uvicorn fastmcp pydantic

import os, sys, asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from fastmcp import Client  # FastMCP Python client

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

BACKEND_URLS = [u.strip() for u in os.getenv("BACKEND_URLS","http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse").split(",") if u.strip()]
TIMEOUT = float(os.getenv("MCP_CLIENT_TIMEOUT","30"))

app = FastAPI(title="Tool Gateway (REST → MCP)")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

async def call_first_healthy(tool: str, args: Dict[str, Any]) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for url in BACKEND_URLS:
        try:
            async with Client(url) as c:
                await c.ping()
                res = await asyncio.wait_for(c.call_tool(tool, args), timeout=TIMEOUT)
                data = getattr(res, "data", None) or getattr(res, "result", None) or res
                return {"server": url, "result": data}
        except Exception as e:
            last_err = str(e)
            continue
    raise HTTPException(status_code=503, detail=f"All MCP servers failed for {tool}: {last_err}")

class RunQueryIn(BaseModel):
    sql: str = Field(..., description="PostgreSQL SQL to execute.")
    unrestricted: Optional[bool] = Field(default=False, description="Allow writes when true.")

@app.post("/tools/run-query")
async def run_query_endpoint(payload: RunQueryIn):
    return await call_first_healthy("run_query", payload.model_dump())

@app.post("/tools/get-tables")
async def get_tables_endpoint():
    return await call_first_healthy("get_tables", {})
