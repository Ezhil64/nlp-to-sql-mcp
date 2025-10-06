# app.py
# Streamlit MCP client with automated routing/failover across multiple Postgres MCP servers (SSE).
#
# HOW TO RUN:
#   pip install streamlit fastmcp pandas requests langsmith
#   # Start 2+ mcp_server.py processes on ports 3101, 3102 ...
#   # Launch Streamlit
#   streamlit run app.py
#
# Sidebar:
# - Enter comma-separated MCP SSE URLs (e.g., http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse)
# - Connect to discover and route automatically

import os
import re
import json
import sys
import asyncio
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st
import pandas as pd
import requests

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from client_utils import MCPRouterClient  # local module for routing

# =========================
# Config
# =========================

DEFAULT_BACKEND_URLS = os.getenv("BACKEND_URLS", "http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse")
DEFAULT_ROUTING_POLICY = os.getenv("ROUTING_POLICY", "round_robin")  # first | round_robin | latency

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "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")
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct:free")

# =========================
# LLM helpers (OpenRouter)
# =========================

def extract_sql_from_text(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"``````", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip().strip(";")
    m2 = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*", text, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        return m2.group(0).strip().strip(";")
    return text.strip().strip(";")

def nl_to_sql_openrouter(
    api_key: str,
    question: str,
    schema_text: str,
    read_only_mode: bool,
    model: str = DEFAULT_LLM_MODEL,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost/",
        "X-Title": "MCP NL2SQL",
    }
    if read_only_mode:
        system_prompt = (
            "Convert the user's question into a single, safe PostgreSQL SQL query that is strictly read-only.\n"
            "- Use only SELECT or WITH (CTE).\n"
            "- No DDL/DML (INSERT/UPDATE/DELETE/CREATE/ALTER/TRUNCATE/DROP/GRANT/REVOKE/etc.).\n"
            "- Use only tables/columns from provided schema.\n"
            "- Return only the SQL, no explanations.\n"
        )
    else:
        system_prompt = (
            "Convert the user's question into a PostgreSQL SQL query.\n"
            "- You may use SELECT/INSERT/UPDATE/DELETE.\n"
            "- Use only tables/columns from provided schema.\n"
            "- Return only the SQL, no explanations.\n"
            "- For write queries, prefer RETURNING to show affected rows.\n"
        )
    user_prompt = f"Database Schema:\n{schema_text}\n\nQuestion:\n{question}\n\nReturn only the SQL query with no explanation. PostgreSQL dialect."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(OPENROUTER_CHAT_COMPLETIONS_URL, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    return extract_sql_from_text(content)

# =========================
# Streamlit UI
# =========================

st.set_page_config(page_title="MCP Multi‑Server NL→SQL", page_icon="🧩", layout="wide")
st.title("🧩 MCP Multi‑Server NL→SQL (PostgreSQL via SSE)")

# Connection state
if "router" not in st.session_state:
    st.session_state.router = None
if "discovery" not in st.session_state:
    st.session_state.discovery = []

with st.sidebar:
    st.header("MCP Servers")
    urls_text = st.text_input(
        "Backend SSE URLs (comma‑separated)",
        value=DEFAULT_BACKEND_URLS,
        help="e.g., http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse",
    )
    policy = st.selectbox("Routing policy", ["round_robin", "latency", "first"], index=["round_robin","latency","first"].index(DEFAULT_ROUTING_POLICY))
    if st.button("Connect", type="primary", use_container_width=True):
        urls = [u.strip() for u in urls_text.split(",") if u.strip()]
        st.session_state.router = MCPRouterClient(urls, policy=policy)
        try:
            st.session_state.discovery = asyncio.run(st.session_state.router.discover())
            ok_count = sum(1 for d in st.session_state.discovery if d["ok"])
            st.success(f"Connected: {ok_count}/{len(st.session_state.discovery)} servers healthy.")
        except Exception as e:
            st.error(f"Connect failed: {e}")

    st.divider()
    st.header("LLM (OpenRouter)")
    openrouter_api_key = st.text_input("OPENROUTER_API_KEY", value=OPENROUTER_API_KEY, type="password")
    llm_model = st.text_input("Model", value=DEFAULT_LLM_MODEL)

if not st.session_state.router:
    st.info("Enter backend SSE URLs and click Connect to initialize routing.")
    st.stop()

# Discovery view
st.subheader("Server status")
if st.session_state.discovery:
    for d in st.session_state.discovery:
        st.write(f"- {d['url']} | ok={d['ok']} | latency={d['latency_ms']} ms | tools={d['tools']} | err={d['error']}")
else:
    st.caption("No discovery data yet; press Connect to scan servers.")

st.divider()

# Tabs
tab_query, tab_tables = st.tabs(["Query Data", "View Tables"])

with tab_query:
    st.subheader("Ask a question")
    read_only = st.toggle("Read‑Only (SELECT/WITH)", value=True, help="Uncheck to allow writes (INSERT/UPDATE/DELETE) on the server that supports it.")
    q = st.text_area("Natural language question", height=140, placeholder="e.g., Show employees with salary > 80000 ordered by salary desc")
    include_schema = st.checkbox("Include schema to the LLM", value=True)
    if st.button("Generate & Execute", type="primary"):
        if not q.strip():
            st.error("Please enter a question.")
        elif not openrouter_api_key.strip():
            st.error("Please provide OPENROUTER_API_KEY.")
        else:
            # Fetch schema via routed get_tables
            schema_text = ""
            try:
                r = st.session_state.router.route_call("get_tables", {})
                out = r.get("result", {})
                if isinstance(out, dict) and "error" not in out and include_schema:
                    parts: List[str] = []
                    for sch, tables in out.items():
                        for tbl, cols in tables.items():
                            parts.append(f'{sch}.{tbl}({", ".join(cols)})')
                    schema_text = "\n".join(parts)
            except Exception as e:
                st.warning(f"Schema unavailable: {e}")
            # Generate SQL
            with st.spinner("Generating SQL via LLM..."):
                try:
                    sql_text = nl_to_sql_openrouter(
                        api_key=openrouter_api_key.strip(),
                        question=q.strip(),
                        schema_text=schema_text,
                        read_only_mode=read_only,
                        model=llm_model.strip() or DEFAULT_LLM_MODEL,
                    )
                except Exception as e:
                    st.error(f"LLM generation failed: {e}")
                    sql_text = ""
            if sql_text:
                st.subheader("Generated SQL")
                st.code(sql_text, language="sql")
                # Execute routed run_query
                with st.spinner("Executing via routed MCP..."):
                    try:
                        args = {"sql": sql_text}
                        if not read_only:
                            args["unrestricted"] = True
                        r2 = st.session_state.router.route_call("run_query", args)
                        pick = r2.get("server", "n/a")
                        st.caption(f"Server selected: {pick}")
                        payload = r2.get("result", {})
                        if isinstance(payload, dict) and "error" in payload:
                            st.error(payload["error"])
                        elif isinstance(payload, dict) and payload.get("type") == "select":
                            rows = payload.get("rows", [])
                            cnt = payload.get("row_count", len(rows))
                            st.success(f"OK, rows: {cnt}")
                            if rows:
                                df = pd.DataFrame(rows)
                                st.dataframe(df, use_container_width=True)
                                st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"), "results.csv", "text/csv")
                            else:
                                st.info("No rows returned.")
                        elif isinstance(payload, dict) and payload.get("type") == "write":
                            st.success(payload.get("message", "Write executed."))
                            st.info(f"Affected rows: {payload.get('affected_rows','?')}")
                        else:
                            st.write(payload)
                    except Exception as e:
                        st.error(f"Execution failed: {e}")

with tab_tables:
    st.subheader("Browse tables")
    if st.button("Refresh servers"):
        try:
            st.session_state.discovery = asyncio.run(st.session_state.router.discover())
            st.success("Discovery refreshed.")
        except Exception as e:
            st.error(f"Refresh failed: {e}")
    try:
        r = st.session_state.router.route_call("get_tables", {})
        out = r.get("result", {})
        if isinstance(out, dict) and "error" not in out and out:
            schemas = list(out.keys())
            sel_schema = st.selectbox("Schema", schemas, index=schemas.index("public") if "public" in schemas else 0)
            tables = sorted(list(out.get(sel_schema, {}).keys()))
            if tables:
                col1, col2 = st.columns([2, 1])
                with col1:
                    sel_table = st.selectbox("Table", tables)
                with col2:
                    limit = st.number_input("Rows", min_value=10, max_value=10000, value=100, step=10)
                if sel_table:
                    sql = f'SELECT * FROM "{sel_schema}"."{sel_table}" ORDER BY 1 DESC LIMIT {int(limit)}'
                    r2 = st.session_state.router.route_call("run_query", {"sql": sql})
                    payload = r2.get("result", {})
                    if isinstance(payload, dict) and "error" in payload:
                        st.error(payload["error"])
                    else:
                        rows = payload.get("rows", [])
                        if rows:
                            df = pd.DataFrame(rows)
                            st.success(f"Showing {len(df)} rows from {sel_schema}.{sel_table}")
                            st.dataframe(df, use_container_width=True)
                            st.download_button(f"Download {sel_table}.csv", df.to_csv(index=False).encode("utf-8"), f"{sel_table}.csv", "text/csv")
                        else:
                            st.info("No rows found.")
        else:
            st.info("No tables found or schema fetch failed.")
    except Exception as e:
        st.error(f"Failed to fetch tables: {e}")
