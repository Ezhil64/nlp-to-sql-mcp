# app.py - Hybrid Agent + Direct MCP client
# Query Tab: Uses TOOL interface (agent decides)
# View Tables Tab: Uses RESOURCE interface (direct fetch)
#
# HOW TO RUN:
#   pip install streamlit fastmcp pandas requests langsmith python-dotenv
#   streamlit run app.py

import os
import re
import json
import sys
import asyncio
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import requests

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from client_utils import MCPRouterClient

# =========================
# Config
# =========================

DEFAULT_BACKEND_URLS = os.getenv("BACKEND_URLS", "http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse")
DEFAULT_ROUTING_POLICY = os.getenv("ROUTING_POLICY", "round_robin")
AGENT_SERVER_URL = os.getenv("AGENT_SERVER_URL", "http://127.0.0.1:8002")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set in environment")
OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_LLM_MODEL = os.getenv("LLM_MODEL", "mistralai/mistral-7b-instruct")


# =========================
# LLM helpers (OpenRouter for SQL generation)
# =========================


def extract_sql_from_text(text: str) -> str:
    if not text:
        return ""

    # Remove model artifacts (Mistral instruction tokens, etc.)
    text = re.sub(r'\[/?[A-Z_]+\]', '', text)
    text = text.strip()

    # Try to extract from code block
    m = re.search(r"``````", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        sql = m.group(1).strip().strip(";")
        return re.sub(r'\[/?[A-Z_]+\]', '', sql).strip()

    # Try to find SQL keywords
    m2 = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*", text, flags=re.IGNORECASE | re.DOTALL)
    if m2:
        sql = m2.group(0).strip().strip(";")
        # Remove everything after model artifacts
        sql = re.split(r'\[/?[A-Z_]+\]', sql)[0].strip()
        return sql

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

st.set_page_config(page_title="MCP Multi‑Server NL→SQL", page_icon="🤖", layout="wide")
st.title("🤖 MCP Multi‑Server NL→SQL (PostgreSQL)")

# Connection state
if "router" not in st.session_state:
    st.session_state.router = None
if "discovery" not in st.session_state:
    st.session_state.discovery = []
if "use_agent" not in st.session_state:
    st.session_state.use_agent = True

with st.sidebar:
    st.header("⚙️ Configuration")

    # Execution mode toggle
    st.subheader("Execution Mode")
    mode = st.radio(
        "Choose execution mode:",
        ["🤖 Agent Mode (via Tool Gateway)", "⚡ Direct Mode (direct MCP)"],
        index=0 if st.session_state.use_agent else 1,
        help="Agent mode routes through Tool Gateway for centralized logging"
    )
    st.session_state.use_agent = (mode.startswith("🤖"))

    if st.session_state.use_agent:
        agent_url = st.text_input("Agent Server URL", value=AGENT_SERVER_URL)

    st.divider()
    st.header("MCP Servers")
    urls_text = st.text_input(
        "Backend SSE URLs (comma-separated)",
        value=DEFAULT_BACKEND_URLS,
        help="e.g., http://127.0.0.1:3101/sse,http://127.0.0.1:3102/sse",
    )
    policy = st.selectbox("Routing policy", ["round_robin", "latency", "first"])

    if st.button("Connect", type="primary", use_container_width=True):
        urls = [u.strip() for u in urls_text.split(",") if u.strip()]
        st.session_state.router = MCPRouterClient(urls, policy=policy)
        try:
            st.session_state.discovery = asyncio.run(st.session_state.router.discover())
            ok_count = sum(1 for d in st.session_state.discovery if d["ok"])
            st.success(f"✅ Connected: {ok_count}/{len(st.session_state.discovery)} servers healthy")
        except Exception as e:
            st.error(f"❌ Connect failed: {e}")

    openrouter_api_key = OPENROUTER_API_KEY
    llm_model = DEFAULT_LLM_MODEL

if not st.session_state.router:
    st.info("👈 Enter backend SSE URLs and click Connect to initialize routing")
    st.stop()

# Server status
with st.expander("📊 Server Status", expanded=False):
    if st.session_state.discovery:
        for d in st.session_state.discovery:
            status = "✅" if d["ok"] else "❌"
            resources = d.get("resources", [])
            res_display = f" | resources={len(resources)}" if resources else ""
            st.write(f"{status} {d['url']} | latency={d['latency_ms']}ms | tools={d['tools']}{res_display}")
    else:
        st.caption("No discovery data yet")

st.divider()

# Tabs
tab_query, tab_tables = st.tabs(["Query Data", "View Tables"])

# =======================
# TAB 1: Query Data
# Uses TOOL interface (unchanged)
# =======================
with tab_query:
    st.subheader("💬 Ask a Question")
    st.caption("🔧 Uses Tool interface via Agent")

    read_only = st.toggle("🔒 Read-Only (SELECT/WITH)", value=True,
                          help="Uncheck to allow writes (INSERT/UPDATE/DELETE)")

    q = st.text_area(
        "Natural language question",
        height=140,
        placeholder="e.g., Show employees with salary > 80000 ordered by salary desc"
    )

    include_schema = st.checkbox("Include schema in LLM context", value=True)

    if st.button("🚀 Generate & Execute", type="primary"):
        if not q.strip():
            st.error("Please enter a question")
        elif not openrouter_api_key.strip():
            st.error("Please provide OPENROUTER_API_KEY")
        else:
            # Step 1: Fetch schema (via TOOL interface)
            schema_text = ""
            try:
                if st.session_state.use_agent:
                    # Get schema via agent → tool
                    resp = requests.post(
                        f"{agent_url}/agent/invoke",
                        json={"goal": "list all tables and their columns"},
                        timeout=30
                    )
                    resp.raise_for_status()
                    result = resp.json()
                    trace = result.get("trace", {})
                    routing = trace.get("mcp_client_routing_result", {})
                    out = routing.get("result", {})
                else:
                    # Direct MCP tool call
                    r = st.session_state.router.route_call("get_tables", {})
                    out = r.get("result", {})

                if isinstance(out, dict) and "error" not in out and include_schema:
                    parts: List[str] = []
                    for sch, tables in out.items():
                        for tbl, cols in tables.items():
                            parts.append(f'{sch}.{tbl}({", ".join(cols)})')
                    schema_text = "\n".join(parts)
            except Exception as e:
                st.warning(f"⚠️ Schema unavailable: {e}")

            # Step 2: Generate SQL via OpenRouter
            with st.spinner("🧠 Generating SQL via LLM..."):
                try:
                    sql_text = nl_to_sql_openrouter(
                        api_key=openrouter_api_key.strip(),
                        question=q.strip(),
                        schema_text=schema_text,
                        read_only_mode=read_only,
                        model=llm_model.strip() or DEFAULT_LLM_MODEL,
                    )
                except Exception as e:
                    st.error(f"❌ LLM generation failed: {e}")
                    sql_text = ""

            if sql_text:
                st.subheader("📝 Generated SQL")

                # Auto-add RETURNING for write operations
                display_sql = sql_text
                if not read_only:
                    if re.search(r'\b(INSERT|UPDATE|DELETE)\b', sql_text, re.IGNORECASE):
                        if not re.search(r'\bRETURNING\b', sql_text, re.IGNORECASE):
                            display_sql = sql_text + " RETURNING *"
                            st.caption("ℹ️ Added RETURNING * to show affected rows")

                st.code(display_sql, language="sql")

                # Step 3: Execute SQL
                with st.spinner("⚡ Executing via MCP..."):
                    try:
                        args = {"sql": display_sql}  # Use SQL with RETURNING
                        if not read_only:
                            args["unrestricted"] = True

                        if st.session_state.use_agent:
                            resp = requests.post(
                                f"{agent_url}/agent/invoke",
                                json={"goal": f"execute this SQL query: {display_sql}" +
                                              ("" if read_only else " (with write permissions)")},
                                timeout=60
                            )
                            resp.raise_for_status()
                            result = resp.json()
                            trace = result.get("trace", {})
                            routing = trace.get("mcp_client_routing_result", {})
                            pick = routing.get("server", "via agent")
                            payload = routing.get("result", {})
                        else:
                            r2 = st.session_state.router.route_call("run_query", args)
                            pick = r2.get("server", "n/a")
                            payload = r2.get("result", {})

                        st.caption(f"📡 Server: {pick}")
                        st.subheader("📊 Results")

                        if isinstance(payload, dict) and "error" in payload:
                            st.error(f"❌ {payload['error']}")

                        elif isinstance(payload, dict) and payload.get("type") == "select":
                            rows = payload.get("rows", [])
                            cnt = payload.get("row_count", len(rows))
                            st.success(f"✅ Query successful: {cnt} rows")

                            if rows:
                                df = pd.DataFrame(rows)
                                st.dataframe(df, use_container_width=True)
                                csv = df.to_csv(index=False).encode("utf-8")
                                st.download_button("📥 Download CSV", csv, "results.csv", "text/csv")
                            else:
                                st.info("No rows returned")

                        elif isinstance(payload, dict) and payload.get("type") == "write":
                            affected = payload.get('affected_rows', '?')
                            st.success(f"✅ {payload.get('message', 'Write executed')}")
                            st.metric("Affected Rows", affected)

                            # NEW: Show returned rows from RETURNING clause
                            rows = payload.get("rows", [])
                            if rows:
                                st.subheader("📋 Modified Rows")
                                df = pd.DataFrame(rows)
                                st.dataframe(df, use_container_width=True)
                                csv = df.to_csv(index=False).encode("utf-8")
                                st.download_button("📥 Download Changes", csv, "changes.csv", "text/csv")
                            else:
                                st.info("No rows returned")

                        else:
                            st.json(payload)

                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Cannot connect to Agent Server at {agent_url}")
                    except Exception as e:
                        st.error(f"❌ Execution failed: {e}")
# =======================
# TAB 2: View Tables
# Uses RESOURCE interface (NEW - hybrid approach)
# =======================
with tab_tables:
    st.subheader("📋 Browse Tables")
    st.caption("🆕 Uses Resource interface (postgres://schema/tables) for direct access")

    col1, col2 = st.columns([1, 3])
    with col1:
        load_tables = st.button("📂 Load Tables", use_container_width=True)
    with col2:
        if st.button("🔄 Refresh Schema", use_container_width=True):
            try:
                st.session_state.discovery = asyncio.run(st.session_state.router.discover())
                st.success("Discovery refreshed")
            except Exception as e:
                st.error(f"Refresh failed: {e}")

    if load_tables or "tables_loaded" in st.session_state:
        st.session_state["tables_loaded"] = True

        try:
            with st.spinner("📦 Fetching schema via MCP Resource..."):
                r = st.session_state.router.read_resource("postgres://schema/tables")
                out = r.get("result", {})
                server_used = r.get("server", "unknown")

            st.caption(f"✅ Resource fetched from: {server_used}")

            if isinstance(out, dict) and "error" not in out and out:
                schemas = list(out.keys())
                sel_schema = st.selectbox("Schema", schemas,
                                          index=schemas.index("public") if "public" in schemas else 0)
                tables = sorted(list(out.get(sel_schema, {}).keys()))

                if tables:
                    col1, col2 = st.columns([2, 1])
                    with col1:
                        sel_table = st.selectbox("Table", tables)
                    with col2:
                        limit = st.number_input("Rows", min_value=10, max_value=10000, value=100, step=10)

                    if st.button(f"📊 Preview {sel_table}", use_container_width=True):
                        sql = f'SELECT * FROM "{sel_schema}"."{sel_table}" ORDER BY 1 DESC LIMIT {int(limit)}'

                        with st.spinner("Executing query..."):
                            if st.session_state.use_agent:
                                resp = requests.post(
                                    f"{agent_url}/agent/invoke",
                                    json={"goal": f"execute this SQL: {sql}"},
                                    timeout=30
                                )
                                resp.raise_for_status()
                                result = resp.json()
                                trace = result.get("trace", {})
                                routing = trace.get("mcp_client_routing_result", {})
                                payload = routing.get("result", {})
                                query_server = routing.get("server", "via agent")
                            else:
                                r2 = st.session_state.router.route_call("run_query", {"sql": sql})
                                payload = r2.get("result", {})
                                query_server = r2.get("server", "n/a")

                        st.caption(f"📡 Query executed on: {query_server}")

                        if isinstance(payload, dict) and "error" in payload:
                            st.error(payload["error"])
                        else:
                            rows = payload.get("rows", [])
                            if rows:
                                df = pd.DataFrame(rows)
                                st.success(f"Showing {len(df)} rows from {sel_schema}.{sel_table}")
                                st.dataframe(df, use_container_width=True)
                                csv = df.to_csv(index=False).encode("utf-8")
                                st.download_button(f"📥 Download {sel_table}.csv", csv, f"{sel_table}.csv", "text/csv")
                            else:
                                st.info("No rows found")
            else:
                st.info("No tables found or schema fetch failed")

        except Exception as e:
            st.error(f"Failed to fetch schema via Resource: {e}")
            st.info("💡 Tip: Make sure MCP servers expose 'postgres://schema/tables' resource")
    else:
        st.info("👆 Click 'Load Tables' to browse database schema")
