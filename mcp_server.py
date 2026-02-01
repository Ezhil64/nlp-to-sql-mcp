# mcp_server.py
# FastMCP PostgreSQL MCP server (SSE) exposing:
# - TOOLS: run_query, get_tables (for agents)
# - RESOURCES: postgres://schema/tables (for direct UI access)

import os, re, sys, asyncio, json
from typing import Any, Dict, List, Optional, Tuple
import psycopg2
from psycopg2 import sql as psql
from psycopg2.extras import RealDictCursor
from fastmcp import FastMCP

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Cvhs@12345")
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
STATEMENT_TIMEOUT_MS = int(os.getenv("STATEMENT_TIMEOUT_MS", "15000"))
DEFAULT_LIMIT = int(os.getenv("DEFAULT_SELECT_LIMIT", "1000"))

MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "3101"))
MCP_NAME = os.getenv("MCP_NAME", f"Postgres MCP {MCP_PORT}")

SEED_EMPLOYEES = os.getenv("SEED_EMPLOYEES", "true").lower() in ("1", "true", "yes")
EMP_TABLE = os.getenv("EMP_TABLE", "employees")


def is_select_only(sql_text: str) -> bool:
    if not sql_text: return False
    s = sql_text.strip().strip(";")
    if ";" in s: return False
    if not re.match(r"^\s*(SELECT|WITH)\b", s, flags=re.IGNORECASE): return False
    if re.search(
            r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|VACUUM|ANALYZE|COMMENT|MERGE)\b",
            s, flags=re.IGNORECASE):
        return False
    return True


def is_dangerous_sql(sql_text: str) -> bool:
    if not sql_text: return False
    s = sql_text.upper().strip()
    return any(x in s for x in ["DROP DATABASE", "DROP SCHEMA", "TRUNCATE"])


def enforce_limit(sql_text: str, default_limit: int = DEFAULT_LIMIT) -> str:
    s = sql_text.strip().rstrip(";")
    if not re.match(r"^\s*SELECT\b", s, flags=re.IGNORECASE): return s
    if re.search(r"\bLIMIT\s+\d+\b", s, flags=re.IGNORECASE): return s
    return f"{s} LIMIT {default_limit}"


def get_connection(read_only: bool = True):
    opts = f"-c statement_timeout={STATEMENT_TIMEOUT_MS}"
    if read_only: opts = f"-c default_transaction_read_only=on " + opts
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD,
                            options=opts)
    conn.autocommit = True
    return conn


def get_tables_and_columns() -> List[Tuple[str, str, str]]:
    q = """
        SELECT c.table_schema, c.table_name, c.column_name
        FROM information_schema.columns c
        WHERE c.table_schema = %s
        ORDER BY c.table_schema, c.table_name, c.ordinal_position \
        """
    with get_connection(read_only=True) as conn:
        with conn.cursor() as cur:
            cur.execute(q, (DB_SCHEMA,))
            return cur.fetchall()


def run_sql(sql_text: str, allow_writes: bool) -> Dict[str, Any]:
    if is_dangerous_sql(sql_text):
        raise ValueError("Extremely dangerous SQL detected.")
    if not allow_writes and not is_select_only(sql_text):
        raise ValueError("Read-only mode: only SELECT/WITH are allowed.")

    final_sql = enforce_limit(sql_text)

    with get_connection(read_only=not allow_writes) as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(final_sql)

            if cur.description:
                # Has results (SELECT or INSERT/UPDATE/DELETE with RETURNING)
                rows = cur.fetchall()
                cols = [d.name for d in cur.description]

                # Check if this was a write operation with RETURNING
                is_write_with_returning = re.search(
                    r'\b(INSERT|UPDATE|DELETE)\b.*\bRETURNING\b',
                    final_sql,
                    re.IGNORECASE
                )

                if is_write_with_returning:
                    return {
                        "type": "write",
                        "affected_rows": len(rows),
                        "message": f"OK, {len(rows)} row(s) affected",
                        "columns": cols,
                        "rows": [dict(r) for r in rows]
                    }
                else:
                    return {
                        "type": "select",
                        "columns": cols,
                        "rows": [dict(r) for r in rows],
                        "row_count": len(rows)
                    }

            # Write without RETURNING
            return {
                "type": "write",
                "affected_rows": cur.rowcount,
                "message": f"OK, {cur.rowcount} row(s) affected."
            }


def qualified_table(schema: str, table: str) -> psql.SQL:
    return psql.SQL("{}.{}").format(psql.Identifier(schema), psql.Identifier(table))


def ensure_employees_table(table_name: Optional[str] = None) -> None:
    schema = DB_SCHEMA
    table = table_name or EMP_TABLE
    create_sql = psql.SQL("""
                          CREATE TABLE IF NOT EXISTS {}
                          (
                              id
                              BIGSERIAL
                              PRIMARY
                              KEY,
                              name
                              TEXT
                              NOT
                              NULL,
                              salary
                              NUMERIC
                          (
                              12,
                              2
                          ) NOT NULL,
                              age INTEGER NOT NULL,
                              created_at TIMESTAMPTZ DEFAULT NOW
                          (
                          )
                              )
                          """).format(qualified_table(schema, table))
    with get_connection(read_only=False) as conn:
        with conn.cursor() as cur:
            cur.execute(create_sql)


def seed_employees_if_empty(table_name: Optional[str] = None) -> None:
    schema = DB_SCHEMA
    table = table_name or EMP_TABLE
    count_sql = psql.SQL("SELECT COUNT(*) FROM {}").format(qualified_table(schema, table))
    with get_connection(read_only=False) as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql)
            count = cur.fetchone()[0]
            if count > 0: return
            rows = [("Alice Johnson", 90000.00, 30), ("Bob Smith", 75000.50, 28), ("Carol Lee", 120000.00, 41)]
            insert_sql = psql.SQL("INSERT INTO {} (name, salary, age) VALUES (%s,%s,%s)").format(
                qualified_table(schema, table))
            cur.executemany(insert_sql, rows)


def _fetch_schema_dict() -> Dict[str, Dict[str, List[str]]]:
    """Single source of truth for schema data"""
    triples = get_tables_and_columns()
    out: Dict[str, Dict[str, List[str]]] = {}
    for sch, tbl, col in triples:
        out.setdefault(sch, {}).setdefault(tbl, []).append(col)
    return out


mcp = FastMCP(MCP_NAME)


@mcp.resource(
    uri="postgres://schema/tables",
    name="Database Schema",
    description="Read-only database table and column metadata",
    mime_type="application/json"
)
async def schema_resource() -> str:
    """MCP Resource interface for schema"""
    try:
        schema = _fetch_schema_dict()
        return json.dumps({
            "uri": "postgres://schema/tables",
            "mimeType": "application/json",
            "text": json.dumps(schema, indent=2)
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@mcp.tool(name="get_tables", description="Get database schema tables and columns (read-only)")
def get_tables() -> Dict[str, Any]:
    """MCP Tool interface for schema"""
    try:
        return _fetch_schema_dict()
    except Exception as e:
        return {" error": str(e)}


@mcp.tool(name="run_query", description="Execute SQL; unrestricted=True permits writes.")
def run_query(sql: str, unrestricted: Optional[bool] = None) -> Dict[str, Any]:
    """Execute SQL with safety checks"""
    try:
        allow_writes = bool(unrestricted) if unrestricted is not None else False
        return run_sql(sql, allow_writes=allow_writes)
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    if SEED_EMPLOYEES:
        try:
            ensure_employees_table(EMP_TABLE)
            seed_employees_if_empty(EMP_TABLE)
        except Exception as se:
            print(f"[warn] seed failed: {se}")
    print(f"Starting {MCP_NAME} at http://{MCP_HOST}:{MCP_PORT}/sse")
    mcp.run(transport="sse", host=MCP_HOST, port=MCP_PORT)
