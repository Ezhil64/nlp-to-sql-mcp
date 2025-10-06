# client_utils.py
# Async router/client utilities for multiple MCP servers over SSE.
# Features:
# - Discover servers and tools with Client.list_tools()
# - Health check & latency tracking
# - Routing policies: first | round_robin | latency
# - Automatic failover: try next server on error/timeouts
# - Optional LangSmith tracing via @traceable (set LANGSMITH_TRACING=true and LANGSMITH_API_KEY)

import os
import sys
import time
import asyncio
from typing import Any, Dict, List, Optional

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastmcp import Client  # FastMCP Python client [web:2]
try:
    from langsmith import traceable  # Optional tracing [web:98]
except Exception:
    def traceable(*args, **kwargs):
        def deco(fn): return fn
        return deco

DEFAULT_BACKEND_URLS = [u.strip() for u in os.getenv("BACKEND_URLS", "").split(",") if u.strip()]
DEFAULT_ROUTING_POLICY = os.getenv("ROUTING_POLICY", "round_robin")  # first | round_robin | latency
DEFAULT_TIMEOUT = float(os.getenv("MCP_CLIENT_TIMEOUT", "30"))

class ServerInfo:
    def __init__(self, url: str):
        self.url = url
        self.latency_ms: float = float("inf")
        self.ok: bool = False
        self.last_error: Optional[str] = None
        self.tools: List[str] = []

class MCPRouterClient:
    def __init__(self, urls: Optional[List[str]] = None, policy: str = DEFAULT_ROUTING_POLICY, timeout: float = DEFAULT_TIMEOUT):
        urls = urls if urls is not None else DEFAULT_BACKEND_URLS
        self.urls = urls[:]
        self.policy = policy
        self.timeout = timeout
        self.servers: List[ServerInfo] = [ServerInfo(u) for u in self.urls]
        self._rr_index = 0

    async def _measure_latency_and_tools(self, si: ServerInfo) -> None:
        t0 = time.perf_counter()
        try:
            async with Client(si.url) as c:
                await c.ping()  # connectivity check [web:2]
                tools = await c.list_tools()  # enumerate tools [web:39]
                names = [t.name if hasattr(t, "name") else t.get("name") for t in tools]
                si.tools = [n for n in names if n]
                si.ok = True
                si.last_error = None
        except Exception as e:
            si.ok = False
            si.tools = []
            si.last_error = str(e)
        finally:
            si.latency_ms = (time.perf_counter() - t0) * 1000.0

    @traceable(run_type="tool", name="discover_servers")  # logs discovery scans [web:98]
    async def discover(self) -> List[Dict[str, Any]]:
        tasks = [self._measure_latency_and_tools(si) for si in self.servers]
        await asyncio.gather(*tasks)
        return [dict(url=s.url, ok=s.ok, latency_ms=round(s.latency_ms, 2), tools=s.tools, error=s.last_error) for s in self.servers]

    def _choose_candidates(self, tool: str) -> List[ServerInfo]:
        candidates = [s for s in self.servers if s.ok and (not tool or tool in s.tools)]
        if not candidates:
            candidates = [s for s in self.servers if s.ok]
        return candidates

    def _pick_server(self, tool: str) -> Optional[ServerInfo]:
        cands = self._choose_candidates(tool)
        if not cands:
            return None
        if self.policy == "first":
            return cands[0]
        if self.policy == "latency":
            return sorted(cands, key=lambda s: s.latency_ms)[0]
        si = cands[self._rr_index % len(cands)]
        self._rr_index += 1
        return si

    @traceable(run_type="tool", name="route_call")  # logs inputs/outputs/errors [web:98]
    async def route_call_async(self, tool: str, arguments: Dict[str, Any]) -> Any:
        if not any(s.ok for s in self.servers):
            await self.discover()
        tried: List[str] = []
        ordered = []
        first_pick = self._pick_server(tool)
        if first_pick:
            ordered.append(first_pick)
        for s in self._choose_candidates(tool):
            if not first_pick or s.url != first_pick.url:
                ordered.append(s)
        last_exc: Optional[Exception] = None
        for si in ordered:
            tried.append(si.url)
            try:
                async with Client(si.url) as c:
                    res = await asyncio.wait_for(c.call_tool(tool, arguments or {}), timeout=self.timeout)  # call a tool [web:2]
                    data = getattr(res, "data", None) or getattr(res, "result", None) or res
                    return {"ok": True, "server": si.url, "result": data, "latency_ms": si.latency_ms, "policy": self.policy}
            except Exception as e:
                si.ok = False
                si.last_error = str(e)
                last_exc = e
                continue
        raise RuntimeError(f"All servers failed for tool '{tool}', tried={tried}, last_error={last_exc}")

    def route_call(self, tool: str, arguments: Dict[str, Any]) -> Any:
        return asyncio.run(self.route_call_async(tool, arguments))
