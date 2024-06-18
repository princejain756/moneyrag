"""HTTP service: the same agent behind FastAPI, with SSE streaming and request tracing.

Run it with `uvicorn moneyrag.service:app --reload`, or `docker compose up`.

The model call is blocking, so handlers are plain `def` and FastAPI runs them in
its threadpool. SQLite connections are opened with `check_same_thread=False` for
the same reason.

Streaming: `POST /ask/stream` returns server-sent events. Token events are
provisional — the guard runs after the model finishes, and if it overrides the
answer a `replaced` event arrives before `done`. Clients should render `done`.

Public mode (`MONEYRAG_PUBLIC=1`) is for an open demo URL: every request gets
a fresh in-memory fact store, so one visitor can never write a fact that lands
in another visitor's system prompt, and the /facts routes are not mounted.
Rate limits and a daily cap bound what a public URL can spend.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel, Field

from .agent import Agent, Result
from .chunk import load_corpus
from .ledger import build_db
from .memory import Memory
from .retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = Path(os.environ.get("MONEYRAG_DATA", ROOT / "data"))
STATE = Path(os.environ.get("MONEYRAG_STATE", ROOT / ".state"))

log = logging.getLogger("moneyrag.service")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)


class Limiter:
    """Per-client sliding window plus one global daily cap. 0 disables either."""

    def __init__(self, per_minute: int = 0, per_day: int = 0, clock=time.time):
        self.per_minute, self.per_day, self.clock = per_minute, per_day, clock
        self.hits: dict[str, deque] = defaultdict(deque)
        self.day, self.day_count = "", 0
        self.lock = threading.Lock()

    def check(self, client: str) -> tuple[bool, str, int]:
        """Returns (allowed, reason, retry_after_seconds) and records the hit if allowed."""
        now = self.clock()
        with self.lock:
            today = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
            if today != self.day:
                self.day, self.day_count = today, 0
            if self.per_day and self.day_count >= self.per_day:
                return False, "daily demo budget used up, try again tomorrow (UTC)", 3600
            if self.per_minute:
                q = self.hits[client]
                while q and q[0] <= now - 60:
                    q.popleft()
                if len(q) >= self.per_minute:
                    return False, "too many questions, slow down", int(60 - (now - q[0])) + 1
                q.append(now)
            self.day_count += 1
            return True, "", 0


def _client_ip(request: Request, trust_proxy: bool) -> str:
    if trust_proxy:
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def build_state() -> dict:
    STATE.mkdir(parents=True, exist_ok=True)
    return {
        "retriever": Retriever(load_corpus(DATA / "docs"), cache_dir=STATE),
        "ledger": build_db(STATE / "ledger.db", check_same_thread=False),
        "memory": Memory(sqlite3.connect(STATE / "memory.db", check_same_thread=False)),
    }


def _payload(result: Result) -> dict:
    return {
        "question": result.question,
        "answer": result.answer,
        "refused": result.refused,
        "reason": result.reason,
        "citations": result.citations,
        "retrieved_ids": result.retrieved_ids,
        "ungrounded_numbers": result.ungrounded_numbers,
        "tool_calls": result.tool_calls,
        "steps": result.steps,
        "latency_s": result.latency_s,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
    }


def _env_int(name: str) -> int:
    return int(os.environ.get(name, "0") or 0)


def create_app(state: dict | None = None, client=None, model: str | None = None,
               public: bool | None = None, limiter: Limiter | None = None,
               trust_proxy: bool | None = None) -> FastAPI:
    public = os.environ.get("MONEYRAG_PUBLIC") == "1" if public is None else public
    trust_proxy = os.environ.get("MONEYRAG_TRUST_PROXY") == "1" if trust_proxy is None else trust_proxy
    limiter = limiter or Limiter(_env_int("MONEYRAG_RATE_PER_MIN"), _env_int("MONEYRAG_DAILY_CAP"))

    app = FastAPI(
        title="moneyrag",
        version="0.1.0",
        description="Personal finance RAG agent over bank policy docs and a transaction ledger. "
                    "Every answer is citation-checked and number-grounded; ungrounded answers become refusals.",
    )
    app.state.components = state
    app.state.client = client
    app.state.model = model

    def components() -> dict:
        if app.state.components is None:
            app.state.components = build_state()
        return app.state.components

    def make_agent() -> Agent:
        c = components()
        memory = Memory(sqlite3.connect(":memory:", check_same_thread=False)) if public else c["memory"]
        return Agent(c["retriever"], c["ledger"], memory, client=app.state.client, model=app.state.model)

    def admit(request: Request) -> None:
        ok, reason, retry = limiter.check(_client_ip(request, trust_proxy))
        if not ok:
            raise HTTPException(status_code=429, detail=reason, headers={"retry-after": str(retry)})

    @app.middleware("http")
    async def trace(request: Request, call_next):
        rid = request.headers.get("x-request-id") or secrets.token_hex(6)
        request.state.rid = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["x-request-id"] = rid
        log.info(json.dumps({
            "rid": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code,
            "ms": round((time.perf_counter() - t0) * 1000, 1),
        }))
        return response

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> str:
        return (Path(__file__).parent / "static" / "index.html").read_text()

    @app.get("/health")
    def health() -> dict:
        c = components()
        return {
            "status": "ok",
            "chunks": len(c["retriever"].chunks),
            "docs": len(list((DATA / "docs").glob("*.md"))),
            "public": public,
        }

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> dict:
        admit(request)
        return {"request_id": request.state.rid, **_payload(make_agent().ask(req.question))}

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        admit(request)
        rid = request.state.rid

        def events():
            for ev in make_agent().ask_stream(req.question):
                payload = {"type": "done", "request_id": rid, **_payload(ev["result"])} \
                    if ev["type"] == "done" else ev
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

    if not public:
        @app.get("/facts")
        def facts() -> dict:
            return components()["memory"].all()

        @app.delete("/facts/{key}")
        def forget(key: str) -> dict:
            if not components()["memory"].forget(key)["deleted"]:
                raise HTTPException(status_code=404, detail=f"no saved fact {key!r}")
            return {"deleted": key}

    return app


app = create_app()
