"""HTTP service: the same agent behind FastAPI, with SSE streaming and request tracing.

Run it with `uvicorn moneyrag.service:app --reload`, or `docker compose up`.

The model call is blocking, so handlers are plain `def` and FastAPI runs them in
its threadpool. SQLite connections are opened with `check_same_thread=False` for
the same reason.

Streaming: `POST /ask/stream` returns server-sent events. Token events are
provisional — the guard runs after the model finishes, and if it overrides the
answer a `replaced` event arrives before `done`. Clients should render `done`.
"""
from __future__ import annotations

import json
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
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


def create_app(state: dict | None = None, client=None, model: str | None = None) -> FastAPI:
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
        return Agent(c["retriever"], c["ledger"], c["memory"],
                     client=app.state.client, model=app.state.model)

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

    @app.get("/health")
    def health() -> dict:
        c = components()
        return {
            "status": "ok",
            "chunks": len(c["retriever"].chunks),
            "docs": len(list((DATA / "docs").glob("*.md"))),
            "facts": len(c["memory"].all()),
        }

    @app.post("/ask")
    def ask(req: AskRequest, request: Request) -> dict:
        return {"request_id": request.state.rid, **_payload(make_agent().ask(req.question))}

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest, request: Request) -> StreamingResponse:
        rid = request.state.rid

        def events():
            for ev in make_agent().ask_stream(req.question):
                payload = {"type": "done", "request_id": rid, **_payload(ev["result"])} \
                    if ev["type"] == "done" else ev
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"cache-control": "no-cache", "x-accel-buffering": "no"})

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
