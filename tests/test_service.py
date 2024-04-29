"""HTTP layer: routes, SSE streaming, request tracing. No network, no model download."""
import json
from types import SimpleNamespace as NS

import pytest
from fastapi.testclient import TestClient

from moneyrag.service import create_app


class ScriptedClient:
    """Fake OpenAI-compatible client serving both blocking and streaming calls."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.calls.append(kw)
        step = self.script.pop(0)
        if kw.get("stream"):
            return self._stream(step)
        if isinstance(step, str):
            return NS(choices=[NS(message=NS(content=step, tool_calls=None))],
                      usage=NS(prompt_tokens=11, completion_tokens=7))
        calls = [NS(index=i, id=f"c{i}", function=NS(name=n, arguments=json.dumps(a)))
                 for i, (n, a) in enumerate(step)]
        return NS(choices=[NS(message=NS(content="", tool_calls=calls))],
                  usage=NS(prompt_tokens=11, completion_tokens=7))

    @staticmethod
    def _stream(step):
        if isinstance(step, str):
            for i in range(0, len(step), 10):
                yield NS(choices=[NS(delta=NS(content=step[i:i + 10], tool_calls=None))])
            return
        for i, (name, args) in enumerate(step):
            raw = json.dumps(args)
            # A real provider splits name and arguments across deltas; mirror that.
            yield NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=i, id=f"c{i}", function=NS(name=name, arguments=raw[:5]))]))])
            yield NS(choices=[NS(delta=NS(content=None, tool_calls=[
                NS(index=i, id=None, function=NS(name=None, arguments=raw[5:]))]))])


def sse_events(text):
    return [json.loads(line[len("data: "):]) for line in text.splitlines() if line.startswith("data: ")]


def make(state, script):
    return TestClient(create_app(state=state, client=ScriptedClient(script)))


def test_health_reports_corpus_size(state):
    r = make(state, []).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok" and r.json()["chunks"] == 59


def test_ask_returns_grounded_answer_and_request_id(state):
    app = make(state, [[("search_docs", {"query": "late fee credit card"})],
                       "Rs 500 for a 12,000 bill [04_credit_card#late-payment-fee]."])
    r = app.post("/ask", json={"question": "Late fee on a 12,000 bill?"})
    assert r.status_code == 200
    body = r.json()
    assert body["refused"] is False
    assert body["citations"] == ["04_credit_card#late-payment-fee"]
    assert body["tool_calls"] == [{"name": "search_docs", "args": {"query": "late fee credit card"}}]
    assert r.headers["x-request-id"] == body["request_id"]


def test_ask_echoes_caller_supplied_request_id(state):
    app = make(state, [[("search_docs", {"query": "late fee"})], "Rs 500 [04_credit_card#late-payment-fee]."])
    r = app.post("/ask", json={"question": "late fee?"}, headers={"x-request-id": "abc123"})
    assert r.headers["x-request-id"] == "abc123" and r.json()["request_id"] == "abc123"


def test_ask_refuses_hallucinated_citation(state):
    app = make(state, [[("search_docs", {"query": "late fee credit card"})],
                       "Gold loans are 9% [11_gold_loan#rates]."])
    body = app.post("/ask", json={"question": "gold loan rate?"}).json()
    assert body["refused"] is True and body["reason"] == "invalid_citation"


def test_ask_rejects_empty_question(state):
    assert make(state, []).post("/ask", json={"question": ""}).status_code == 422


def test_stream_emits_progress_tokens_and_done(state):
    app = make(state, [[("search_docs", {"query": "late fee credit card"})],
                       "Rs 500 for a 12,000 bill [04_credit_card#late-payment-fee]."])
    r = app.post("/ask/stream", json={"question": "Late fee?"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    events = sse_events(r.text)
    kinds = [e["type"] for e in events]
    assert kinds[0] == "step"
    assert "tool_call" in kinds and "tool_result" in kinds and "token" in kinds
    assert kinds[-1] == "done"
    assert events[-1]["refused"] is False
    assert "".join(e["text"] for e in events if e["type"] == "token").startswith("Rs 500")


def test_stream_reports_replacement_when_guard_overrides(state):
    app = make(state, [[("search_docs", {"query": "late fee credit card"})],
                       "Trust me, it's fine."])
    events = sse_events(app.post("/ask/stream", json={"question": "late fee?"}).text)
    replaced = [e for e in events if e["type"] == "replaced"]
    assert replaced and replaced[0]["reason"] == "uncited"
    assert events[-1]["type"] == "done" and events[-1]["refused"] is True


def test_facts_read_and_delete(state):
    state["memory"].remember("city", "Pune")
    app = make(state, [])
    assert app.get("/facts").json() == {"city": "Pune"}
    assert app.delete("/facts/city").json() == {"deleted": "city"}
    assert app.get("/facts").json() == {}
    assert app.delete("/facts/city").status_code == 404
