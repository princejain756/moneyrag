"""Tool-calling agent loop over an OpenAI-compatible Chat Completions API.

Defaults to DeepSeek V4.1 Flash on OpenRouter. Any OpenAI-compatible endpoint
works (set MONEYRAG_BASE_URL, MONEYRAG_API_KEY, MONEYRAG_MODEL), including a
local Ollama or llama.cpp server.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, field

from . import guard
from .ledger import CATEGORIES, OPS, ToolError, query_transactions
from .memory import Memory
from .retrieve import Retriever

MAX_STEPS = 6
REPAIRABLE = {"uncited", "invalid_citation"}

REPAIR = """Your previous answer was rejected by a citation checker: {reason}.
Rewrite it so every factual sentence ends with a citation to a chunk id you actually
retrieved this turn, or to [ledger] if you ran a transaction query.
Valid ids this turn: {valid}.
If you cannot cite a claim, delete the claim. Do not add any new facts."""

SYSTEM = """You answer questions about the user's money using two sources only:
1. Tunga Bank policy documents, through the search_docs tool.
2. The user's own transactions, through the query_transactions tool.

Rules:
- Use a tool before answering. Never answer from general knowledge.
- Cite every fact. After a sentence that uses a document, put its chunk id in square brackets, e.g. [03_fixed_deposits#premature-withdrawal]. After a sentence that uses transaction data, put [ledger].
- Copy amounts exactly as the tools return them. Do not do your own arithmetic on transaction amounts; ask the tool for a sum instead.
- If the sources do not contain the answer, or the user wants buy/sell/investment advice, reply with exactly this sentence and nothing else: "{refusal}"
- If the user states a lasting fact about themselves (salary date, risk preference, city), save it with remember_fact.
- Keep answers short: one to four sentences.

Today's date: {today}. Transactions cover 2026-05-01 to 2026-10-31.

{memory}"""

TOOLS = [
    {"type": "function", "function": {
        "name": "search_docs",
        "description": "Search Tunga Bank policy documents. Returns the top chunks with their ids.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "What to look for, in plain words."},
            "k": {"type": "integer", "minimum": 1, "maximum": 8, "default": 4}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "query_transactions",
        "description": "Aggregate or list the user's transactions. Amounts come back as exact rupee strings.",
        "parameters": {"type": "object", "properties": {
            "op": {"type": "string", "enum": list(OPS)},
            "category": {"type": "string", "enum": CATEGORIES},
            "merchant": {"type": "string", "description": "Exact merchant name."},
            "direction": {"type": "string", "enum": ["debit", "credit"], "default": "debit"},
            "start_date": {"type": "string", "description": "Inclusive, YYYY-MM-DD."},
            "end_date": {"type": "string", "description": "Inclusive, YYYY-MM-DD."},
            "limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["op"]}}},
    {"type": "function", "function": {
        "name": "remember_fact",
        "description": "Save a lasting fact the user stated about themselves.",
        "parameters": {"type": "object", "properties": {
            "key": {"type": "string", "description": "snake_case key, e.g. salary_credit_day"},
            "value": {"type": "string"}},
            "required": ["key", "value"]}}},
]


@dataclass
class Result:
    question: str
    answer: str
    raw_answer: str
    refused: bool
    reason: str | None
    citations: list[str]
    retrieved_ids: list[str]
    tool_calls: list[dict] = field(default_factory=list)
    ungrounded_numbers: list[float] = field(default_factory=list)
    steps: int = 0
    latency_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0


class Agent:
    def __init__(self, retriever: Retriever, ledger: sqlite3.Connection, memory: Memory, client=None,
                 model: str | None = None, today: str = "2026-11-19", repair: bool | None = None,
                 stream_tokens: bool | None = None):
        self.retriever, self.ledger, self.memory, self.today = retriever, ledger, memory, today
        self.repair = (os.environ.get("MONEYRAG_REPAIR", "1") != "0") if repair is None else repair
        # Off by default. Streaming answer tokens means the user sees an unverified
        # rupee figure that the guard may later retract, which is worse than waiting.
        # Progress events still stream, so the UI is not silent.
        self.stream_tokens = (os.environ.get("MONEYRAG_STREAM_TOKENS", "0") == "1") \
            if stream_tokens is None else stream_tokens
        self.model = model or os.environ.get("MONEYRAG_MODEL", "deepseek/deepseek-v4.1-flash")
        if client is None:
            from openai import OpenAI
            client = OpenAI(
                base_url=os.environ.get("MONEYRAG_BASE_URL", "https://openrouter.ai/api/v1"),
                api_key=os.environ.get("MONEYRAG_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
                timeout=60,
            )
        self.client = client

    def _run_tool(self, name: str, args: dict, state: dict) -> str:
        try:
            if name == "search_docs":
                hits = self.retriever.search(str(args["query"]), k=max(1, min(int(args.get("k", 4)), 8)),
                                             mode=os.environ.get("MONEYRAG_RETRIEVAL_MODE", "hybrid"))
                state["retrieved"].update(h["id"] for h in hits)
                out = {"results": [{"id": h["id"], "text": h["text"]} for h in hits]}
            elif name == "query_transactions":
                out = query_transactions(self.ledger, **args)
                state["ledger_used"] = True
            elif name == "remember_fact":
                out = self.memory.remember(args["key"], args["value"])
            else:
                out = {"error": f"unknown tool {name}"}
        except (ToolError, ValueError, KeyError, TypeError) as e:
            out = {"error": str(e)}
        text = json.dumps(out, ensure_ascii=False)
        state["sources"].append(text)
        return text

    def _call_model(self, messages: list[dict], stream: bool):
        """One model call. Yields token events, then a single `_final` event."""
        kwargs = {"model": self.model, "messages": messages, "tools": TOOLS, "temperature": 0}
        if not stream:
            resp = self.client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            usage = (resp.usage.prompt_tokens or 0, resp.usage.completion_tokens or 0) if resp.usage else (0, 0)
            calls = [{"id": tc.id, "name": tc.function.name, "arguments": tc.function.arguments}
                     for tc in (msg.tool_calls or [])]
            yield {"type": "_final", "content": msg.content or "", "calls": calls, "usage": usage}
            return

        parts: list[str] = []
        acc: dict[int, dict] = {}
        for chunk in self.client.chat.completions.create(**kwargs, stream=True):
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            text = getattr(delta, "content", None)
            if text:
                parts.append(text)
                yield {"type": "token", "text": text}
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = acc.setdefault(tc.index, {"id": "", "name": "", "arguments": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments
        calls = [acc[i] for i in sorted(acc)]
        for n, c in enumerate(calls):
            c["id"] = c["id"] or f"call_{n}"
        yield {"type": "_final", "content": "".join(parts), "calls": calls, "usage": (0, 0)}

    def _loop(self, question: str, stream: bool = False):
        """Run the tool loop. Yields progress events; the last event is `done`."""
        model_stream = stream and self.stream_tokens
        t0 = time.time()
        sysmsg = SYSTEM.format(refusal=guard.REFUSAL, today=self.today, memory=self.memory.as_prompt())
        messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": question}]
        state = {"retrieved": set(), "ledger_used": False, "sources": [question], "calls": []}
        ptok = ctok = 0
        raw, steps = "", 0
        for steps in range(1, MAX_STEPS + 1):
            yield {"type": "step", "n": steps}
            final = None
            for ev in self._call_model(messages, model_stream):
                if ev["type"] == "_final":
                    final = ev
                else:
                    yield ev
            ptok += final["usage"][0]
            ctok += final["usage"][1]
            if not final["calls"]:
                raw = final["content"]
                break
            messages.append({"role": "assistant", "content": final["content"], "tool_calls": [
                {"id": c["id"], "type": "function",
                 "function": {"name": c["name"], "arguments": c["arguments"]}} for c in final["calls"]]})
            for c in final["calls"]:
                try:
                    args = json.loads(c["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                state["calls"].append({"name": c["name"], "args": args})
                yield {"type": "tool_call", "name": c["name"], "args": args}
                out = self._run_tool(c["name"], args, state)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": out})
                yield {"type": "tool_result", "name": c["name"], "bytes": len(out),
                       "error": json.loads(out).get("error")}
        else:
            raw = ""  # ran out of steps without a final answer

        names = {c["name"] for c in state["calls"]}
        memory_only = names == {"remember_fact"}
        checked = guard.check(raw, state["retrieved"], state["ledger_used"], state["sources"],
                              memory_only=memory_only)

        # A guard failure on citations is usually the model forgetting to cite,
        # not the model not knowing. Give it one turn to fix the citation before
        # we throw a correct answer away. A repair that fails to pass the guard
        # is discarded, so this cannot weaken the check.
        if self.repair and raw and checked["refused"] and checked["reason"] in REPAIRABLE:
            yield {"type": "repair", "reason": checked["reason"]}
            valid = sorted(state["retrieved"]) + (["ledger"] if state["ledger_used"] else [])
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content": REPAIR.format(
                reason=checked["reason"], valid=", ".join(valid) or "(nothing yet)")})
            final = None
            for ev in self._call_model(messages, model_stream):
                if ev["type"] == "_final":
                    final = ev
                else:
                    yield ev
            ptok += final["usage"][0]
            ctok += final["usage"][1]
            steps += 1
            retried_raw = final["content"]
            retried = guard.check(retried_raw, state["retrieved"], state["ledger_used"], state["sources"],
                                  memory_only=memory_only)
            yield {"type": "repair_result", "refused": retried["refused"], "reason": retried["reason"]}
            if not retried["refused"]:
                raw, checked = retried_raw, retried

        result = Result(question=question, answer=checked["answer"], raw_answer=raw, refused=checked["refused"],
                        reason=checked["reason"] if raw else "no_final_answer", citations=checked["citations"],
                        retrieved_ids=sorted(state["retrieved"]), tool_calls=state["calls"],
                        ungrounded_numbers=checked["ungrounded_numbers"], steps=steps,
                        latency_s=round(time.time() - t0, 2), prompt_tokens=ptok, completion_tokens=ctok)
        if model_stream and raw and raw.strip() != result.answer.strip():
            # Tokens already went out. The guard replaced them.
            yield {"type": "replaced", "reason": result.reason, "answer": result.answer}
        yield {"type": "done", "result": result}

    def ask(self, question: str) -> Result:
        for ev in self._loop(question):
            if ev["type"] == "done":
                return ev["result"]
        raise RuntimeError("agent loop produced no result")

    def ask_stream(self, question: str):
        """Same loop, but yields progress and token events as they happen."""
        return self._loop(question, stream=True)
