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
                 model: str | None = None, today: str = "2026-11-19"):
        self.retriever, self.ledger, self.memory, self.today = retriever, ledger, memory, today
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

    def ask(self, question: str) -> Result:
        t0 = time.time()
        sysmsg = SYSTEM.format(refusal=guard.REFUSAL, today=self.today, memory=self.memory.as_prompt())
        messages = [{"role": "system", "content": sysmsg}, {"role": "user", "content": question}]
        state = {"retrieved": set(), "ledger_used": False, "sources": [question], "calls": []}
        ptok = ctok = 0
        raw, steps = "", 0
        for steps in range(1, MAX_STEPS + 1):
            resp = self.client.chat.completions.create(model=self.model, messages=messages, tools=TOOLS, temperature=0)
            if resp.usage:
                ptok += resp.usage.prompt_tokens or 0
                ctok += resp.usage.completion_tokens or 0
            msg = resp.choices[0].message
            if not msg.tool_calls:
                raw = msg.content or ""
                break
            messages.append({"role": "assistant", "content": msg.content or "",
                             "tool_calls": [tc.model_dump() for tc in msg.tool_calls]})
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                state["calls"].append({"name": tc.function.name, "args": args})
                messages.append({"role": "tool", "tool_call_id": tc.id,
                                 "content": self._run_tool(tc.function.name, args, state)})
        else:
            raw = ""  # ran out of steps without a final answer
        names = {c["name"] for c in state["calls"]}
        checked = guard.check(raw, state["retrieved"], state["ledger_used"], state["sources"],
                              memory_only=names == {"remember_fact"})
        return Result(question=question, answer=checked["answer"], raw_answer=raw, refused=checked["refused"],
                      reason=checked["reason"] if raw else "no_final_answer", citations=checked["citations"],
                      retrieved_ids=sorted(state["retrieved"]), tool_calls=state["calls"],
                      ungrounded_numbers=checked["ungrounded_numbers"], steps=steps,
                      latency_s=round(time.time() - t0, 2), prompt_tokens=ptok, completion_tokens=ctok)
