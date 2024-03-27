"""Agent loop with a scripted fake client: checks tool dispatch and the guard, no network."""
import json
import sqlite3
from types import SimpleNamespace as NS

from moneyrag.agent import Agent
from moneyrag.ledger import build_db
from moneyrag.memory import Memory


class FakeClient:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.chat = NS(completions=NS(create=self._create))

    def _create(self, **kw):
        self.seen.append(kw["messages"])
        step = self.script.pop(0)
        if isinstance(step, str):
            msg = NS(content=step, tool_calls=None)
        else:
            calls = [NS(id=f"c{i}", function=NS(name=n, arguments=json.dumps(a)),
                        model_dump=lambda n=n, a=a, i=i: {"id": f"c{i}", "type": "function",
                                                          "function": {"name": n, "arguments": json.dumps(a)}})
                     for i, (n, a) in enumerate(step)]
            msg = NS(content="", tool_calls=calls)
        return NS(choices=[NS(message=msg)], usage=NS(prompt_tokens=10, completion_tokens=5))


def make(retriever, tmp_path, script):
    return Agent(retriever, build_db(tmp_path / "l.db"), Memory(sqlite3.connect(":memory:")), client=FakeClient(script))


def test_doc_answer_with_real_citation(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("search_docs", {"query": "late fee credit card"})],
                                   "Rs 500 for a 12,000 bill [04_credit_card#late-payment-fee]."])
    r = a.ask("Late fee on a 12,000 bill?")
    assert not r.refused and r.citations == ["04_credit_card#late-payment-fee"]
    assert r.ungrounded_numbers == []


def test_hallucinated_citation_is_refused(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("search_docs", {"query": "late fee credit card"})],
                                   "Gold loans are 9% [11_gold_loan#rates]."])
    r = a.ask("gold loan rate?")
    assert r.refused and r.reason == "invalid_citation"


def test_ledger_tool_result_flows_back(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("query_transactions", {"op": "count", "category": "salary", "direction": "credit"})],
                                   "You had 6 salary credits [ledger]."])
    r = a.ask("how many salary credits?")
    assert not r.refused
    tool_msg = a.client.seen[1][-1]
    assert tool_msg["role"] == "tool" and json.loads(tool_msg["content"])["count"] == 6


def test_tool_error_is_returned_to_model_not_raised(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("query_transactions", {"op": "sum", "category": "crypto"})], "I can't answer that from your bank documents or transactions."])
    r = a.ask("crypto spend?")
    assert "unknown category" in a.client.seen[1][-1]["content"]
    assert r.refused


def test_step_limit(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("search_docs", {"query": "x"})]] * 6)
    r = a.ask("loop forever")
    assert r.refused and r.reason == "no_final_answer" and r.steps == 6


def test_memory_tool_and_prompt_injection(retriever, tmp_path):
    a = make(retriever, tmp_path, [[("remember_fact", {"key": "salary_credit_day", "value": "1st"})],
                                   "Noted, your salary comes on the 1st."])
    r = a.ask("My salary comes on the 1st.")
    assert a.memory.all() == {"salary_credit_day": "1st"}
    assert not r.refused
