"""Tests for the eval grader itself.

A grader that is wrong silently marks correct answers as failures, which is how
the near_miss_refusal case went wrong: it demanded a refusal for a question the
corpus can answer with a correction. These tests pin the corrected behaviour.
"""
import json
from pathlib import Path
from types import SimpleNamespace as NS

from evals.run_agent import grade, subcheck

HARD = Path(__file__).resolve().parent.parent / "evals" / "agent_hard.jsonl"


def result(answer, refused=False, citations=(), reason=None):
    return NS(answer=answer, refused=refused, reason=reason, citations=list(citations),
              ungrounded_numbers=[], tool_calls=[], latency_s=0.0, prompt_tokens=0, completion_tokens=0)


def hard_item(tag):
    for line in HARD.read_text().splitlines():
        item = json.loads(line)
        if item.get("tag") == tag:
            return item
    raise AssertionError(f"no hard item tagged {tag}")


def test_subcheck_refuse_only():
    assert subcheck({"refuse": True}, True, "anything", [])
    assert not subcheck({"refuse": True}, False, "anything", [])


def test_subcheck_must_and_must_not():
    assert subcheck({"must": ["fixed deposit"]}, False, "that is a fixed deposit rate", [])
    assert not subcheck({"must": ["fixed deposit"]}, False, "that is a savings rate", [])
    assert not subcheck({"must": ["fixed deposit"], "must_not": ["0.5% on savings"]},
                        False, "fixed deposit and 0.5% on savings", [])
    assert not subcheck({"must": ["x"]}, True, "x", []), "a refusal cannot satisfy an answer check"


def test_subcheck_need_cites():
    assert subcheck({"must": ["a"], "need_cites": ["03_fixed_deposits#interest-rates"]},
                    False, "a", ["03_fixed_deposits#interest-rates"])
    assert not subcheck({"must": ["a"], "need_cites": ["03_fixed_deposits#interest-rates"]},
                        False, "a", ["04_credit_card#late-payment-fee"])


def test_near_miss_refusal_accepts_both_correct_behaviours():
    item = hard_item("near_miss_refusal")
    refusal = result("I can't answer that from your bank documents or transactions.",
                     refused=True, reason="model_refused")
    correction = result("Savings accounts do not get an extra senior-citizen rate; the extra "
                        "0.5% applies to fixed deposits [03_fixed_deposits#interest-rates].",
                        citations=["03_fixed_deposits#interest-rates"])
    assert grade(item, refusal, [])["correct"] is True
    assert grade(item, correction, [])["correct"] is True


def test_near_miss_refusal_still_rejects_an_invented_savings_rate():
    """The point of the case: a fabricated senior savings rate must still fail."""
    item = hard_item("near_miss_refusal")
    invented = result("Senior citizens get an extra 0.5% on savings accounts, same as fixed deposits.",
                      citations=["03_fixed_deposits#interest-rates"])
    assert grade(item, invented, [])["correct"] is False
