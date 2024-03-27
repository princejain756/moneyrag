import sqlite3

import pytest

from moneyrag import guard
from moneyrag.memory import Memory

DOC = '{"results":[{"id":"03_fixed_deposits#premature-withdrawal","text":"minus a 1% penalty ... within 7 days"}]}'


def test_valid_citation_passes():
    r = guard.check("Breaking early costs 1% [03_fixed_deposits#premature-withdrawal].",
                    {"03_fixed_deposits#premature-withdrawal"}, False, [DOC])
    assert not r["refused"] and r["ungrounded_numbers"] == []


def test_citation_to_unretrieved_chunk_is_blocked():
    r = guard.check("Penalty is 1% [03_fixed_deposits#auto-renewal].", {"03_fixed_deposits#premature-withdrawal"}, False, [DOC])
    assert r["refused"] and r["reason"] == "invalid_citation"


def test_uncited_answer_is_blocked():
    r = guard.check("The penalty is 1%.", {"03_fixed_deposits#premature-withdrawal"}, False, [DOC])
    assert r["refused"] and r["reason"] == "uncited"


def test_ledger_citation_needs_ledger_call():
    assert guard.check("You spent Rs 10 [ledger].", set(), False, ["Rs 10"])["reason"] == "invalid_citation"
    assert not guard.check("You spent Rs 10 [ledger].", set(), True, ["Rs 10"])["refused"]


def test_multiple_ids_in_one_bracket():
    assert guard.parse_citations("x [a_b#c-d, ledger] y [not a cite]") == ["a_b#c-d", "ledger"]


def test_made_up_number_is_flagged():
    r = guard.check("Penalty is 2% [03_fixed_deposits#premature-withdrawal].",
                    {"03_fixed_deposits#premature-withdrawal"}, False, [DOC])
    assert r["ungrounded_numbers"] == [2.0]


def test_refusal_detected():
    assert guard.check(guard.REFUSAL, set(), False, [])["reason"] == "model_refused"


def test_memory_upserts_and_validates():
    m = Memory(sqlite3.connect(":memory:"))
    m.remember("salary_credit_day", "1st of the month")
    m.remember("salary_credit_day", "2nd of the month")
    assert m.all() == {"salary_credit_day": "2nd of the month"}
    assert "salary_credit_day" in m.as_prompt()
    with pytest.raises(ValueError):
        m.remember("Bad Key!", "x")
    with pytest.raises(ValueError):
        m.remember("k_ok", "x" * 500)
