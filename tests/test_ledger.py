import pytest

from moneyrag.ledger import ToolError, build_db, generate_rows, query_transactions


@pytest.fixture
def conn(tmp_path):
    return build_db(tmp_path / "l.db", seed=7)


def test_generation_is_deterministic():
    assert generate_rows(42) == generate_rows(42)
    assert generate_rows(42) != generate_rows(43)


def test_sum_matches_python_loop_to_the_paisa(conn):
    rows = generate_rows(7)
    want = sum(r[1] for r in rows if r[2] == "debit" and r[3] == "groceries" and "2026-08-01" <= r[0] <= "2026-08-31")
    got = query_transactions(conn, "sum", category="groceries", start_date="2026-08-01", end_date="2026-08-31")
    assert got["total_paise"] == want
    assert got["total"] == f"Rs {want // 100:,}.{want % 100:02d}"


def test_salary_is_credit_not_debit(conn):
    assert query_transactions(conn, "count", category="salary")["count"] == 0
    assert query_transactions(conn, "count", category="salary", direction="credit")["count"] == 6


@pytest.mark.parametrize("kwargs", [
    {"op": "drop"},
    {"op": "sum", "category": "crypto"},
    {"op": "sum", "merchant": "x'; DROP TABLE transactions;--"},
    {"op": "sum", "start_date": "2026-13-01"},
    {"op": "sum", "start_date": "Aug 2026"},
    {"op": "sum", "direction": "both"},
])
def test_rejects_bad_arguments(conn, kwargs):
    with pytest.raises(ToolError):
        query_transactions(conn, **kwargs)
    assert query_transactions(conn, "count")["count"] > 0  # table still there


def test_merchant_match_is_case_insensitive(conn):
    a = query_transactions(conn, "sum", merchant="zesto")
    b = query_transactions(conn, "sum", merchant="Zesto")
    assert a["total_paise"] == b["total_paise"] > 0


def test_limit_is_clamped(conn):
    assert len(query_transactions(conn, "list", limit=999)["rows"]) == 50
