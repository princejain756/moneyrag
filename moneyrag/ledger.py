"""Synthetic transaction ledger in SQLite, plus a structured query tool.

The model never writes SQL. It fills a typed argument object; this module
validates every field against allowlists and builds a parameterised query.
Money is stored as integer paise so sums are exact.
"""
from __future__ import annotations

import calendar
import random
import re
import sqlite3
from datetime import date
from pathlib import Path

# Fictional merchants. Seeded, so every build produces the same ledger.
MERCHANTS = {
    "groceries": ["Basket Mart", "Fresh Fields", "Kirana Corner"],
    "food_delivery": ["Zesto", "TiffinGo"],
    "fuel": ["Hindpet Fuels", "Roadstar Pump"],
    "shopping": ["Cartly", "StyleBay", "Gadget Hut"],
    "entertainment": ["ScreenBox", "Tunes Plus", "Cine Square"],
    "travel": ["RailNow", "SkyHop Air", "Cably"],
    "medical": ["CareWell Pharmacy", "City Clinic"],
    "utilities": ["BESCOM Power", "AirFiber Net", "Jalam Water"],
    "rent": ["Landlord R. Iyer"],
    "investments": ["Tunga SIP Desk"],
    "loan_emi": ["Tunga Personal Loan"],
    "salary": ["Northwind Labs Payroll"],
}
# Imported statements carry categories the seeded ledger never generates.
IMPORT_CATEGORIES = ("cash", "fees", "income", "insurance", "other", "person", "transfer")
CATEGORIES = sorted(set(MERCHANTS) | set(IMPORT_CATEGORIES))
OPS = ("sum", "count", "list", "top_merchants", "monthly_totals")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS transactions (
  id INTEGER PRIMARY KEY,
  date TEXT NOT NULL,
  amount_paise INTEGER NOT NULL CHECK (amount_paise > 0),
  direction TEXT NOT NULL CHECK (direction IN ('debit','credit')),
  category TEXT NOT NULL,
  merchant TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_tx_date ON transactions(date);
"""


def _rs(paise: int) -> str:
    return f"Rs {paise // 100:,}.{paise % 100:02d}"


def generate_rows(seed: int = 42, months: tuple[tuple[int, int], ...] = tuple((2026, m) for m in range(5, 11))):
    rng = random.Random(seed)
    rows = []

    def add(d, rupees, direction, cat, merchant=None):
        paise = int(round(rupees * 100))
        rows.append((d.isoformat(), paise, direction, cat, merchant or rng.choice(MERCHANTS[cat])))

    spread = {  # category: (min per month, max per month, min Rs, max Rs)
        "groceries": (6, 10, 180, 3200),
        "food_delivery": (8, 15, 149, 1100),
        "fuel": (3, 5, 500, 2500),
        "shopping": (2, 6, 399, 8999),
        "entertainment": (2, 4, 149, 1200),
        "travel": (0, 3, 250, 9800),
        "medical": (0, 2, 120, 2400),
    }
    for y, m in months:
        last = calendar.monthrange(y, m)[1]
        add(date(y, m, 1), 85000, "credit", "salary")
        add(date(y, m, 3), 24000, "debit", "rent")
        add(date(y, m, 5), 5000, "debit", "investments")
        add(date(y, m, 10), 7850, "debit", "loan_emi")
        for merchant in MERCHANTS["utilities"]:
            add(date(y, m, rng.randint(6, 15)), rng.randint(400, 2600) + rng.choice([0, 0.5]), "debit", "utilities", merchant)
        for cat, (lo, hi, amin, amax) in spread.items():
            for _ in range(rng.randint(lo, hi)):
                add(date(y, m, rng.randint(1, last)), rng.randint(amin * 100, amax * 100) / 100, "debit", cat)
    rows.sort(key=lambda r: r[0])
    return rows


def build_db(path: str | Path, seed: int = 42, check_same_thread: bool = True) -> sqlite3.Connection:
    path = Path(path)
    if path.exists():
        path.unlink()
    conn = sqlite3.connect(path, check_same_thread=check_same_thread)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO transactions(date, amount_paise, direction, category, merchant) VALUES (?,?,?,?,?)",
        generate_rows(seed),
    )
    conn.commit()
    return conn


class ToolError(ValueError):
    pass


def _valid_date(s: str, field: str) -> str:
    if not DATE_RE.match(s):
        raise ToolError(f"{field} must be YYYY-MM-DD, got {s!r}")
    try:
        date.fromisoformat(s)
    except ValueError as e:
        raise ToolError(f"{field} is not a real date: {s!r}") from e
    return s


def query_transactions(conn: sqlite3.Connection, op: str, category: str | None = None,
                       merchant: str | None = None, direction: str = "debit",
                       start_date: str | None = None, end_date: str | None = None,
                       limit: int = 10) -> dict:
    if op not in OPS:
        raise ToolError(f"op must be one of {OPS}")
    if direction not in ("debit", "credit"):
        raise ToolError("direction must be 'debit' or 'credit'")
    where, params = ["direction = ?"], [direction]
    if category is not None:
        if category not in CATEGORIES:
            raise ToolError(f"unknown category {category!r}; valid: {CATEGORIES}")
        where.append("category = ?")
        params.append(category)
    if merchant is not None:
        known = {m.lower(): m for ms in MERCHANTS.values() for m in ms}
        if merchant.lower() not in known:
            raise ToolError(f"unknown merchant {merchant!r}; valid: {sorted(known.values())}")
        where.append("merchant = ?")
        params.append(known[merchant.lower()])
    if start_date:
        where.append("date >= ?")
        params.append(_valid_date(start_date, "start_date"))
    if end_date:
        where.append("date <= ?")
        params.append(_valid_date(end_date, "end_date"))
    limit = max(1, min(int(limit), 50))
    w = " AND ".join(where)
    filters = {"category": category, "merchant": merchant, "direction": direction,
               "start_date": start_date, "end_date": end_date}

    if op == "sum":
        (total, n), = conn.execute(f"SELECT COALESCE(SUM(amount_paise),0), COUNT(*) FROM transactions WHERE {w}", params)
        return {"op": op, "filters": filters, "total": _rs(total), "total_paise": total, "count": n}
    if op == "count":
        (n,), = conn.execute(f"SELECT COUNT(*) FROM transactions WHERE {w}", params)
        return {"op": op, "filters": filters, "count": n}
    if op == "list":
        rows = conn.execute(
            f"SELECT date, merchant, category, amount_paise FROM transactions WHERE {w} ORDER BY date DESC, id DESC LIMIT ?",
            params + [limit]).fetchall()
        return {"op": op, "filters": filters,
                "rows": [{"date": d, "merchant": mer, "category": c, "amount": _rs(a)} for d, mer, c, a in rows]}
    if op == "top_merchants":
        rows = conn.execute(
            f"SELECT merchant, SUM(amount_paise) s, COUNT(*) FROM transactions WHERE {w} GROUP BY merchant ORDER BY s DESC LIMIT ?",
            params + [limit]).fetchall()
        return {"op": op, "filters": filters,
                "merchants": [{"merchant": mer, "total": _rs(s), "count": n} for mer, s, n in rows]}
    rows = conn.execute(
        f"SELECT substr(date,1,7) mo, SUM(amount_paise), COUNT(*) FROM transactions WHERE {w} GROUP BY mo ORDER BY mo", params).fetchall()
    return {"op": op, "filters": filters, "months": [{"month": mo, "total": _rs(s), "count": n} for mo, s, n in rows]}
