"""Prove the parsed arithmetic against the bank's own running balance.

A parser that drops a row, flips a direction or misreads an Indian grouping still
produces a plausible-looking ledger. The closing balance is the only thing in a
statement that disagrees, so it is the thing to check: if the chain does not
close, the parse is wrong and should fail loudly rather than produce a number
someone might act on.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .parse import RawRow


def signed(row: RawRow) -> int:
    return row.credit_paise if row.credit_paise else -(row.debit_paise or 0)


@dataclass
class Reconciliation:
    checked_pairs: int = 0
    mismatches: list[dict] = field(default_factory=list)
    opening_paise: int | None = None
    closing_paise: int | None = None
    order: str | None = None          # "ascending" | "descending" | None
    total_paise: int = 0

    @property
    def checked(self) -> bool:
        """False when the statement carried no balances, so nothing could be verified."""
        return self.checked_pairs > 0

    @property
    def ok(self) -> bool:
        return self.checked and not self.mismatches


def reconcile(rows: list[RawRow]) -> Reconciliation:
    with_balance = [r for r in rows if r.balance_paise is not None]
    report = Reconciliation(total_paise=sum(signed(r) for r in rows))
    if len(with_balance) < 2:
        return report

    # Some exports run newest-first. Work out which, from the majority of pairs,
    # rather than assuming, then hold every pair to that reading.
    ascending = descending = 0
    for earlier, later in zip(with_balance, with_balance[1:]):
        delta = later.balance_paise - earlier.balance_paise
        ascending += delta == signed(later)
        descending += delta == -signed(earlier)
    report.order = "descending" if descending > ascending else "ascending"

    for earlier, later in zip(with_balance, with_balance[1:]):
        delta = later.balance_paise - earlier.balance_paise
        expected = signed(later) if report.order == "ascending" else -signed(earlier)
        report.checked_pairs += 1
        if delta != expected:
            report.mismatches.append({
                "row": later.source_row if report.order == "ascending" else earlier.source_row,
                "narration": (later if report.order == "ascending" else earlier).narration[:60],
                "expected_delta_paise": expected,
                "actual_delta_paise": delta,
                "off_by_paise": delta - expected,
            })

    first, last = (with_balance[0], with_balance[-1]) if report.order == "ascending" \
        else (with_balance[-1], with_balance[0])
    report.opening_paise = first.balance_paise - signed(first)
    report.closing_paise = last.balance_paise
    derived_close = report.opening_paise + report.total_paise
    if derived_close != report.closing_paise:
        report.mismatches.append({
            "row": None, "narration": "opening + sum(txns) vs closing balance",
            "expected_delta_paise": report.closing_paise - derived_close,
            "actual_delta_paise": 0,
            "off_by_paise": derived_close - report.closing_paise,
        })
    return report
