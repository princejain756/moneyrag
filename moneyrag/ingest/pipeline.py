"""Statement text in, typed and reconciled transactions out."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from .classify import Classification, classify
from .parse import ColumnMap, ParseResult, RawRow, read_rows
from .reconcile import Reconciliation, reconcile, signed

REVERSAL_WINDOW = timedelta(days=45)
SELF_TRANSFER_WINDOW = timedelta(days=3)

# Categories where a merchant is expected, so a missing one is a classifier miss
# rather than a fact about the transaction.
MERCHANT_CATEGORIES = {"groceries", "food_delivery", "fuel", "shopping", "entertainment",
                       "travel", "medical", "utilities", "investments", "insurance"}


@dataclass
class Txn:
    date: str
    amount_paise: int
    direction: str
    channel: str
    category: str
    merchant: str | None
    counterparty: str | None
    vpa: str | None
    is_self_transfer: bool
    reversal_of: int | None
    balance_paise: int | None
    narration: str
    source_row: int
    is_reversal: bool = False

    def as_row(self) -> tuple:
        merchant = self.merchant or self.counterparty or self.channel
        return (self.date, self.amount_paise, self.direction, self.category, merchant)


@dataclass
class IngestResult:
    txns: list[Txn] = field(default_factory=list)
    parse: ParseResult = field(default_factory=ParseResult)
    reconciliation: Reconciliation = field(default_factory=Reconciliation)

    @property
    def unknown_channel_rate(self) -> float:
        return self._rate(lambda t: t.channel == "other")

    @property
    def unknown_merchant_rate(self) -> float:
        """Only counts rows where a merchant was expected. Cash and fees have none."""
        return self._rate(lambda t: t.category in MERCHANT_CATEGORIES and t.merchant is None)

    def _rate(self, predicate) -> float:
        return round(sum(1 for t in self.txns if predicate(t)) / len(self.txns), 4) if self.txns else 0.0

    def summary(self) -> dict:
        return {
            "transactions": len(self.txns),
            "skipped_rows": len(self.parse.skipped),
            "reconciled": self.reconciliation.ok,
            "pairs_checked": self.reconciliation.checked_pairs,
            "mismatches": len(self.reconciliation.mismatches),
            "order": self.reconciliation.order,
            "opening_rs": _rs(self.reconciliation.opening_paise),
            "closing_rs": _rs(self.reconciliation.closing_paise),
            "total_rs": _rs(self.reconciliation.total_paise),
            "unknown_channel_rate": self.unknown_channel_rate,
            "unknown_merchant_rate": self.unknown_merchant_rate,
            "self_transfers": sum(1 for t in self.txns if t.is_self_transfer),
            "reversals_linked": sum(1 for t in self.txns if t.reversal_of is not None),
        }


def _rs(paise: int | None) -> str | None:
    return None if paise is None else f"{paise // 100:,}.{abs(paise) % 100:02d}"


def ingest(text: str, mapping: ColumnMap | None = None,
           owner_aliases: tuple[str, ...] = ()) -> IngestResult:
    parsed = read_rows(text, mapping)
    txns: list[Txn] = []
    for raw in parsed.rows:
        c: Classification = classify(raw.narration, raw.direction, owner_aliases)
        txns.append(Txn(date=raw.date.isoformat(), amount_paise=raw.amount_paise,
                        direction=raw.direction, channel=c.channel, category=c.category,
                        merchant=c.merchant, counterparty=c.counterparty, vpa=c.vpa,
                        is_self_transfer=c.is_self_transfer, reversal_of=None,
                        balance_paise=raw.balance_paise, narration=raw.narration,
                        source_row=raw.source_row, is_reversal=c.is_reversal))

    _pair_self_transfers(txns)
    _link_reversals(txns)
    return IngestResult(txns=txns, parse=parsed, reconciliation=reconcile(parsed.rows))


def _pair_self_transfers(txns: list[Txn]) -> None:
    """A debit and credit of equal size within a few days, one side owned, is a move between accounts."""
    for a in txns:
        if a.is_self_transfer:
            for b in txns:
                if b is a or b.is_self_transfer or b.amount_paise != a.amount_paise:
                    continue
                if b.direction == a.direction or abs(_days(a.date) - _days(b.date)) > SELF_TRANSFER_WINDOW.days:
                    continue
                b.is_self_transfer = True
                b.category, b.merchant = "transfer", b.merchant


def _link_reversals(txns: list[Txn]) -> None:
    """Point a refund/reversal at the transaction it undoes, so spend is not double counted."""
    for i, rev in enumerate(txns):
        if not rev.is_reversal or rev.reversal_of is not None:
            continue
        best = None
        for j, orig in enumerate(txns):
            if j == i or orig.direction == rev.direction or orig.amount_paise != rev.amount_paise:
                continue
            if orig.is_reversal or orig.reversal_of is not None:
                continue
            if abs(_days(orig.date) - _days(rev.date)) > REVERSAL_WINDOW.days:
                continue
            if best is None or _days(orig.date) < _days(txns[best].date):
                best = j
        if best is not None:
            # A refund belongs to whatever it refunded, not to a generic bucket.
            rev.reversal_of = best
            rev.category = txns[best].category
            rev.merchant = txns[best].merchant


def _days(iso: str) -> int:
    return datetime.fromisoformat(iso).toordinal()


def to_ledger(path: str | Path, txns: list[Txn]) -> int:
    """Write imported rows into the same schema the query tool already reads."""
    path = Path(path)
    if path.exists():
        path.unlink()
    from ..ledger import SCHEMA
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO transactions(date, amount_paise, direction, category, merchant) VALUES (?,?,?,?,?)",
        [t.as_row() for t in txns],
    )
    conn.commit()
    conn.close()
    return len(txns)
