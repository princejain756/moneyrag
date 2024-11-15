"""Read a bank statement export into raw rows.

Column layout differs per bank, so nothing here guesses at a specific bank:
`infer_mapping` tries to recognise the common header names, and a caller can
always pass an explicit `ColumnMap`. Amounts are returned in integer paise,
because that is the only representation that sums exactly.
"""
from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%d/%m/%y", "%d-%m-%y", "%d %b %Y")
_CURRENCY = re.compile(r"[₹$]|(?<=\d)\s*(?:rs\.?|inr)\b|\brs\.?\s*(?=\d)", re.I)
# 1,20,500.00 or 120500.5 or 120500
_NUMBER = re.compile(r"-?\d+(?:,\d+)*(?:\.\d+)?")
_DR_CR = re.compile(r"\b(dr|cr)\b\.?", re.I)


def parse_amount_paise(text: str | None) -> int | None:
    """'1,20,500.50' -> 12050050. Blank, '-' and 'nil' mean no amount on this leg."""
    if text is None:
        return None
    raw = _CURRENCY.sub("", str(text)).strip()
    if not raw or raw in {"-", "--", "nil", "NA", "N/A"}:
        return None
    negative = raw.startswith("(") and raw.endswith(")")
    if negative:
        raw = raw[1:-1]
    match = _NUMBER.search(raw)
    if not match:
        return None
    value = round(float(match.group(0).replace(",", "")) * 100)
    return -value if negative else value


def sign_hint(text: str | None) -> str | None:
    """Some exports put the direction in the amount ('1,200.00 Dr') rather than a column."""
    if not text:
        return None
    match = _DR_CR.search(str(text))
    if not match:
        return None
    return "debit" if match.group(1).lower() == "dr" else "credit"


def parse_date(text: str | None, formats: tuple[str, ...] = DATE_FORMATS) -> date | None:
    raw = (text or "").strip()
    if not raw:
        return None
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


@dataclass(frozen=True)
class ColumnMap:
    date: str
    narration: str
    debit: str | None = None
    credit: str | None = None
    amount: str | None = None          # when one column carries both legs
    balance: str | None = None
    ref: str | None = None
    date_formats: tuple[str, ...] = DATE_FORMATS


@dataclass
class RawRow:
    source_row: int
    date: date
    narration: str
    debit_paise: int | None
    credit_paise: int | None
    balance_paise: int | None
    ref: str | None = None

    @property
    def amount_paise(self) -> int:
        return self.debit_paise or self.credit_paise or 0

    @property
    def direction(self) -> str:
        return "debit" if self.debit_paise else "credit"


@dataclass
class ParseResult:
    rows: list[RawRow] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)   # (row number, reason)


_ALIASES = {
    "date": ("date", "txn date", "tran date", "transaction date", "value dt", "value date", "post date"),
    "narration": ("narration", "particulars", "description", "transaction remarks", "remarks", "details"),
    "debit": ("withdrawal amt.", "withdrawal amt", "withdrawal", "debit", "debit amount", "withdrawals"),
    "credit": ("deposit amt.", "deposit amt", "deposit", "credit", "credit amount", "deposits"),
    "amount": ("amount", "transaction amount", "amt"),
    "balance": ("closing balance", "balance", "running balance", "available balance"),
    "ref": ("chq/ref no", "chq/ref no.", "cheque no", "ref no", "reference", "chq no."),
}


def _find(header: list[str], names: tuple[str, ...]) -> str | None:
    lowered = {h.strip().lower(): h for h in header}
    for name in names:
        if name in lowered:
            return lowered[name]
    return None


def infer_mapping(header: list[str]) -> ColumnMap | None:
    """Recognise a layout from the header names. Returns None if it cannot."""
    cols = {key: _find(header, names) for key, names in _ALIASES.items()}
    if not cols["date"] or not cols["narration"]:
        return None
    if not (cols["debit"] or cols["credit"] or cols["amount"]):
        return None
    return ColumnMap(date=cols["date"], narration=cols["narration"], debit=cols["debit"],
                     credit=cols["credit"], amount=cols["amount"], balance=cols["balance"],
                     ref=cols["ref"])


def read_rows(text: str, mapping: ColumnMap | None = None) -> ParseResult:
    """Parse CSV text. Blank/summary rows go to `skipped` with a reason, never dropped silently."""
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    delimiter = ";" if first.count(";") > first.count(",") else ","
    reader = csv.DictReader(io.StringIO(text), delimiter=delimiter)
    if not reader.fieldnames:
        return ParseResult(skipped=[(0, "no header row")])
    mapping = mapping or infer_mapping(reader.fieldnames)
    if mapping is None:
        return ParseResult(skipped=[(0, f"unrecognised columns: {reader.fieldnames}")])

    result = ParseResult()
    for offset, row in enumerate(reader, start=2):
        narration = (row.get(mapping.narration) or "").strip()
        when = parse_date(row.get(mapping.date), mapping.date_formats)
        if when is None:
            result.skipped.append((offset, "unparseable date"))
            continue

        debit = parse_amount_paise(row.get(mapping.debit)) if mapping.debit else None
        credit = parse_amount_paise(row.get(mapping.credit)) if mapping.credit else None
        if mapping.amount and not (debit or credit):
            value = parse_amount_paise(row.get(mapping.amount)) or 0
            hint = sign_hint(row.get(mapping.amount))
            if hint == "credit" or value < 0 < -value + 0:
                credit = abs(value)
            elif hint == "debit" or value < 0:
                debit = abs(value)
            elif value:
                debit = value
        debit = abs(debit) if debit else None
        credit = abs(credit) if credit else None
        if not debit and not credit:
            result.skipped.append((offset, "no amount"))
            continue

        result.rows.append(RawRow(
            source_row=offset, date=when, narration=narration,
            debit_paise=debit, credit_paise=credit,
            balance_paise=parse_amount_paise(row.get(mapping.balance)) if mapping.balance else None,
            ref=(row.get(mapping.ref) or "").strip() or None if mapping.ref else None,
        ))
    return result
