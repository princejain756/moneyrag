"""Bank statement ingestion: parse, classify, reconcile, load into the ledger.

    python -m moneyrag.ingest data/statements/example_a.csv --db imported.db
"""
from .classify import Classification, classify
from .parse import ColumnMap, RawRow, infer_mapping, parse_amount_paise, read_rows
from .pipeline import IngestResult, Txn, ingest, to_ledger
from .reconcile import Reconciliation, reconcile

__all__ = [
    "Classification", "ColumnMap", "IngestResult", "RawRow", "Reconciliation", "Txn",
    "classify", "infer_mapping", "ingest", "parse_amount_paise", "read_rows",
    "reconcile", "to_ledger",
]
