"""python -m moneyrag.ingest statement.csv [--db imported.db] [--owner "PRINCE JAIN"] [--strict]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .pipeline import ingest, to_ledger


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Import a bank statement into the moneyrag ledger.")
    ap.add_argument("statement", help="CSV export from your bank")
    ap.add_argument("--db", help="write a ledger SQLite file the agent can query")
    ap.add_argument("--owner", action="append", default=[],
                    help="your own name as it appears in narrations, for self-transfer detection")
    ap.add_argument("--strict", action="store_true", help="exit non-zero if reconciliation fails")
    args = ap.parse_args(argv)

    result = ingest(Path(args.statement).read_text(), owner_aliases=tuple(args.owner))
    print(json.dumps(result.summary(), indent=2))

    for skip in result.parse.skipped[:10]:
        print(f"skipped row {skip[0]}: {skip[1]}", file=sys.stderr)
    for mismatch in result.reconciliation.mismatches[:10]:
        print(f"reconciliation mismatch: {mismatch}", file=sys.stderr)

    if not result.reconciliation.checked:
        print("warning: statement had no balance column, so the parse could not be verified",
              file=sys.stderr)
    if args.db:
        written = to_ledger(args.db, result.txns)
        print(f"wrote {written} transactions to {args.db}")

    if args.strict and not result.reconciliation.ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
