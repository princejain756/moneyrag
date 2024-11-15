"""Statement ingestion eval. Offline: no model, no network, no API key.

Two layers are measured separately, because they answer different questions:

* reconciliation (end-to-end, on the fixtures) — did we parse and classify without
  corrupting the arithmetic?
* classification (per narration) — did we read the rail, merchant and self-transfer
  correctly?

Labels are per row in `ingest_golden.jsonl`. `category` is graded end-to-end through
the pipeline, so a linked refund is graded on its inherited category, not on the
placeholder the classifier emits before linking.

Usage: python -m evals.run_ingest [--json]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from moneyrag.ingest import classify, ingest

HERE = Path(__file__).parent
ROOT = HERE.parent
GOLDEN = HERE / "ingest_golden.jsonl"
STATEMENTS = {"a": ROOT / "data" / "statements" / "example_a.csv",
              "b": ROOT / "data" / "statements" / "example_b.csv"}
OWNER = ("PRINCE JAIN",)


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text().splitlines() if line.strip()]


def run() -> dict:
    golden = load_golden()

    # --- reconciliation, end-to-end on the fixtures -------------------------
    results = {key: ingest(path.read_text(), owner_aliases=OWNER) for key, path in STATEMENTS.items()}
    reconciliation = {
        key: {"ok": r.reconciliation.ok, "pairs": r.reconciliation.checked_pairs,
              "mismatches": len(r.reconciliation.mismatches), "order": r.reconciliation.order,
              "transactions": len(r.txns)}
        for key, r in results.items()
    }

    # --- classification, per narration --------------------------------------
    channel_ok = merchant_ok = 0
    self_tp = self_fp = self_fn = 0
    misses = []
    for row in golden:
        got = classify(row["narration"], row["direction"], OWNER)
        if got.channel == row["channel"]:
            channel_ok += 1
        else:
            misses.append(("channel", row["narration"][:44], row["channel"], got.channel))
        if got.merchant == row["merchant"]:
            merchant_ok += 1
        else:
            misses.append(("merchant", row["narration"][:44], row["merchant"], got.merchant))
        if row["self_transfer"]:
            self_tp += got.is_self_transfer
            self_fn += not got.is_self_transfer
        elif got.is_self_transfer:
            self_fp += 1

    # --- category, end-to-end through the pipeline --------------------------
    expected = {row["narration"]: row["category"] for row in golden}
    category_ok = category_total = 0
    unlinked_reversals = 0
    for result in results.values():
        for txn in result.txns:
            want = expected.get(txn.narration)
            if want is None:
                continue
            category_total += 1
            category_ok += txn.category == want
            if not (txn.category == want):
                misses.append(("category", txn.narration[:44], want, txn.category))
            if txn.is_reversal and txn.reversal_of is None:
                unlinked_reversals += 1

    n = len(golden)
    precision = self_tp / (self_tp + self_fp) if (self_tp + self_fp) else 1.0
    recall = self_tp / (self_tp + self_fn) if (self_tp + self_fn) else 1.0

    return {
        "rows_labelled": n,
        "reconciliation": reconciliation,
        "all_reconciled": all(v["ok"] for v in reconciliation.values()),
        "channel_accuracy": round(channel_ok / n, 4),
        "merchant_accuracy": round(merchant_ok / n, 4),
        "self_transfer_precision": round(precision, 3),
        "self_transfer_recall": round(recall, 3),
        "category_accuracy": round(category_ok / category_total, 4) if category_total else None,
        "unlinked_reversals": unlinked_reversals,
        "misses": misses,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    report = run()
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    print(f"labelled rows: {report['rows_labelled']}")
    print("reconciliation:", json.dumps(report["reconciliation"]))
    print(f"channel accuracy:   {report['channel_accuracy']:.3f}")
    print(f"merchant accuracy:  {report['merchant_accuracy']:.3f}")
    print(f"category accuracy:  {report['category_accuracy']:.3f} (end-to-end)")
    print(f"self-transfer p/r:  {report['self_transfer_precision']:.2f} / {report['self_transfer_recall']:.2f}")
    print(f"unlinked reversals: {report['unlinked_reversals']}")
    if report["misses"]:
        print("\nmisses:")
        for kind, narration, want, got in report["misses"]:
            print(f"  {kind:9} {narration:46} want={want} got={got}")
    print(f"\nNote: this is a regression gate on {report['rows_labelled']} labelled rows, not a benchmark.")
    print("The classifier is rules, so a change here is deterministic and reviewable.")


if __name__ == "__main__":
    main()
