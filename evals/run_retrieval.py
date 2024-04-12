"""Retrieval eval: recall@k and MRR for BM25, vector and hybrid (RRF) on the golden set.

Usage: python -m evals.run_retrieval
No API calls. Embeddings run locally through fastembed.
"""
from __future__ import annotations

import json
from pathlib import Path

from moneyrag.app import DATA, STATE
from moneyrag.chunk import load_corpus
from moneyrag.retrieve import Retriever

HERE = Path(__file__).parent
KS = (1, 3, 4, 5)


def evaluate(retriever: Retriever, golden: list[dict]) -> dict:
    report = {}
    for mode in ("bm25", "vector", "hybrid"):
        rank_fn = {"bm25": retriever.rank_bm25, "vector": retriever.rank_vector, "hybrid": retriever.rank_hybrid}[mode]
        hits = {k: 0 for k in KS}
        rr = 0.0
        misses = []
        for g in golden:
            ids = [cid for cid, _ in rank_fn(g["q"])]
            rank = ids.index(g["gold"]) + 1 if g["gold"] in ids else None
            for k in KS:
                hits[k] += int(rank is not None and rank <= k)
            rr += 1.0 / rank if rank else 0.0
            if rank != 1:
                misses.append({"q": g["q"], "gold": g["gold"], "rank": rank, "top1": ids[0]})
        n = len(golden)
        report[mode] = {**{f"recall@{k}": round(hits[k] / n, 3) for k in KS}, "mrr": round(rr / n, 3),
                        "n": n, "misses_at_1": misses}
    return report


def main():
    golden = [json.loads(l) for l in (HERE / "retrieval_golden.jsonl").read_text().splitlines() if l.strip()]
    retriever = Retriever(load_corpus(DATA / "docs"), cache_dir=STATE)
    rep = evaluate(retriever, golden)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "retrieval.json").write_text(json.dumps(rep, indent=2))
    print(f"{'mode':8} {'R@1':>6} {'R@3':>6} {'R@4':>6} {'R@5':>6} {'MRR':>6}   (n={len(golden)}, corpus={len(retriever.chunks)} chunks)")
    for mode, r in rep.items():
        print(f"{mode:8} {r['recall@1']:6.3f} {r['recall@3']:6.3f} {r['recall@4']:6.3f} {r['recall@5']:6.3f} {r['mrr']:6.3f}")


if __name__ == "__main__":
    main()
