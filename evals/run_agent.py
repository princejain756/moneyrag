"""End-to-end agent eval against a live model.

Usage: OPENROUTER_API_KEY=... python -m evals.run_agent [--model deepseek/deepseek-v4.1-flash]

Ground truth for ledger questions is computed here with plain Python loops over
the generated rows, not through the SQL tool, so a bug in the tool cannot
grade itself as correct.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from moneyrag.agent import Agent
from moneyrag.app import build
from moneyrag.ledger import generate_rows

HERE = Path(__file__).parent
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def ledger_truth(t: dict, rows) -> object:
    sel = [r for r in rows
           if r[2] == t.get("direction", "debit")
           and t["start"] <= r[0] <= t["end"]
           and ("category" not in t or r[3] == t["category"])
           and ("merchant" not in t or r[4] == t["merchant"])]
    if t["op"] == "sum":
        return sum(r[1] for r in sel) / 100
    if t["op"] == "count":
        return len(sel)
    if t["op"] == "avg_month":
        return round(sum(r[1] for r in sel) / 100 / t["months"], 2)
    if t["op"] == "top_merchant":
        c = defaultdict(int)
        for r in sel:
            c[r[4]] += r[1]
        return max(c, key=c.get)
    raise ValueError(t["op"])


def numbers(text: str) -> list[float]:
    return [float(m.replace(",", "")) for m in NUM.findall(re.sub(r"\[[^\]]*\]", " ", text))]


def grade(item: dict, res, rows) -> dict:
    out = {"type": item["type"], "q": item["q"], "answer": res.answer, "reason": res.reason,
           "citations": res.citations, "ungrounded_numbers": res.ungrounded_numbers,
           "tools": [c["name"] for c in res.tool_calls], "latency_s": res.latency_s,
           "tokens": res.prompt_tokens + res.completion_tokens}
    if item["type"] == "hard":
        c, text = item["check"], res.answer.lower()
        ok = True
        if c.get("refuse"):
            ok = res.refused
        if c.get("answered") or "must" in c or "any" in c or "truth" in c or "truth_all" in c:
            ok = ok and not res.refused
        ok = ok and all(m.lower() in text for m in c.get("must", []))
        if "any" in c:
            ok = ok and any(m.lower() in text for m in c["any"])
        ok = ok and all(x in res.citations for x in c.get("need_cites", []))
        nums = numbers(res.answer)
        if "truth" in c:
            t = ledger_truth(c["truth"], rows)
            out["truth"] = t
            ok = ok and any(abs(n - t) <= c["tol"] for n in nums)
        if "truth_all" in c:
            ts = [ledger_truth(t, rows) for t in c["truth_all"]]
            out["truth"] = ts
            ok = ok and all(any(abs(n - t) < 0.005 for n in nums) for t in ts)
        if c.get("no_nonzero_money"):
            ok = not re.search(r"(rs\.?|₹|inr)\s*[1-9]", text) or bool(re.search(r"(rs\.?|₹|inr)\s*0(\.0+)?\b", text))
        out["tag"] = item["tag"]
        out["correct"] = bool(ok)
        return out
    if item["type"] == "refuse":
        out["correct"] = res.refused
    elif item["type"] == "doc":
        text = res.answer.lower()
        out["correct"] = (not res.refused) and all(m.lower() in text for m in item["must"])
        out["cited_gold"] = item["gold"] in res.citations
    else:
        truth = ledger_truth(item["truth"], rows)
        out["truth"] = truth
        if res.refused:
            out["correct"] = False
        elif isinstance(truth, str):
            out["correct"] = truth.lower() in res.answer.lower()
        else:
            out["correct"] = any(abs(n - truth) < 0.005 for n in numbers(res.answer))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default="agent.json")
    ap.add_argument("--set", default="agent_golden.jsonl")
    args = ap.parse_args()
    items = [json.loads(l) for l in (HERE / args.set).read_text().splitlines() if l.strip()]
    retriever, ledger, memory = build()
    memory.conn.execute("DELETE FROM facts")
    memory.conn.commit()
    agent = Agent(retriever, ledger, memory, model=args.model)
    rows = generate_rows(42)
    graded = []
    for it in items:
        try:
            res = agent.ask(it["q"])
            g = grade(it, res, rows)
        except Exception as e:  # count API errors as failures, do not drop them
            g = {"type": it["type"], "q": it["q"], "correct": False, "error": f"{type(e).__name__}: {e}"}
        graded.append(g)
        print(f"[{'ok ' if g['correct'] else 'BAD'}] {it['type']:6} {it['q'][:70]}")
        time.sleep(0.3)

    by = defaultdict(list)
    for g in graded:
        by[g["type"]].append(g)
    answered = [g for g in graded if "error" not in g and g["reason"] is None]
    summary = {
        "model": agent.model,
        "n": len(graded),
        "accuracy": {t: f"{sum(g['correct'] for g in gs)}/{len(gs)}" for t, gs in by.items()},
        "overall": f"{sum(g['correct'] for g in graded)}/{len(graded)}",
        "doc_cited_gold_chunk": f"{sum(g.get('cited_gold', False) for g in by['doc'])}/{len(by['doc'])}",
        "false_refusals": sum(1 for g in graded if g["type"] in ("doc", "ledger") and g.get("reason") is not None),
        "guard_reasons": dict(Counter(g.get("reason") for g in graded)),
        "answers_with_ungrounded_numbers": f"{sum(bool(g.get('ungrounded_numbers')) for g in answered)}/{len(answered)}",
        "by_tag": {t: f"{sum(g['correct'] for g in graded if g.get('tag') == t)}/{sum(1 for g in graded if g.get('tag') == t)}"
                   for t in sorted({g.get("tag") for g in graded if g.get("tag")})},
        "errors": sum("error" in g for g in graded),
        "median_latency_s": sorted(g.get("latency_s", 0) for g in graded)[len(graded) // 2],
        "total_tokens": sum(g.get("tokens", 0) for g in graded),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / args.out).write_text(json.dumps({"summary": summary, "items": graded}, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
