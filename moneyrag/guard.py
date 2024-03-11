"""Post-answer checks. These run on every answer before it reaches the user.

1. Citation check: every [chunk-id] the model cites must be a chunk that a
   search_docs call actually returned in this turn. [ledger] is valid only if a
   query_transactions call succeeded.
2. Uncited answers are replaced with the refusal line. A money answer with no
   source is worse than no answer.
3. Number grounding: every number in the answer must appear in the question or
   in some tool output. Misses are flagged (not blocked) and counted in evals.
"""
from __future__ import annotations

import re

REFUSAL = "I can't answer that from your bank documents or transactions."

CITE_GROUP = re.compile(r"\[([^\[\]]+)\]")
CITE_ID = re.compile(r"^(?:[0-9a-z_]+#[a-z0-9-]+|ledger)$")
NUM = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")


def is_refusal(text: str) -> bool:
    return REFUSAL.lower().rstrip(".") in " ".join(text.lower().split())


def parse_citations(text: str) -> list[str]:
    out = []
    for group in CITE_GROUP.findall(text):
        for part in re.split(r"[,;]\s*", group):
            part = part.strip()
            if CITE_ID.match(part):
                out.append(part)
    return out


def _nums(text: str) -> set[float]:
    vals = set()
    for m in NUM.findall(text):
        try:
            vals.add(round(float(m.replace(",", "")), 2))
        except ValueError:
            pass
    return vals


def ungrounded_numbers(answer: str, sources: list[str]) -> list[float]:
    body = CITE_GROUP.sub(" ", answer)  # chunk ids like 03_fixed... carry digits
    allowed = set().union(*(_nums(s) for s in sources)) if sources else set()
    return sorted(n for n in _nums(body) if n not in allowed)


def check(answer: str, retrieved_ids: set[str], ledger_used: bool, sources: list[str],
          memory_only: bool = False) -> dict:
    if is_refusal(answer):
        return {"answer": REFUSAL, "refused": True, "reason": "model_refused", "citations": [], "ungrounded_numbers": []}
    cites = parse_citations(answer)
    valid = retrieved_ids | ({"ledger"} if ledger_used else set())
    bad = [c for c in cites if c not in valid]
    if bad:
        return {"answer": REFUSAL, "refused": True, "reason": "invalid_citation", "citations": cites,
                "invalid": bad, "ungrounded_numbers": []}
    if not cites and memory_only and not ungrounded_numbers(answer, sources):
        # Acknowledging a saved fact ("Noted, salary on the 1st") needs no source.
        return {"answer": answer, "refused": False, "reason": None, "citations": [], "ungrounded_numbers": []}
    if not cites:
        return {"answer": REFUSAL, "refused": True, "reason": "uncited", "citations": [], "ungrounded_numbers": []}
    return {"answer": answer, "refused": False, "reason": None, "citations": cites,
            "ungrounded_numbers": ungrounded_numbers(answer, sources)}
