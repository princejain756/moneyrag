"""Wiring + CLI: `python -m moneyrag.app "how much did I spend on fuel in August?"`"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

from .agent import Agent
from .chunk import load_corpus
from .ledger import build_db
from .memory import Memory
from .retrieve import Retriever

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
STATE = ROOT / ".state"


def build(seed: int = 42) -> tuple[Retriever, sqlite3.Connection, Memory]:
    STATE.mkdir(exist_ok=True)
    retriever = Retriever(load_corpus(DATA / "docs"), cache_dir=STATE)
    ledger = build_db(STATE / "ledger.db", seed=seed)
    memory = Memory(sqlite3.connect(STATE / "memory.db"))
    return retriever, ledger, memory


def main(argv: list[str]) -> int:
    if not argv:
        print('usage: python -m moneyrag.app "your question"')
        return 2
    r, l, m = build()
    res = Agent(r, l, m).ask(" ".join(argv))
    print(res.answer)
    print(f"\n[steps={res.steps} tools={[c['name'] for c in res.tool_calls]} citations={res.citations} "
          f"guard={res.reason} ungrounded={res.ungrounded_numbers} {res.latency_s}s]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
