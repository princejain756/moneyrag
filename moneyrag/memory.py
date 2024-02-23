"""User-fact memory: small key/value facts the user tells the agent, kept in SQLite.

Facts are injected into the system prompt on every turn. Writing a fact with an
existing key overwrites it, so stale facts get replaced rather than piling up.
"""
from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timezone

KEY_RE = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
MAX_VALUE = 200
MAX_FACTS = 50


class Memory:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS facts (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL)")
        conn.commit()

    def remember(self, key: str, value: str) -> dict:
        key = key.strip().lower()
        if not KEY_RE.match(key):
            raise ValueError("key must be snake_case, 2 to 40 chars")
        value = " ".join(str(value).split())
        if not value or len(value) > MAX_VALUE:
            raise ValueError(f"value must be 1 to {MAX_VALUE} chars")
        (n,), = self.conn.execute("SELECT COUNT(*) FROM facts WHERE key != ?", (key,))
        if n >= MAX_FACTS:
            raise ValueError("memory full")
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.conn.execute("INSERT INTO facts(key, value, updated_at) VALUES (?,?,?) "
                          "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                          (key, value, now))
        self.conn.commit()
        return {"stored": key}

    def forget(self, key: str) -> dict:
        cur = self.conn.execute("DELETE FROM facts WHERE key = ?", (key.strip().lower(),))
        self.conn.commit()
        return {"deleted": cur.rowcount}

    def all(self) -> dict[str, str]:
        return dict(self.conn.execute("SELECT key, value FROM facts ORDER BY key"))

    def as_prompt(self) -> str:
        facts = self.all()
        if not facts:
            return "No saved facts about the user yet."
        return "Saved facts about the user (user-provided, may be outdated):\n" + "\n".join(f"- {k}: {v}" for k, v in facts.items())
