"""Split markdown policy docs into one chunk per `##` section.

Chunk ids look like `03_fixed_deposits#premature-withdrawal`. The agent cites
these ids, and the citation validator checks them against what was retrieved.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Chunk:
    id: str
    doc: str
    title: str
    section: str
    text: str


def slugify(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def chunk_markdown(doc_id: str, md: str) -> list[Chunk]:
    title = ""
    chunks: list[Chunk] = []
    section, buf = None, []

    def flush():
        if section and "".join(buf).strip():
            body = " ".join(line.strip() for line in buf if line.strip())
            chunks.append(Chunk(
                id=f"{doc_id}#{slugify(section)}", doc=doc_id, title=title,
                section=section, text=f"{title}. {section}. {body}",
            ))

    for line in md.splitlines():
        if line.startswith("# "):
            title = line[2:].strip()
        elif line.startswith("## "):
            flush()
            section, buf = line[3:].strip(), []
        elif section is not None:
            buf.append(line)
    flush()
    return chunks


def load_corpus(docs_dir: str | Path) -> list[Chunk]:
    out: list[Chunk] = []
    for p in sorted(Path(docs_dir).glob("*.md")):
        out.extend(chunk_markdown(p.stem, p.read_text(encoding="utf-8")))
    ids = [c.id for c in out]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"duplicate chunk ids: {sorted(dupes)}")
    return out
