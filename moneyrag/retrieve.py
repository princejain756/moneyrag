"""Hybrid retrieval: BM25 + dense vectors (fastembed, bge-small-en-v1.5), fused with RRF."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from .bm25 import BM25
from .chunk import Chunk

EMBED_MODEL = os.environ.get("MONEYRAG_EMBED_MODEL", "BAAI/bge-small-en-v1.5")
RRF_K = 60


class Embedder:
    def __init__(self, model: str = EMBED_MODEL):
        from fastembed import TextEmbedding  # local ONNX, no API calls
        self.model_name = model
        self.model = TextEmbedding(model)

    def passages(self, texts: list[str]) -> np.ndarray:
        return _norm(np.array(list(self.model.passage_embed(texts)), dtype=np.float32))

    def query(self, text: str) -> np.ndarray:
        return _norm(np.array(list(self.model.query_embed(text)), dtype=np.float32))[0]


def _norm(m: np.ndarray) -> np.ndarray:
    return m / np.linalg.norm(m, axis=-1, keepdims=True).clip(min=1e-12)


class Retriever:
    def __init__(self, chunks: list[Chunk], embedder: Embedder | None = None, cache_dir: str | Path | None = None):
        self.chunks = chunks
        self.by_id = {c.id: c for c in chunks}
        self.bm25 = BM25([c.text for c in chunks])
        self.embedder = embedder or Embedder()
        self.vectors = self._load_or_embed(cache_dir)

    def _load_or_embed(self, cache_dir):
        texts = [c.text for c in self.chunks]
        key = hashlib.sha256(("\x00".join(texts) + self.embedder.model_name).encode()).hexdigest()[:16]
        path = Path(cache_dir) / f"emb_{key}.npy" if cache_dir else None
        if path and path.exists():
            return np.load(path)
        vecs = self.embedder.passages(texts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, vecs)
        return vecs

    def rank_bm25(self, query: str) -> list[tuple[str, float]]:
        s = self.bm25.scores(query)
        order = sorted(range(len(s)), key=lambda i: -s[i])
        return [(self.chunks[i].id, s[i]) for i in order]

    def rank_vector(self, query: str) -> list[tuple[str, float]]:
        s = self.vectors @ self.embedder.query(query)
        order = np.argsort(-s)
        return [(self.chunks[i].id, float(s[i])) for i in order]

    def rank_hybrid(self, query: str, depth: int = 20) -> list[tuple[str, float]]:
        fused: dict[str, float] = {}
        for ranking in (self.rank_bm25(query)[:depth], self.rank_vector(query)[:depth]):
            for r, (cid, _) in enumerate(ranking, start=1):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + r)
        return sorted(fused.items(), key=lambda kv: -kv[1])

    def search(self, query: str, k: int = 4, mode: str = "hybrid") -> list[dict]:
        ranking = {"bm25": self.rank_bm25, "vector": self.rank_vector, "hybrid": self.rank_hybrid}[mode](query)
        out = []
        for cid, score in ranking[:k]:
            c = self.by_id[cid]
            out.append({"id": cid, "score": round(score, 4), "text": c.text})
        return out
