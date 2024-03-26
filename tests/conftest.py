import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from moneyrag.bm25 import tokenize  # noqa: E402


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder so tests never download a model."""
    model_name = "test-hash"

    def _vec(self, text):
        v = np.zeros(256, dtype=np.float32)
        for t in tokenize(text):
            v[hash(t) % 256] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def passages(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def query(self, text):
        return self._vec(text)


@pytest.fixture
def docs_dir():
    return ROOT / "data" / "docs"


@pytest.fixture
def retriever(docs_dir):
    from moneyrag.chunk import load_corpus
    from moneyrag.retrieve import Retriever
    return Retriever(load_corpus(docs_dir), embedder=HashEmbedder())
