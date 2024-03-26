from moneyrag.bm25 import BM25
from moneyrag.chunk import chunk_markdown, load_corpus


def test_chunk_ids_are_stable_and_unique(docs_dir):
    chunks = load_corpus(docs_dir)
    assert len(chunks) == 59
    assert len({c.id for c in chunks}) == 59
    assert "03_fixed_deposits#premature-withdrawal" in {c.id for c in chunks}


def test_chunk_text_carries_title_and_section():
    md = "# Doc Title\n\nintro is skipped\n\n## First Part\nalpha\n\nbeta\n## Second\ngamma\n"
    cs = chunk_markdown("x", md)
    assert [c.id for c in cs] == ["x#first-part", "x#second"]
    assert cs[0].text == "Doc Title. First Part. alpha beta"


def test_bm25_prefers_exact_terms():
    bm = BM25(["rtgs minimum two lakh", "upi daily limit", "neft branch charges"])
    s = bm.scores("rtgs minimum")
    assert s.index(max(s)) == 0
    assert bm.scores("zzz unseen") == [0.0, 0.0, 0.0]


def test_hybrid_returns_k_unique_ids(retriever):
    hits = retriever.search("late fee on credit card bill", k=4, mode="hybrid")
    assert len(hits) == 4 and len({h["id"] for h in hits}) == 4
    assert any(h["id"] == "04_credit_card#late-payment-fee" for h in hits)


def test_rrf_scores_are_sorted(retriever):
    ranked = retriever.rank_hybrid("fixed deposit penalty")
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
