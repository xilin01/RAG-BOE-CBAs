"""Retrieval tests on a tiny synthetic corpus; the dense side uses a fake encoder (no torch)."""

import json

import numpy as np
import pytest

from rag_boe import embeddings
from rag_boe.retrieval import BM25Index, DenseIndex, Retriever, rrf, strip_accents, tokenize

DOCS = [
    ("BOE-A-1", "art-1", "El periodo de prueba será de seis meses para el grupo 1."),
    ("BOE-A-1", "art-2", "Las vacaciones anuales serán de 30 días naturales."),
    ("BOE-A-2", "art-1", "El período de prueba no podrá exceder de quince días."),
    ("BOE-A-2", "art-2", "El plus de nocturnidad se abonará con un incremento del 25 %."),
]
CHUNKS = [{"chunk_id": f"{b}:{u}:0", "boe_id": b, "unit_id": u, "text": t} for b, u, t in DOCS]


def test_tokenize_normalises_accents_plurals_and_keeps_short_tokens():
    assert tokenize("período") == tokenize("periodo")
    assert tokenize("vacación") == tokenize("vacaciones")
    assert "3" in tokenize("Grupo 3") and "1" in tokenize("nivel 1")
    assert tokenize("de la y el") == []  # stopwords only
    assert strip_accents("años dieciséis pingüino") == "años dieciseis pinguino"


def test_rrf_rewards_agreement_between_rankings():
    fused = rrf([[1, 2, 3], [3, 1, 4]], k=60)
    assert [idx for idx, _ in fused] == [1, 3, 2, 4]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def fake_encoder(query: str) -> np.ndarray:
    # Unit vectors on two axes: "prueba" vs everything else.
    return np.array([1.0, 0.0] if "prueba" in query else [0.0, 1.0], dtype=np.float32)


def make_retriever() -> Retriever:
    bm25 = BM25Index([c["text"] for c in CHUNKS])
    vectors = np.array([[1, 0], [0, 1], [0.9, 0.1], [0, 1]], dtype=np.float32)
    return Retriever(CHUNKS, bm25, DenseIndex(vectors, fake_encoder))


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_search_finds_both_probation_articles(mode):
    hits = make_retriever().search("¿Cuánto dura el periodo de prueba?", k=2, mode=mode)
    assert {h.chunk["unit_id"] for h in hits} == {"art-1"}
    assert [h.rank for h in hits] == [1, 2]


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
def test_agreement_filter_is_applied_before_top_k(mode):
    hits = make_retriever().search("periodo de prueba", k=2, mode=mode, boe_id="BOE-A-2")
    assert hits and all(h.chunk["boe_id"] == "BOE-A-2" for h in hits)
    assert hits[0].chunk["unit_id"] == "art-1"


def test_bm25_returns_nothing_for_unknown_words():
    assert make_retriever().search("xyzzy", mode="bm25") == []


def test_dense_mode_requires_dense_index():
    retriever = Retriever(CHUNKS, BM25Index([c["text"] for c in CHUNKS]))
    with pytest.raises(ValueError):
        retriever.search("prueba", mode="hybrid")


def test_stale_embeddings_are_rejected(tmp_path, monkeypatch):
    chunks_file = tmp_path / "chunks.jsonl"
    chunks_file.write_text("\n".join(json.dumps(c) for c in CHUNKS), encoding="utf-8")
    monkeypatch.setattr(embeddings, "CHUNKS", chunks_file)
    ids = [c["chunk_id"] for c in CHUNKS]
    embeddings.save(np.zeros((4, 2), dtype=np.float32), ids, "fake", tmp_path)

    vectors, model = embeddings.load(ids, tmp_path, chunks_file)
    assert vectors.shape == (4, 2) and model == "fake"
    with pytest.raises(RuntimeError):
        embeddings.load(ids[::-1], tmp_path, chunks_file)  # different chunk order
    chunks_file.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError):
        embeddings.load(ids, tmp_path, chunks_file)  # chunks.jsonl rebuilt
