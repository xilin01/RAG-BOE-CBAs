"""Smoke queries on the real index. Skipped when the index is absent (e.g. in CI).

These queries were written while looking at the articles, so they only check that the
pipeline works end to end. Retrieval quality is measured on the evaluation set (phase 4).
"""

import pytest

from rag_boe.embeddings import INDEX_DIR
from rag_boe.retrieval import CHUNKS

pytestmark = pytest.mark.skipif(
    not (CHUNKS.exists() and (INDEX_DIR / "embeddings.npy").exists()),
    reason="needs data/processed/chunks.jsonl and indexes/ (run ingest + index)",
)

QUERIES = [
    (
        "¿Cuánto dura el periodo de prueba de un auxiliar administrativo?",
        "BOE-A-2025-19507",
        "art-20",
    ),
    (
        "¿Cuántas horas se pueden distribuir de forma irregular al año?",
        "BOE-A-2023-6346",
        "art-22bis",
    ),
    ("¿Cómo se paga el trabajo nocturno?", "BOE-A-2023-13741", "art-50"),
    (
        "¿En qué fondo se integra el plan de pensiones del sector?",
        "BOE-A-2023-19903",
        "libro-3-art-45",
    ),
]


@pytest.fixture(scope="module")
def retriever():
    from rag_boe.search import build_retriever

    return build_retriever()


@pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
@pytest.mark.parametrize(("query", "boe_id", "unit_id"), QUERIES)
def test_known_article_in_top5_with_filter(retriever, mode, query, boe_id, unit_id):
    hits = retriever.search(query, k=5, mode=mode, boe_id=boe_id)
    assert unit_id in [h.chunk["unit_id"] for h in hits]
