"""Sparse (BM25), dense and hybrid (RRF) retrieval over the chunk store.

All three modes score every chunk first and apply the optional agreement filter before
taking the top results, so a filtered search never comes back short.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import bm25s
import numpy as np
import Stemmer
from bm25s.stopwords import STOPWORDS_SPANISH

from rag_boe.corpus import ROOT, load_manifest

CHUNKS = ROOT / "data" / "processed" / "chunks.jsonl"
MODES = ("bm25", "dense", "hybrid")

_STEMMER = Stemmer.Stemmer("spanish")
_TOKEN = re.compile(r"\w+")  # keeps 1-char tokens: "grupo 3", "nivel 1", "art. 4"


def strip_accents(text: str) -> str:
    """Remove acute accents and diaeresis but keep "ñ" ("años" must not become "anos").

    In NFD, "ñ" is n + U+0303 (tilde), which is not removed.
    """
    decomposed = unicodedata.normalize("NFD", text)
    return unicodedata.normalize("NFC", re.sub("[́̈]", "", decomposed))


_STOPWORDS = {strip_accents(w) for w in STOPWORDS_SPANISH}


def tokenize(text: str) -> list[str]:
    """Lowercase, drop stopwords, stem (Snowball Spanish), then strip accents.

    The same function runs on chunks and queries, so "período"/"periodo" and
    "vacación"/"vacaciones" end up as the same token.
    """
    words = [w for w in _TOKEN.findall(text.lower()) if strip_accents(w) not in _STOPWORDS]
    return [strip_accents(stem) for stem in _STEMMER.stemWords(words)]


def load_chunks(path: Path = CHUNKS) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def index_text(chunk: dict, doc_names: dict[str, str]) -> str:
    """Text that gets indexed: agreement name, section and heading give each chunk context.

    Shared by BM25 and the embedder so both modes see exactly the same input.
    """
    parts = [doc_names.get(chunk["boe_id"], ""), chunk["section"], chunk["label"], chunk["text"]]
    return "\n".join(p for p in parts if p)


def doc_names() -> dict[str, str]:
    return {d.boe_id: d.name for d in load_manifest()}


class BM25Index:
    """BM25 built in memory: ~1.3k chunks index in well under a second, so it is not saved."""

    def __init__(self, texts: list[str]):
        self.size = len(texts)
        self._bm25 = bm25s.BM25()
        self._bm25.index([tokenize(t) for t in texts], show_progress=False)

    def scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if not tokens:
            return np.zeros(self.size, dtype=np.float32)
        return self._bm25.get_scores(tokens).astype(np.float32)


class DenseIndex:
    """Brute-force cosine similarity over L2-normalised embeddings (a dot product)."""

    def __init__(self, embeddings: np.ndarray, encode_query: Callable[[str], np.ndarray]):
        self.embeddings = embeddings
        self.encode_query = encode_query

    def scores(self, query: str) -> np.ndarray:
        return self.embeddings @ self.encode_query(query)


def top_indices(scores: np.ndarray, mask: np.ndarray | None, n: int) -> list[int]:
    """Indices of the n best scores among allowed chunks; zero BM25 scores are not matches."""
    scores = scores.astype(np.float64, copy=True)
    if mask is not None:
        scores[~mask] = -np.inf
    order = np.argsort(-scores, kind="stable")[:n]
    return [int(i) for i in order if np.isfinite(scores[i])]


def rrf(rankings: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: score(d) = sum over rankings of 1 / (k + rank(d)).

    Uses ranks only, so BM25 and cosine scores need no calibration against each other.
    """
    fused: dict[int, float] = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, start=1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


@dataclass(frozen=True)
class Hit:
    rank: int
    score: float
    chunk: dict


class Retriever:
    def __init__(self, chunks: list[dict], bm25: BM25Index, dense: DenseIndex | None = None):
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self._boe_ids = np.array([c["boe_id"] for c in chunks])

    def search(
        self,
        query: str,
        k: int = 5,
        mode: str = "hybrid",
        boe_id: str | None = None,
        pool: int = 100,
        rrf_k: int = 60,
    ) -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode != "bm25" and self.dense is None:
            raise ValueError(f"mode '{mode}' needs a dense index")
        mask = self._boe_ids == boe_id if boe_id else None

        if mode == "bm25":
            scores = self.bm25.scores(query)
            ranked = [(i, float(scores[i])) for i in top_indices(scores, mask, k) if scores[i] > 0]
        elif mode == "dense":
            scores = self.dense.scores(query)
            ranked = [(i, float(scores[i])) for i in top_indices(scores, mask, k)]
        else:
            sparse = self.bm25.scores(query)
            sparse_rank = [i for i in top_indices(sparse, mask, pool) if sparse[i] > 0]
            dense_rank = top_indices(self.dense.scores(query), mask, pool)
            ranked = rrf([sparse_rank, dense_rank], k=rrf_k)[:k]
        return [Hit(r, s, self.chunks[i]) for r, (i, s) in enumerate(ranked, start=1)]
