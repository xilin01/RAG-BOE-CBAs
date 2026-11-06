"""Dense embeddings: build once (GPU if available), save, and verify on load.

The saved metadata pins model, dimension and the exact chunk order; loading refuses to run
if chunks.jsonl changed since the embeddings were built, so rows can never misalign.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np

from rag_boe.corpus import ROOT
from rag_boe.retrieval import CHUNKS

INDEX_DIR = ROOT / "indexes"
MODEL = "BAAI/bge-m3"  # multilingual, 1024-d, no query/passage prefixes needed


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_model(name: str = MODEL, device: str | None = None):
    # Lazy import: BM25-only code paths and the CI tests never load torch.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device=device)


def encode(model, texts: list[str], batch_size: int = 16) -> np.ndarray:
    vectors = model.encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=True
    )
    return np.asarray(vectors, dtype=np.float32)


def token_stats(model, texts: list[str]) -> dict[str, int]:
    """Token lengths with the model's own tokenizer (answers "does it fit in 512?")."""
    lengths = sorted(len(ids) for ids in model.tokenizer(texts)["input_ids"])
    return {
        "max": lengths[-1],
        "p95": lengths[int(0.95 * (len(lengths) - 1))],
        "p50": lengths[len(lengths) // 2],
        "over_512": sum(n > 512 for n in lengths),
    }


def save(vectors: np.ndarray, chunk_ids: list[str], model_name: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "embeddings.npy", vectors)
    meta = {
        "model": model_name,
        "dim": int(vectors.shape[1]),
        "normalized": True,
        "chunks_sha256": file_sha256(CHUNKS),
        "chunk_ids": chunk_ids,
    }
    (out_dir / "embeddings.meta.json").write_text(json.dumps(meta), encoding="utf-8")


def load(chunk_ids: list[str], out_dir: Path = INDEX_DIR, chunks_path: Path = CHUNKS):
    """Return (vectors, model_name) after checking they match the current chunks."""
    meta = json.loads((out_dir / "embeddings.meta.json").read_text(encoding="utf-8"))
    vectors = np.load(out_dir / "embeddings.npy")
    if meta["chunk_ids"] != chunk_ids or meta["chunks_sha256"] != file_sha256(chunks_path):
        raise RuntimeError("embeddings are stale: run `python -m rag_boe.index` again")
    if vectors.shape != (len(chunk_ids), meta["dim"]):
        raise RuntimeError(f"unexpected embeddings shape {vectors.shape}")
    return vectors, meta["model"]


def query_encoder(model) -> Callable[[str], np.ndarray]:
    def encode_query(query: str) -> np.ndarray:
        vector = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        return np.asarray(vector, dtype=np.float32)

    return encode_query
