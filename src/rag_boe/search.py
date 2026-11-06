"""Search CLI.

python -m rag_boe.search "¿Cuánto dura el periodo de prueba?" --mode hybrid -k 5
python -m rag_boe.search "plus de nocturnidad" --boe-id BOE-A-2023-13741
"""

from __future__ import annotations

import argparse
import sys

from rag_boe import embeddings
from rag_boe.retrieval import (
    MODES,
    BM25Index,
    DenseIndex,
    Retriever,
    doc_names,
    index_text,
    load_chunks,
)


def build_retriever(dense: bool = True) -> Retriever:
    chunks = load_chunks()
    names = doc_names()
    bm25 = BM25Index([index_text(c, names) for c in chunks])
    if not dense:
        return Retriever(chunks, bm25)
    vectors, model_name = embeddings.load([c["chunk_id"] for c in chunks])
    model = embeddings.load_model(model_name)
    return Retriever(chunks, bm25, DenseIndex(vectors, embeddings.query_encoder(model)))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=MODES, default="hybrid")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--boe-id", default=None, help="restrict to one agreement")
    args = parser.parse_args()

    retriever = build_retriever(dense=args.mode != "bm25")
    for hit in retriever.search(args.query, k=args.k, mode=args.mode, boe_id=args.boe_id):
        c = hit.chunk
        pages = f"p. {c['page_start']}" + (
            f"-{c['page_end']}" if c["page_end"] != c["page_start"] else ""
        )
        print(f"{hit.rank}. [{hit.score:.4f}] {c['boe_id']} · {c['label'][:80]} · BOE {pages}")
        print(f"   {c['chunk_id']} | {c['text'][:160].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
