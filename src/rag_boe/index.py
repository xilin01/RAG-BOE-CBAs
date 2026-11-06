"""Build the dense index: python -m rag_boe.index [--model BAAI/bge-m3] [--batch-size 16]"""

from __future__ import annotations

import argparse
import sys
import time

from rag_boe import embeddings
from rag_boe.retrieval import doc_names, index_text, load_chunks


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=embeddings.MODEL)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    chunks = load_chunks()
    names = doc_names()
    texts = [index_text(c, names) for c in chunks]
    model = embeddings.load_model(args.model)
    print(f"model={args.model} device={model.device} chunks={len(chunks)}")
    print(f"index text tokens: {embeddings.token_stats(model, texts)}")
    start = time.perf_counter()
    vectors = embeddings.encode(model, texts, args.batch_size)
    print(f"encoded {vectors.shape} in {time.perf_counter() - start:.0f}s")
    embeddings.save(vectors, [c["chunk_id"] for c in chunks], args.model, embeddings.INDEX_DIR)
    print(f"saved -> {embeddings.INDEX_DIR}")


if __name__ == "__main__":
    main()
