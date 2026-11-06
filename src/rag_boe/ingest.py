"""Ingestion CLI: download PDFs, parse, segment into legal units and write chunks.jsonl.

Usage: python -m rag_boe.ingest [--max-chars 2000] [--review reports/phase1_chunk_review.md]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from rag_boe.chunking import Chunk, article_report, chunk_units, segment
from rag_boe.corpus import ROOT, download, load_manifest
from rag_boe.parse import parse_pdf

PROCESSED = ROOT / "data" / "processed"


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def stratified_sample(chunks: list[Chunk], seed: int = 0) -> list[Chunk]:
    """Pick review chunks covering every document and every hard case, not a uniform draw."""
    rng = random.Random(seed)
    picked: dict[str, Chunk] = {}

    def take(pool: list[Chunk], n: int = 1) -> None:
        pool = [c for c in pool if c.chunk_id not in picked]
        for c in rng.sample(pool, min(n, len(pool))):
            picked[c.chunk_id] = c

    for boe_id in sorted({c.boe_id for c in chunks}):
        take([c for c in chunks if c.boe_id == boe_id and c.unit_type == "article"], 2)
    take([c for c in chunks if c.unit_type == "preamble"])
    take([c for c in chunks if c.n_parts > 1 and c.unit_type == "article"])
    take([c for c in chunks if c.has_table])
    take([c for c in chunks if c.unit_type == "annex"])
    take([c for c in chunks if c.unit_type == "disposition"])
    take([c for c in chunks if c.page_start != c.page_end])
    take(chunks, 20 - len(picked))
    return sorted(picked.values(), key=lambda c: c.chunk_id)


def write_review(path: Path, sample: list[Chunk], reports: dict[str, dict]) -> None:
    lines = ["# Phase 1 - chunk review sample", ""]
    lines += ["## Article numbering check per document", ""]
    for boe_id, report in reports.items():
        lines.append(f"- `{boe_id}`: {json.dumps(report, ensure_ascii=False)}")
    lines.append("")
    for c in sample:
        flags = f"part {c.part + 1}/{c.n_parts}, BOE p. {c.page_start}-{c.page_end}"
        flags += ", table" if c.has_table else ""
        lines += [f"## `{c.chunk_id}`", "", f"**{c.label}** ({flags})", "", "```text"]
        lines += [c.text[:1500] + (" [...]" if len(c.text) > 1500 else ""), "```", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-chars", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=PROCESSED / "chunks.jsonl")
    parser.add_argument("--review", type=Path, default=None, help="write a review sample (md)")
    args = parser.parse_args()

    all_chunks: list[Chunk] = []
    reports: dict[str, dict] = {}
    for doc in load_manifest():
        units = segment(parse_pdf(download(doc)))
        chunks = chunk_units(doc.boe_id, units, args.max_chars)
        reports[doc.boe_id] = article_report(units)
        all_chunks += chunks
        types = Counter(u.unit_type for u in units)
        avg = sum(len(c.text) for c in chunks) // max(len(chunks), 1)
        print(f"{doc.boe_id}: units={dict(types)} chunks={len(chunks)} avg_chars={avg}")
        for scope, r in reports[doc.boe_id].items():
            print(f"    articles[{scope}]: {r}")

    ids = [c.chunk_id for c in all_chunks]
    assert len(ids) == len(set(ids)), "duplicate chunk ids"
    write_jsonl(args.out, [asdict(c) for c in all_chunks])
    print(f"wrote {len(all_chunks)} chunks -> {args.out}")
    if args.review:
        write_review(args.review, stratified_sample(all_chunks), reports)
        print(f"wrote review sample -> {args.review}")


if __name__ == "__main__":
    main()
