import csv
from pathlib import Path

import rag_boe

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.csv"


def test_package_imports():
    assert rag_boe.__version__


def test_corpus_manifest_is_well_formed():
    with CORPUS.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 7
    ids = [r["boe_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate BOE ids"
    for r in rows:
        assert r["boe_id"].startswith("BOE-A-")
        assert r["pdf_url"].startswith("https://www.boe.es/")
