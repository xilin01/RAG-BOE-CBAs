import csv
import re
from datetime import date
from pathlib import Path

import rag_boe

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus.csv"
REQUIRED = ("boe_id", "sector", "doc_type", "name", "published", "pdf_url")
DOC_TYPES = {"agreement", "framework_agreement"}


def load_corpus() -> list[dict[str, str]]:
    with CORPUS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def test_package_imports():
    assert rag_boe.__version__


def test_corpus_manifest_is_well_formed():
    rows = load_corpus()
    assert len(rows) >= 7
    ids = [r["boe_id"] for r in rows]
    assert len(ids) == len(set(ids)), "duplicate BOE ids"
    for r in rows:
        assert all(r[field].strip() for field in REQUIRED), f"empty field in {r['boe_id']}"
        assert re.fullmatch(r"BOE-A-\d{4}-\d+", r["boe_id"])
        assert r["doc_type"] in DOC_TYPES
        published = date.fromisoformat(r["published"])
        if r["valid_until"]:
            assert date.fromisoformat(r["valid_until"]) > published
        # The canonical BOE PDF URL encodes the publication date and the BOE id.
        expected_url = f"https://www.boe.es/boe/dias/{published:%Y/%m/%d}/pdfs/{r['boe_id']}.pdf"
        assert r["pdf_url"] == expected_url
