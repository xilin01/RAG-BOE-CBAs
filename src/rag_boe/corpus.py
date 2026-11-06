"""Corpus manifest (data/corpus.csv) and PDF download."""

from __future__ import annotations

import csv
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "data" / "corpus.csv"
RAW_DIR = ROOT / "data" / "raw"

# Generic client name: never send personal data to third-party servers.
USER_AGENT = "rag-boe-cbas/0.1 (+https://github.com/xilin01/RAG-BOE-CBAs)"


@dataclass(frozen=True)
class Document:
    boe_id: str
    sector: str
    doc_type: str
    name: str
    published: str
    valid_until: str
    pdf_url: str

    @property
    def pdf_path(self) -> Path:
        return RAW_DIR / f"{self.boe_id}.pdf"


def load_manifest(path: Path = MANIFEST) -> list[Document]:
    with path.open(encoding="utf-8", newline="") as f:
        return [Document(**row) for row in csv.DictReader(f)]


def download(doc: Document, retries: int = 1) -> Path:
    """Download the PDF once; later runs reuse the local copy."""
    if doc.pdf_path.exists():
        return doc.pdf_path
    doc.pdf_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(doc.pdf_url, headers={"User-Agent": USER_AGENT})
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data.startswith(b"%PDF"):
                raise ValueError(f"{doc.boe_id}: response is not a PDF")
            doc.pdf_path.write_bytes(data)
            return doc.pdf_path
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2)
    raise AssertionError("unreachable")
