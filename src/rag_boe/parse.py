"""PDF -> clean paragraphs with page provenance.

BOE PDFs are single-column text. PyMuPDF text blocks map well to paragraphs, and every
structural heading (article, disposition, annex) starts its own block, so blocks are the
unit we work with. Page furniture (header, footer, verification code) is removed per block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

# Lines printed on every BOE page that are not part of the agreement text.
_FURNITURE = [
    re.compile(r"^BOLETÍN OFICIAL DEL ESTADO$"),
    re.compile(r"^Núm\. \d+$"),
    re.compile(r"^(Lunes|Martes|Miércoles|Jueves|Viernes|Sábado|Domingo) \d{1,2} de \w+ de \d{4}$"),
    re.compile(r"^Sec\. III\.\s+Pág\. \d+$"),
    re.compile(r"^cve: BOE-A-\d{4}-\d+$"),
    re.compile(r"^(Verificable en )?https://www\.boe\.es$"),
    re.compile(r"^Verificable en$"),
    re.compile(r"^D\. L\.: M-1/1958 - ISSN: 0212-033X$"),
]
# Lines that only appear at the top of the first page of a BOE disposition.
_FIRST_PAGE_FURNITURE = [
    re.compile(r"^III\. OTRAS DISPOSICIONES$"),
    re.compile(r"^MINISTERIO DE .+$"),
    re.compile(r"^\d{3,6}$"),  # disposition number
]
_BOE_PAGE = re.compile(r"Sec\. III\.\s+Pág\. (\d+)")
_HYPHEN_BREAK = re.compile(r"([a-záéíóúñü])-\n([a-záéíóúñü])")


@dataclass(frozen=True)
class Paragraph:
    text: str
    pdf_page: int  # 1-based page index inside the PDF
    boe_page: int  # official BOE page number ("Pág."), the one to cite
    in_table: bool


def is_furniture(line: str, first_page: bool = False) -> bool:
    line = line.strip()
    patterns = _FURNITURE + (_FIRST_PAGE_FURNITURE if first_page else [])
    return any(p.match(line) for p in patterns)


def clean_block(lines: list[str], first_page: bool = False) -> str:
    """Drop page furniture, re-join hyphenated words and collapse whitespace."""
    kept = [line for line in lines if line.strip() and not is_furniture(line, first_page)]
    text = "\n".join(kept)
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = text.replace(" ", " ").replace(" ", " ")
    return re.sub(r"\s+", " ", text).strip()


def _overlaps(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def parse_pdf(path: Path, detect_tables: bool = True) -> list[Paragraph]:
    paragraphs: list[Paragraph] = []
    with pymupdf.open(path) as pdf:
        for index, page in enumerate(pdf):
            match = _BOE_PAGE.search(page.get_text())
            if not match:
                raise ValueError(f"{path.name}: no BOE page number on PDF page {index + 1}")
            boe_page = int(match.group(1))
            tables = [t.bbox for t in page.find_tables().tables] if detect_tables else []
            for x0, y0, x1, y1, raw, _no, block_type in page.get_text("blocks"):
                if block_type != 0:  # skip image blocks
                    continue
                # Furniture lines at the top of the first page belong to the BOE header.
                text = clean_block(raw.splitlines(), first_page=index == 0 and y0 < 200)
                if not text:
                    continue
                in_table = any(_overlaps((x0, y0, x1, y1), t) for t in tables)
                paragraphs.append(Paragraph(text, index + 1, boe_page, in_table))
    return paragraphs
