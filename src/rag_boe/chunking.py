"""Paragraphs -> legal units (article, disposition, annex) -> retrieval chunks.

Unit ids are deterministic and human-readable because the evaluation gold set references
them (e.g. ``art-14``, ``art-74bis``, ``dt-1``, ``anexo-II``, ``libro-3-art-1``). Chunk ids
are ``<boe_id>:<unit_id>:<part>``, so changing the chunk size never invalidates the gold set.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rag_boe.parse import Paragraph

ARTICLE = re.compile(r"^Art[íi]culo (\d+)(?: (bis|ter))?\. ?(.*)$")
DISPOSITION = re.compile(
    r"^Disposici[óo]n (adicional|transitoria|final|derogatoria|preliminar)(?: (\w+))?\. ?(.*)$",
    re.IGNORECASE,
)
ANNEX = re.compile(r"^ANEXO(?: ([IVXLC]+|ÚNICO|\d+))?(?: (.*))?$")
BOOK = re.compile(r"^LIBRO (PRIMERO|SEGUNDO|TERCERO|CUARTO|QUINTO)\b")
# Chapter/title headings carry no content of their own; they are dropped from unit bodies.
SECTION = re.compile(r"^(CAP[ÍI]TULO|T[ÍI]TULO|SECCI[ÓO]N|LIBRO) [\wÍÚ]+\.?(?: .*)?$")

DISPOSITION_CODES = {
    "adicional": "da",
    "transitoria": "dt",
    "final": "df",
    "derogatoria": "dd",
    "preliminar": "dp",
}
ORDINALS = {
    "primera": 1, "segunda": 2, "tercera": 3, "cuarta": 4, "quinta": 5, "sexta": 6,
    "séptima": 7, "septima": 7, "octava": 8, "novena": 9, "décima": 10, "decima": 10,
    "undécima": 11, "decimoprimera": 11, "duodécima": 12, "decimosegunda": 12,
    "decimotercera": 13, "decimocuarta": 14, "decimoquinta": 15, "decimosexta": 16,
    "decimoséptima": 17, "decimoctava": 18, "decimonovena": 19, "vigésima": 20,
    "única": "unica", "unica": "unica",
}  # fmt: skip
BOOKS = {"PRIMERO": 1, "SEGUNDO": 2, "TERCERO": 3, "CUARTO": 4, "QUINTO": 5}

# Max title length taken from a heading paragraph; longer text is body.
_MAX_TITLE = 160
_TITLE = re.compile(rf"^(.{{1,{_MAX_TITLE}}}?\.)(?: (.*))?$")


@dataclass
class Unit:
    unit_id: str
    unit_type: str  # preamble | article | disposition | annex
    label: str  # e.g. "Artículo 14. Plazo y procedimiento..."
    section: str = ""  # enclosing chapter/section title, e.g. "CAPÍTULO III. Contratación"
    paragraphs: list[Paragraph] = field(default_factory=list)


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    boe_id: str
    unit_id: str
    unit_type: str
    label: str
    section: str
    part: int
    n_parts: int
    page_start: int  # official BOE page
    page_end: int
    pdf_page_start: int
    pdf_page_end: int
    has_table: bool
    text: str


def _split_title(rest: str) -> tuple[str, str]:
    """Split heading remainder into (title, body) at the first full stop."""
    match = _TITLE.match(rest)
    if not match:
        return rest[:_MAX_TITLE], rest[_MAX_TITLE:].strip()
    return match.group(1), (match.group(2) or "").strip()


def _is_heading_like(para: Paragraph) -> bool:
    """Short line with no closing punctuation: a chapter or section title, not body text."""
    text = para.text
    return (
        not para.in_table
        and len(text) < 100
        and text[0].isupper()
        and not re.search(r"[.:;,)]$", text)
    )


def segment(paragraphs: list[Paragraph]) -> list[Unit]:
    """Group paragraphs into legal units. Text before the first heading is the preamble."""
    units = [Unit("preamble", "preamble", "Preámbulo y resolución de registro")]
    book: int | None = None
    annex: str | None = None
    scope = ""  # prefix for article ids when numbering restarts (books, annexes)
    seen: set[str] = set()  # article ids in the current scope
    disp_scope = ""  # prefix for disposition ids; follows the article scope
    last_ordinal: dict[str, int] = {}  # last disposition ordinal per kind, to spot quotes
    used: set[str] = set()  # every unit id emitted, to keep ids unique
    section = ""
    section_open = False  # a chapter number was seen; its title may follow in the next block
    title_open = False  # the last heading's title wraps into the next block

    def start(unit_id: str, unit_type: str, label: str, body: str, para: Paragraph) -> None:
        nonlocal section, title_open
        previous = units[-1]
        if previous.unit_type in ("article", "disposition"):
            # Section titles printed just before this heading are not part of the last unit.
            trailing = []
            while previous.paragraphs and _is_heading_like(previous.paragraphs[-1]):
                trailing.insert(0, previous.paragraphs.pop().text)
            if trailing:
                section = " ".join(trailing)
        if unit_id in used:  # last resort so ids stay unique; the report counts these
            n = 2
            while f"{unit_id}-{n}" in used:
                n += 1
            unit_id = f"{unit_id}-{n}"
        used.add(unit_id)
        unit = Unit(unit_id, unit_type, label.strip(), section)
        if body:
            unit.paragraphs.append(Paragraph(body, para.pdf_page, para.boe_page, para.in_table))
        units.append(unit)
        title_open = not body and (
            not label.endswith(".") if unit_type != "annex" else "." not in label
        )

    for para in paragraphs:
        text = para.text
        if title_open:
            title_open = False
            unit = units[-1]
            if unit.unit_type == "annex":
                if len(text) < _MAX_TITLE and not para.in_table:
                    unit.label = f"{unit.label}. {text}"
                    continue
            elif _TITLE.match(text):
                # Article titles always end with a full stop, so the rest of it is here.
                title, body = _split_title(text)
                unit.label = f"{unit.label} {title}"
                if body:
                    unit.paragraphs.append(
                        Paragraph(body, para.pdf_page, para.boe_page, para.in_table)
                    )
                continue
        if section_open:
            section_open = False
            if _is_heading_like(para):
                section = f"{section}. {text}"
                continue
        if m := BOOK.match(text):
            book = BOOKS[m.group(1)]
            continue
        if annex is None and SECTION.match(text) and len(text) < 200:
            section = text
            section_open = len(text.split()) <= 2  # bare "CAPÍTULO III": title comes next
            continue
        if m := ANNEX.match(text) if len(text) < 200 else None:
            annex = m.group(1) or "unico"
            section = ""  # chapter titles of the articulated text do not apply to annexes
            scope, seen, last_ordinal = f"anexo-{annex}-", set(), {}
            label = f"ANEXO {m.group(1) or ''}".strip() + (f". {m.group(2)}" if m.group(2) else "")
            start(f"anexo-{annex}", "annex", label, "", para)
            continue
        if m := ARTICLE.match(text):
            number = m.group(1) + (m.group(2) or "")
            if f"{scope}art-{number}" in seen:
                # Numbering restarted outside an annex: scope by book (or a counter fallback).
                scope = f"libro-{book}-" if book else f"parte-{len(seen)}-"
                seen = set()
            unit_id = f"{scope}art-{number}"
            seen.add(unit_id)
            disp_scope = scope
            title, body = _split_title(m.group(3))
            label = f"Artículo {m.group(1)}{' ' + m.group(2) if m.group(2) else ''}. {title}"
            start(unit_id, "article", label.strip(), body, para)
            continue
        if (m := DISPOSITION.match(text)) and text[0] == "D":
            kind = DISPOSITION_CODES[m.group(1).lower()]
            ordinal_word = (m.group(2) or "").lower()
            ordinal = ORDINALS.get(ordinal_word, ordinal_word or "unica")
            if isinstance(ordinal, int):
                expected = last_ordinal.get(kind, 0) + 1
                if ordinal not in (1, expected):
                    # Out-of-sequence heading: a disposition quoted inside another one.
                    units[-1].paragraphs.append(para)
                    continue
                last_ordinal[kind] = ordinal
            base = f"{kind}-{ordinal}"
            unit_id = f"{disp_scope}{base}"
            if unit_id in used and disp_scope:
                # Dispositions restarted after a scoped block: they belong to the main text.
                disp_scope, unit_id = "", base
            title, body = _split_title(m.group(3))
            heading = f"Disposición {m.group(1).lower()} {ordinal_word}".strip()
            start(unit_id, "disposition", f"{heading}. {title}", body, para)
            continue
        units[-1].paragraphs.append(para)

    return [u for u in units if u.paragraphs or u.unit_type != "preamble"]


def _split_long(text: str, max_chars: int) -> list[str]:
    """Split one oversized paragraph at sentence boundaries, hard-cutting if needed."""
    pieces, current = [], ""
    for sentence in re.split(r"(?<=[.;:])\s+", text):
        while len(sentence) > max_chars:
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:]
        if current and len(current) + len(sentence) + 1 > max_chars:
            pieces.append(current)
            current = sentence
        else:
            current = f"{current} {sentence}".strip()
    if current:
        pieces.append(current)
    return pieces


def chunk_units(boe_id: str, units: list[Unit], max_chars: int = 2000) -> list[Chunk]:
    """Pack each unit's paragraphs greedily into chunks of at most ``max_chars``.

    Chunks never cross unit boundaries, so every chunk maps to exactly one gold unit.
    """
    chunks: list[Chunk] = []
    for unit in units:
        groups: list[list[Paragraph]] = [[]]
        size = 0
        for para in unit.paragraphs:
            parts = (
                [para]
                if len(para.text) <= max_chars
                else [
                    Paragraph(t, para.pdf_page, para.boe_page, para.in_table)
                    for t in _split_long(para.text, max_chars)
                ]
            )
            for p in parts:
                if groups[-1] and size + len(p.text) + 1 > max_chars:
                    groups.append([])
                    size = 0
                groups[-1].append(p)
                size += len(p.text) + 1
        groups = [g for g in groups if g] or [[]]
        for i, group in enumerate(groups):
            pages = [p.boe_page for p in group] or [0]
            pdf_pages = [p.pdf_page for p in group] or [0]
            chunks.append(
                Chunk(
                    chunk_id=f"{boe_id}:{unit.unit_id}:{i}",
                    boe_id=boe_id,
                    unit_id=unit.unit_id,
                    unit_type=unit.unit_type,
                    label=unit.label,
                    section=unit.section,
                    part=i,
                    n_parts=len(groups),
                    page_start=min(pages),
                    page_end=max(pages),
                    pdf_page_start=min(pdf_pages),
                    pdf_page_end=max(pdf_pages),
                    has_table=any(p.in_table for p in group),
                    text="\n".join(p.text for p in group),
                )
            )
    return chunks


def article_report(units: list[Unit]) -> dict[str, dict]:
    """Per article scope: count, max number, missing numbers and duplicated ids.

    Gaps or duplicates signal a parsing problem (or a quirk of the source text).
    """
    report: dict[str, dict] = {}
    by_scope: dict[str, list[str]] = {}
    for unit in units:
        if unit.unit_type == "article":
            scope, _, number = unit.unit_id.rpartition("art-")
            by_scope.setdefault(scope or "main", []).append(number)
    for scope, numbers in by_scope.items():
        ints = {int(re.match(r"\d+", n).group()) for n in numbers}
        counts = Counter(numbers)
        report[scope] = {
            "count": len(numbers),
            "max": max(ints),
            "gaps": sorted(set(range(1, max(ints) + 1)) - ints),
            "duplicates": sorted(n for n, c in counts.items() if c > 1),
        }
    return report
