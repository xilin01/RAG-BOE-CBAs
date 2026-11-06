"""Chunker tests on synthetic paragraphs (no PDFs needed, so they run in CI)."""

from rag_boe.chunking import article_report, chunk_units, segment
from rag_boe.parse import Paragraph, clean_block, is_furniture


def paras(*texts: str, page: int = 1) -> list[Paragraph]:
    return [Paragraph(t, page, 1000 + page, False) for t in texts]


def by_id(units):
    return {u.unit_id: u for u in units}


def test_page_furniture_is_removed():
    lines = [
        "El personal tendrá derecho a 30 días",
        "BOLETÍN OFICIAL DEL ESTADO",
        "Núm. 137",
        "Viernes 9 de junio de 2023",
        "Sec. III.   Pág. 82467",
        "cve: BOE-A-2023-13741",
        "Verificable en https://www.boe.es",
    ]
    assert clean_block(lines) == "El personal tendrá derecho a 30 días"
    assert not is_furniture("13741")  # bare numbers are furniture only on the first page
    assert is_furniture("13741", first_page=True)


def test_compound_split_at_line_end_keeps_its_hyphen():
    assert clean_block(["el contrato fijo-", "discontinuo y el e- ", "learning"]) == (
        "el contrato fijo-discontinuo y el e-learning"
    )


def test_heading_only_units_produce_no_chunks():
    units = segment(paras("Artículo 1. Objeto.", "x", "ANEXO V"))
    chunks = chunk_units("BOE-A-2023-1", units)
    assert [c.unit_id for c in chunks] == ["art-1"]
    assert all(c.text and c.page_start > 0 for c in chunks)


def test_articles_preamble_and_inline_references():
    units = segment(
        paras(
            "Visto el texto del Convenio colectivo...",
            "Artículo 1. Ámbito funcional.",
            "Se aplica a todas las empresas del sector.",
            "Conforme al artículo 30.1 del Estatuto, los permisos...",  # not a heading
            "Artículo 2 bis. Teletrabajo. El teletrabajo será voluntario.",
        )
    )
    ids = by_id(units)
    assert list(ids) == ["preamble", "art-1", "art-2bis"]
    assert ids["art-1"].label == "Artículo 1. Ámbito funcional."
    assert len(ids["art-1"].paragraphs) == 2
    assert ids["art-2bis"].label == "Artículo 2 bis. Teletrabajo."
    assert ids["art-2bis"].paragraphs[0].text == "El teletrabajo será voluntario."


def test_wrapped_title_and_section_headings():
    units = segment(
        paras(
            "CAPÍTULO III",
            "Contratación",
            "Artículo 14. Plazo y procedimiento para el cumplimiento de los porcentajes de",
            "contratación. 1. Los porcentajes serán efectivos desde 2024.",
            "Pluses salariales",  # section title printed before the next article
            "Artículo 15. Plus de nocturnidad.",
            "Se abonará un 25 %.",
        )
    )
    ids = by_id(units)
    art14 = ids["art-14"]
    assert art14.label.endswith("de los porcentajes de contratación.")
    assert art14.section == "CAPÍTULO III. Contratación"
    assert [p.text for p in art14.paragraphs] == ["1. Los porcentajes serán efectivos desde 2024."]
    assert ids["art-15"].section == "Pluses salariales"


def test_disposition_ids_and_quoted_dispositions():
    units = segment(
        paras(
            "Artículo 1. Objeto.",
            "Texto.",
            "Disposición transitoria primera. Mantenimiento de disposiciones anteriores.",
            "Disposición transitoria tercera. Clasificación.",  # quoted, out of sequence
            "Disposición transitoria segunda. Jubilación.",
            "Disposición derogatoria única.",
            "Queda derogado el convenio anterior.",
        )
    )
    ids = by_id(units)
    assert [u for u in ids if u not in ("preamble", "art-1")] == ["dt-1", "dt-2", "dd-unica"]
    assert ids["dt-1"].paragraphs[0].text.startswith("Disposición transitoria tercera")


def test_restarted_numbering_is_scoped_by_book_and_annex():
    units = segment(
        paras(
            "LIBRO PRIMERO",
            "Artículo 1. Partes.",
            "a",
            "Artículo 2. Naturaleza.",
            "b",
            "LIBRO TERCERO",
            "Artículo 1. Objeto del plan.",
            "c",
            "ANEXO V",
            "Reglamento de la comisión",
            "Artículo 1. Composición.",
            "d",
        )
    )
    ids = by_id(units)
    assert {"art-1", "art-2", "libro-3-art-1", "anexo-V", "anexo-V-art-1"} <= set(ids)
    assert ids["anexo-V"].label == "ANEXO V. Reglamento de la comisión"
    report = article_report(units)
    assert report["main"] == {"count": 2, "max": 2, "gaps": [], "duplicates": []}


def test_article_report_flags_gaps():
    units = segment(paras("Artículo 1. A.", "x", "Artículo 3. C.", "y"))
    assert article_report(units)["main"]["gaps"] == [2]


def test_long_unit_is_split_without_crossing_units():
    sentence = "La jornada máxima será de cuarenta horas semanales. "
    body = [Paragraph(sentence * 10, page, 2000 + page, False) for page in (1, 2, 3)]
    units = segment(paras("Artículo 7. Jornada.") + body + paras("Artículo 8. Vacaciones.", "z"))
    chunks = chunk_units("BOE-A-2023-1", units, max_chars=1200)
    art7 = [c for c in chunks if c.unit_id == "art-7"]
    assert [c.chunk_id for c in art7] == [f"BOE-A-2023-1:art-7:{i}" for i in range(len(art7))]
    assert len(art7) > 1 and all(len(c.text) <= 1200 for c in art7)
    assert art7[0].page_start == 2001 and art7[-1].page_end == 2003
    assert {c.n_parts for c in art7} == {len(art7)}
    assert "Vacaciones" not in " ".join(c.text for c in art7)
