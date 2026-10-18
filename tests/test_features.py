# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Pruebas por funcionalidad: formularios, tablas, cabeceras/pies, párrafos, duplicados, imágenes."""

import io
import re
import zipfile

import pymupdf
import pytest

import extraction
import forms
import main
import tables
from tests import pdf_fixtures as fx
from tests.markdown_checks import assert_valid_markdown, table_groups


@pytest.fixture(autouse=True)
def no_ocr_engine(monkeypatch):
    monkeypatch.setattr(extraction, "_ocr_available", lambda: False)


def convert(pdf, **kw):
    return main.convert_pdf(pdf, **kw)


# ---------------------------------------------------------------------------
# Formularios y contratos
# ---------------------------------------------------------------------------
def test_contract_form_keeps_field_value_pairs_on_separate_lines():
    res = convert(fx.contract_form_pdf(), mode="clean")
    lines = res["markdown"].splitlines()
    assert "Nombre: Maria Lopez Garcia" in lines
    assert "DNI: 12345678Z" in lines
    assert "Direccion: Calle Mayor 5, 3 B" in lines
    assert "Email:" in lines  # etiqueta sin valor: no se rellena ni se inventa


def test_contract_form_blank_field_is_marked_not_deleted():
    md = convert(fx.contract_form_pdf(), mode="clean")["markdown"]
    assert "Telefono (opcional): ___" in md.splitlines()


def test_contract_form_never_alters_critical_data():
    for mode in ("faithful", "clean", "compact"):
        md = convert(fx.contract_form_pdf(), mode=mode)["markdown"]
        assert "NO podra subarrendar" in md  # negación intacta
        assert "850,00 EUR" in md
        assert "art. 17.2 de la Ley 29/1994" in md
        assert "12345678Z" in md


def test_checkbox_state_is_preserved_and_si_no_is_never_interpreted():
    md = convert(fx.contract_form_pdf(), mode="clean")["markdown"]
    assert "[X] Acepto las condiciones" in md
    assert "[ ] No acepto" in md and "[ ] Autorizo comunicaciones" in md


def test_plain_si_no_text_is_not_turned_into_a_checked_box():
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 100), "Contrato indefinido:   SI    NO", fontsize=11)
    page.insert_text((72, 130), "Trabajo a distancia:   SI    NO", fontsize=11)
    md = convert(doc.tobytes())["markdown"]
    assert "[x]" not in md.lower()
    assert md.count("SI") == 2 and md.count("NO") == 2


def test_acroform_widgets_become_explicit_states():
    res = convert(fx.widget_form_pdf(), mode="clean")
    md = res["markdown"]
    assert "Nombre completo: Juan Perez" in md
    assert "Telefono (opcional): ___" in md
    assert "Acepto condiciones [x]" in md
    assert "No acepto [ ]" in md
    assert res["stats"]["forms"] == {
        "fields": 4, "text_filled": 1, "text_empty": 1, "checked": 1, "unchecked": 1, "unknown": 0,
    }


def test_widget_conversion_does_not_modify_the_original_pdf():
    pdf = fx.widget_form_pdf()
    before = pymupdf.open(stream=pdf, filetype="pdf")
    n_before = len(list(before[0].widgets()))
    convert(pdf)
    after = pymupdf.open(stream=pdf, filetype="pdf")
    assert len(list(after[0].widgets())) == n_before == 4


def test_symbol_font_glyphs_become_checkbox_states():
    md = convert(fx.symbols_pdf())["markdown"]
    assert "Opcion marcada [x]" in md
    assert "Otra opcion [ ]" in md


def test_symbol_glyph_mapping_is_explicit_and_flags_unknown_codes():
    assert forms.normalize_checkbox_glyphs("\u2611 a \u2610 b") == "[x] a [ ] b"
    text, unknown = forms.map_span_text("\uf0fe", "Wingdings")
    assert text == "[x]" and unknown == 0
    text, unknown = forms.map_span_text("\uf0aa", "Wingdings")  # código sin significado de casilla conocido
    assert unknown == 1 and text == "[?]"


def test_omit_empty_fields_is_explicit_and_reported():
    default = convert(fx.contract_form_pdf())
    omitted = convert(fx.contract_form_pdf(), omit_empty_fields=True)
    assert "Telefono (opcional)" in default["markdown"]
    assert "Telefono" not in omitted["markdown"]
    assert omitted["stats"]["empty_fields_omitted"] == 1
    assert any(r["kind"] == "empty_field" for r in omitted["report"]["removed"])
    assert "Maria Lopez Garcia" in omitted["markdown"]  # los datos rellenados siguen estando


# ---------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------
EMPTY_CELLS_ROWS = [
    ["Concepto", "Importe", "Notas"],
    ["Salario base", "1200", ""],
    ["", "150", "Plus transporte"],
    ["Antiguedad", "", ""],
]


def _parse_first_table(md):
    group = table_groups(md)[0]
    return tables.parse_markdown_table("\n".join(group)).rows


def test_table_with_empty_cells_keeps_every_row_and_column():
    res = convert(fx.table_empty_cells_pdf(), table_format="markdown")
    assert _parse_first_table(res["markdown"]) == EMPTY_CELLS_ROWS
    assert res["stats"]["tables"] == 1 and res["stats"]["tables_approximate"] == 0


def test_text_after_a_table_is_not_absorbed_into_it():
    md = convert(fx.table_empty_cells_pdf())["markdown"]
    assert "Texto justo despues de la tabla que no debe ser celda." in md.splitlines()
    assert len(table_groups(md)[0]) == 5  # cabecera + separador + 3 filas


def test_two_tables_stay_independent():
    res = convert(fx.two_tables_pdf())
    groups = table_groups(res["markdown"])
    assert len(groups) == 2 and res["stats"]["tables"] == 2
    assert "Producto" in groups[0][0] and "Ciudad" in groups[1][0]
    assert "Entre las dos tablas hay un parrafo" in res["markdown"]


def test_linear_format_labels_every_cell_and_marks_empty_ones():
    md = convert(fx.table_empty_cells_pdf(), table_format="linear")["markdown"]
    assert "- Fila 2: Concepto: (vacío) | Importe: 150 | Notas: Plus transporte" in md
    assert "- Fila 3: Concepto: Antiguedad | Importe: (vacío) | Notas: (vacío)" in md
    assert not table_groups(md)


def test_linear_format_with_omit_empty_drops_only_empty_cells():
    md = convert(fx.table_empty_cells_pdf(), table_format="linear", omit_empty_fields=True)["markdown"]
    assert "(vacío)" not in md
    assert "- Fila 2: Importe: 150 | Notas: Plus transporte" in md
    assert "- Fila 3: Concepto: Antiguedad" in md


def test_literal_format_keeps_rows_in_a_code_block():
    md = convert(fx.table_empty_cells_pdf(), table_format="literal")["markdown"]
    assert "```text\nConcepto | Importe | Notas\n" in md
    assert "Antiguedad |  |" in md
    assert_valid_markdown(md)


def test_auto_format_uses_markdown_for_regular_tables_and_reports_it():
    res = convert(fx.two_tables_pdf())
    assert res["stats"]["table_formats"] == {"markdown": 2}


def test_auto_format_switches_to_linear_for_very_wide_tables():
    header = "|" + "|".join(f"C{i}" for i in range(12)) + "|"
    body = "|" + "|".join(str(i) for i in range(12)) + "|"
    md = f"{header}\n|{'|'.join(['---'] * 12)}|\n{body}\n"
    out = tables.render_table(md, "auto")
    assert out.format == "linear" and out.rows == 2 and out.cols == 12


def test_irregular_table_is_flagged_as_approximate_never_as_faithful():
    out = tables.render_table("| a | b | c |\n|---|---|---|\n| 1 | 2 |\n", "markdown")
    assert out.approximate is True
    assert "aproximada" in out.text and "distinto número de columnas" in out.text
    assert "| 1 | 2 |  |" in out.text  # se completa con vacío, sin inventar valores


def test_regular_table_is_never_labelled_as_approximate():
    out = tables.render_table("| a | b |\n|---|---|\n| 1 | 2 |\n", "markdown")
    assert out.approximate is False and "aproximada" not in out.text


def test_uninterpretable_table_is_preserved_with_a_note():
    out = tables.render_table("esto no es una tabla", "auto")
    assert out.approximate is True and "esto no es una tabla" in out.text and "no interpretable" in out.text


def test_table_cell_values_are_never_changed():
    md = convert(fx.table_pdf())["markdown"]
    for value in ("1200", "150", "75", "Salario base", "Plus transporte"):
        assert value in md


# ---------------------------------------------------------------------------
# Cabeceras, pies y numeración de página
# ---------------------------------------------------------------------------
def test_sequential_page_numbers_are_removed_but_numbers_in_text_stay():
    res = convert(fx.numbered_pdf(5))
    md = res["markdown"]
    assert not re.search(r"(?m)^\s*[1-5]\s*$", md)
    assert "articulo 1" in md.lower() or "Articulo 1" in md
    assert any(m["kind"] == "page_number" for m in res["report"]["margins"])


def test_constant_number_at_the_margin_is_kept_and_reported_as_uncertain():
    res = convert(fx.numbered_pdf(5, constant_number=True))
    assert res["markdown"].count("2026") == 5
    assert res["report"]["margins_uncertain"]


def test_varying_header_is_not_removed():
    res = convert(fx.varying_header_pdf())  # capítulos 3, 3, 7, 7, 9: el número no sigue a la página
    md = res["markdown"]
    for chapter, times in ((3, 2), (7, 2), (9, 1)):
        assert md.count(f"Capitulo {chapter} del reglamento") == times
    assert res["report"]["margins"] == [] and res["report"]["margins_uncertain"]


def test_text_repeated_in_the_body_is_not_treated_as_header():
    res = convert(fx.repeated_in_body_pdf())
    assert "ANEXO DE CONDICIONES" in res["markdown"]
    assert not any(m["text"] == "ANEXO DE CONDICIONES" for m in res["report"]["margins"])


def test_margin_removal_options_are_independent():
    pdf = fx.contract_pdf(5)
    no_numbers = convert(pdf, mode="clean", remove_page_numbers=False)["markdown"]
    no_headers = convert(pdf, mode="clean", remove_headers_footers=False)["markdown"]
    assert len(re.findall(r"Pagina \d+ de 5", no_numbers)) == 5 and "CONTRATO DE TRABAJO - ACME S.L." not in no_numbers
    assert no_headers.count("CONTRATO DE TRABAJO - ACME S.L.") == 5


def test_faithful_mode_can_opt_in_to_margin_removal():
    md = convert(fx.contract_pdf(5), mode="faithful", remove_headers_footers=True, remove_page_numbers=True)["markdown"]
    assert "CONTRATO DE TRABAJO - ACME S.L." not in md and "Pagina 1 de 5" not in md


# ---------------------------------------------------------------------------
# Párrafos, guiones, listas, duplicados
# ---------------------------------------------------------------------------
def test_hyphenated_words_are_rejoined_without_touching_compound_terms():
    md = convert(fx.hyphenation_pdf(), mode="clean")["markdown"]
    assert "contratacion" in md and "contra-" not in md
    assert "COVID-19" in md and "Decreto-ley" in md


def test_faithful_mode_keeps_the_original_line_structure():
    md = convert(fx.hyphenation_pdf(), mode="faithful")["markdown"]
    assert "contra-\ntacion" in md


def test_join_paragraphs_option_controls_reflow():
    on = convert(fx.hyphenation_pdf(), mode="faithful", join_paragraphs=True)["markdown"]
    off = convert(fx.hyphenation_pdf(), mode="clean", join_paragraphs=False)["markdown"]
    assert "contratacion" in on
    assert "contra-\ntacion" in off


def test_paragraph_split_across_pages_is_joined_only_when_continuation_is_evident():
    md = convert(fx.wrapped_pages_pdf(), mode="clean")["markdown"]
    paragraphs = [p.strip() for p in md.split("\n\n") if p.strip()]
    assert len(paragraphs) == 2  # el primer párrafo termina con punto: no se une al siguiente
    assert paragraphs[0].endswith("todos sus terminos.")
    assert paragraphs[1].endswith("toda la relacion y hasta su completa extincion segun lo previsto.")
    assert all("\n" not in p for p in paragraphs)


def test_nested_lists_keep_numbering_and_indentation():
    md = convert(fx.list_pdf(), mode="clean")["markdown"]
    assert "1. Primero\n  a) subitem uno\n  b) subitem dos\n2. Segundo" in md
    assert "- punto A\n  - punto B" in md


def test_duplicate_paragraph_removed_in_clean_but_signature_block_kept():
    res = convert(fx.duplicates_pdf(), mode="clean")
    md = res["markdown"]
    assert md.count("El arrendatario no podra subarrendar el inmueble") == 1
    assert md.count("IMPORTE TOTAL: 1.500,00 EUR") == 1
    assert "Firmado digitalmente por: GARCIA LOPEZ MARIA - 12345678Z" in md
    assert "Fecha: 2026.04.20 10:33:12 +02'00'" in md
    assert res["stats"]["duplicates_removed"] == 1
    assert any(r["kind"] == "duplicate_block" for r in res["report"]["removed"])


def test_duplicate_paragraphs_are_kept_in_faithful_mode_and_when_disabled():
    faithful = convert(fx.duplicates_pdf(), mode="faithful")["markdown"]
    disabled = convert(fx.duplicates_pdf(), mode="clean", remove_duplicates=False)["markdown"]
    for md in (faithful, disabled):
        assert md.count("El arrendatario no podra subarrendar el inmueble") == 2


# ---------------------------------------------------------------------------
# Imágenes
# ---------------------------------------------------------------------------
LOGO_MARKER = "[Imagen repetida omitida: páginas 1-5]"
FIGURE_MARKER = "[Imagen omitida: página 2]"


def test_image_markers_are_honest_and_group_repeated_images():
    res = convert(fx.logo_pdf(5, with_figure=True), image_mode="markers")
    md = res["markdown"]
    assert md.count(LOGO_MARKER) == 1 and md.count(FIGURE_MARKER) == 1
    for claim in ("gráfico", "grafico", "firma", "logo", "tabla", "fotografía"):  # no se afirma qué es la imagen
        assert claim not in md.lower().replace("figura", "")
    detail = res["stats"]["images_detail"]
    assert detail["marked"] == 2 and detail["extracted"] == 0 and detail["described"] == 0
    assert detail["detected"] == detail["relevant"] + detail["repeated"] + detail["small_ignored"]
    assert res["stats"]["images"] == 2


def test_image_markers_are_in_page_order():
    md = convert(fx.logo_pdf(5, with_figure=True))["markdown"]
    assert md.index(FIGURE_MARKER) < md.index("Figura 2: Evolucion mensual") < md.index("Texto de la pagina 3.")


def test_ignore_removes_markers_but_still_counts_what_was_omitted():
    res = convert(fx.logo_pdf(5, with_figure=True), image_mode="ignore")
    assert "[Imagen" not in res["markdown"]
    assert res["stats"]["images_detail"]["marked"] == 0
    assert res["stats"]["images_detail"]["omitted"] == 2


def test_ignore_with_markers_option_still_marks_positions():
    res = convert(fx.logo_pdf(5, with_figure=True), image_mode="ignore", image_markers=True)
    assert FIGURE_MARKER in res["markdown"] and LOGO_MARKER in res["markdown"]


def test_image_treatment_is_independent_of_the_text_profile():
    outputs = [convert(fx.logo_pdf(5, with_figure=True), mode=m)["markdown"] for m in ("faithful", "clean", "compact")]
    for md in outputs:
        assert FIGURE_MARKER in md and LOGO_MARKER in md


def test_extract_mode_references_only_files_that_exist_and_zips_them():
    res = convert(fx.logo_pdf(5, with_figure=True), image_mode="extract")
    md = res["markdown"]
    refs = re.findall(r"!\[[^\]]*\]\((images/[^)]+)\)", md)
    assert len(refs) == 2 and res["stats"]["images_detail"]["extracted"] == 2
    assert res["job_id"]
    out = main.download(main.DownloadRequest(markdown=md, filename="doc.md", job_id=res["job_id"]))
    with zipfile.ZipFile(io.BytesIO(out.body)) as zf:
        names = set(zf.namelist())
        assert "doc.md" in names and set(refs) <= names
        for ref in refs:
            assert zf.read(ref)[:8] == b"\x89PNG\r\n\x1a\n"
    main._gc_jobs()


def test_describe_is_rejected_because_no_vision_model_is_configured():
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as err:
        convert(fx.logo_pdf(2), image_mode="describe")
    assert err.value.status_code == 400 and "visión" in err.value.detail.lower()
    assert main.vision.is_available() is False


def test_document_without_images_reports_zero():
    res = convert(fx.clean_pdf())
    assert res["stats"]["images"] == 0 and res["stats"]["images_detail"]["detected"] == 0
