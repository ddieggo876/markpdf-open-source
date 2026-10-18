# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Pruebas del pipeline completo: PDF -> extracción -> limpieza -> métricas, y de los perfiles."""

import re

import pymupdf
import pytest

import extraction
import main
from options import MODES
from tests import pdf_fixtures as fx

LONG_DOTS = re.compile(r"\.{6,}")


@pytest.fixture(autouse=True)
def no_ocr_engine(monkeypatch):
    """Resultados deterministas: por defecto se simula que no hay motor de OCR instalado."""
    monkeypatch.setattr(extraction, "_ocr_available", lambda: False)


# ---------------------------------------------------------------------------
# Contrato: campos en blanco, datos conservados, cabeceras/pies repetidos
# ---------------------------------------------------------------------------
def test_contract_removes_repeated_headers_but_keeps_data_and_blank_fields():
    res = main.convert_pdf(fx.contract_pdf(5), mode="clean")
    md = res["markdown"]

    assert not LONG_DOTS.search(md)
    assert "CONTRATO DE TRABAJO - ACME S.L." not in md  # cabecera repetida
    assert not re.search(r"Pagina \d+ de 5", md)  # pie con numeración
    # Datos importantes intactos en las 5 páginas.
    for i in range(1, 6):
        assert f"CLAUSULA {i}" in md
        assert f"sede numero {i}" in md
    assert md.count("20/04/2026") == 5
    assert md.count("1.500,00 EUR") == 5
    assert md.count("___") == 5  # el relleno de puntos es un campo en blanco: no se borra, se normaliza
    assert res["stats"]["margin_lines_removed"] == 10  # 5 cabeceras + 5 pies
    assert res["stats"]["fillers_removed"] == 5


def test_removed_margin_items_are_reported_with_pages():
    res = main.convert_pdf(fx.contract_pdf(5), mode="clean")
    kinds = {(m["kind"], m["text"]): m["pages"] for m in res["report"]["margins"]}
    assert kinds[("header", "CONTRATO DE TRABAJO - ACME S.L.")] == [1, 2, 3, 4, 5]
    assert any(k == "page_number" for k, _ in kinds)
    assert len([r for r in res["report"]["removed"] if r["kind"] == "margin"]) == 10


def test_faithful_mode_keeps_headers_and_footers_but_normalizes_fillers():
    res = main.convert_pdf(fx.contract_pdf(5), mode="faithful")
    md = res["markdown"]
    assert md.count("CONTRATO DE TRABAJO - ACME S.L.") == 5
    assert len(re.findall(r"Pagina \d+ de 5", md)) == 5
    assert not LONG_DOTS.search(md)
    assert md.count("___") == 5
    assert res["stats"]["margin_lines_removed"] == 0
    assert res["report"]["margins"] == []


def test_two_page_document_does_not_treat_content_as_header():
    # Con menos de 3 páginas nada se considera "repetido"; solo números de página con palabra clave.
    res = main.convert_pdf(fx.contract_pdf(2), mode="clean")
    assert "CONTRATO DE TRABAJO - ACME S.L." in res["markdown"]
    assert "Pagina 1 de 2" not in res["markdown"]


def test_detect_margin_noise_is_literal_per_page():
    doc = pymupdf.open(stream=fx.contract_pdf(4), filetype="pdf")
    noise = main.detect_margin_noise(doc)
    assert len(noise) == 4
    for i, literals in enumerate(noise, start=1):
        assert "CONTRATO DE TRABAJO - ACME S.L." in literals
        assert f"Pagina {i} de 4" in literals
        assert not any("CLAUSULA" in x for x in literals)


# ---------------------------------------------------------------------------
# Columnas / orden de lectura
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_two_columns_are_read_one_after_the_other(mode):
    md = main.convert_pdf(fx.two_column_pdf(), mode=mode)["markdown"]
    words = re.findall(r"(?:alfa|beta)\d+", md)
    assert len(words) == 240
    assert words == [f"alfa{i}" for i in range(1, 121)] + [f"beta{i}" for i in range(1, 121)]


# ---------------------------------------------------------------------------
# Tablas (el detalle de formatos está en test_features.py)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_tables_survive_as_markdown_tables(mode):
    md = main.convert_pdf(fx.table_pdf(), mode=mode)["markdown"]
    rows = [ln for ln in md.splitlines() if ln.startswith("|")]
    assert len(rows) >= 5  # cabecera + separador + 3 filas
    joined = "\n".join(rows)
    for cell in ("Concepto", "Importe", "Salario base", "1200", "Plus transporte", "150", "Antiguedad", "75"):
        assert cell in joined


# ---------------------------------------------------------------------------
# Documentos vacíos o escaneados
# ---------------------------------------------------------------------------
def test_empty_document_returns_empty_markdown_with_warning():
    res = main.convert_pdf(fx.empty_pdf())
    assert res["markdown"] == ""
    assert res["metrics"]["tokens"] == 0
    assert any("No se encontró texto" in w for w in res["warnings"])


def test_scanned_document_warns_when_no_ocr_engine_is_available():
    res = main.convert_pdf(fx.scanned_pdf())
    assert res["markdown"] == ""
    assert any("escaneada" in w and "OCR" in w for w in res["warnings"])
    assert not any("No se encontró texto" in w for w in res["warnings"])
    assert res["stats"]["scanned_pages_without_text"] == 1


def test_scanned_page_inside_a_text_document_leaves_an_honest_placeholder():
    res = main.convert_pdf(fx.mixed_scanned_pdf())
    assert "Pagina con texto seleccionable y normal." in res["markdown"]
    assert "[Página sin texto extraíble (probable escaneo): página 2]" in res["markdown"]
    assert any("escaneada" in w for w in res["warnings"])


def test_scanned_document_uses_ocr_only_when_engine_is_available(monkeypatch):
    calls = []
    real = extraction._call_to_markdown

    def fake(doc, **opts):
        if opts.get("force_ocr"):
            calls.append(opts["pages"])
            return [{"text": "Texto reconocido por OCR\n"} for _ in opts["pages"]]
        return real(doc, **opts)

    monkeypatch.setattr(extraction, "_ocr_available", lambda: True)
    monkeypatch.setattr(extraction, "_call_to_markdown", fake)
    res = main.convert_pdf(fx.scanned_pdf())
    assert calls == [[0]]
    assert "Texto reconocido por OCR" in res["markdown"]
    assert any("Se aplicó OCR" in w for w in res["warnings"])
    assert res["stats"]["ocr_pages"] == 1


def test_ocr_is_never_invoked_for_documents_with_text(monkeypatch):
    def boom():
        raise AssertionError("no debería consultarse el OCR en un PDF con texto")

    monkeypatch.setattr(extraction, "_ocr_available", boom)
    res = main.convert_pdf(fx.contract_pdf(3))
    assert "CLAUSULA 1" in res["markdown"]
    assert res["warnings"] == []


# ---------------------------------------------------------------------------
# Documento ya limpio
# ---------------------------------------------------------------------------
def test_clean_pdf_content_is_unchanged_by_cleaning():
    res = main.convert_pdf(fx.clean_pdf(), mode="clean")
    faithful = main.convert_pdf(fx.clean_pdf(), mode="faithful")
    # Solo cambia el reflujo de líneas del párrafo: las palabras son exactamente las mismas.
    assert res["markdown"].split() == faithful["markdown"].split()
    stats = res["stats"]
    assert stats["pages"] == 1 and stats["margin_lines_removed"] == 0 and stats["fillers_removed"] == 0
    assert stats["images"] == 0 and stats["tables"] == 0 and stats["duplicates_removed"] == 0
    assert res["report"]["removed"] == []
    change = res["metrics"]["tokens_change_percent"]
    assert change in (0.0, None) or abs(change) < 3


# ---------------------------------------------------------------------------
# Métricas: fuentes, fórmulas y honestidad
# ---------------------------------------------------------------------------
def test_metrics_are_consistent_and_state_their_reference():
    res = main.convert_pdf(fx.contract_pdf(5), mode="clean")
    m = res["metrics"]
    md = res["markdown"]
    assert m["chars"] == len(md) == m["final"]["chars"]
    assert m["words"] == len(md.split()) == m["final"]["words"]
    assert m["tokens"] == -(-len(md) // 4) == m["final"]["tokens"]
    assert m["tokens_exact"] is False and "4" in m["token_estimator"]
    # La referencia es la extracción antes de limpiar, medida con el mismo estimador.
    assert m["tokens_before"] == m["extraction"]["tokens"] > m["tokens"]
    cmp_ = m["comparison"]
    assert cmp_["reference_tokens"] == m["tokens_before"] and cmp_["final_tokens"] == m["tokens"]
    assert cmp_["change_percent"] == m["tokens_change_percent"] == round((m["tokens"] - m["tokens_before"]) / m["tokens_before"] * 100, 1)
    assert m["tokens_change_percent"] < 0
    assert "extracción" in cmp_["reference"].lower() and "denominador" not in cmp_  # la fórmula lo explica
    assert "tokens_referencia" in cmp_["formula"]
    # El texto plano de la capa de texto del PDF es una fuente distinta y se informa aparte.
    assert m["source"]["chars"] > 0 and m["source"] != m["extraction"]


def test_metrics_report_an_increase_instead_of_inventing_savings(monkeypatch):
    from cleaning import CleanResult

    monkeypatch.setattr(main, "clean_pages", lambda pages, mode, margins, options=None: CleanResult("x" * 4000))
    res = main.convert_pdf(fx.clean_pdf(), mode="clean")
    m = res["metrics"]
    assert m["tokens"] == 1001  # 4000 caracteres + salto de línea final
    assert m["tokens_change_percent"] > 0


def test_profiles_are_ordered_by_size_without_losing_data():
    sizes = {m: main.convert_pdf(fx.contract_pdf(5), mode=m)["metrics"]["tokens"] for m in MODES}
    assert sizes["compact"] <= sizes["clean"] < sizes["faithful"]
    for mode in MODES:
        md = main.convert_pdf(fx.contract_pdf(5), mode=mode)["markdown"]
        assert md.count("1.500,00 EUR") == 5 and md.count("20/04/2026") == 5


def test_every_profile_returns_valid_markdown_for_every_fixture():
    from tests.markdown_checks import assert_valid_markdown

    docs = [
        fx.contract_pdf(5), fx.table_pdf(), fx.table_empty_cells_pdf(), fx.two_tables_pdf(), fx.contract_form_pdf(),
        fx.widget_form_pdf(), fx.duplicates_pdf(), fx.list_pdf(), fx.wrapped_pages_pdf(), fx.logo_pdf(4, with_figure=True),
        fx.hyphenation_pdf(),
    ]
    for pdf in docs:
        for mode in MODES:
            for table_format in ("auto", "markdown", "linear", "literal"):
                res = main.convert_pdf(pdf, mode=mode, table_format=table_format)
                assert_valid_markdown(res["markdown"])
