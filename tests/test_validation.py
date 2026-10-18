# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Validación del Markdown, cobertura de la extracción, opciones y garantías de fidelidad."""

import re
from collections import Counter

import pytest

import extraction
import main
import metrics
from options import (
    BOOL_OPTIONS,
    ConvertOptions,
    OptionsError,
    normalize_image_mode,
    normalize_mode,
    options_from_form,
    parse_bool,
)
from tests import pdf_fixtures as fx
from tests.markdown_checks import assert_valid_markdown
from validation import coverage_warning, extraction_coverage, validate_markdown


@pytest.fixture(autouse=True)
def no_ocr_engine(monkeypatch):
    monkeypatch.setattr(extraction, "_ocr_available", lambda: False)


# ---------------------------------------------------------------------------
# validate_markdown
# ---------------------------------------------------------------------------
def test_validate_closes_an_open_code_block():
    md, issues = validate_markdown("texto\n\n```text\nabierto\n")
    assert md.count("```") == 2 and any("bloque de código" in i for i in issues)


def test_validate_pads_ragged_table_rows_without_inventing_values():
    md, issues = validate_markdown("| a | b | c |\n|---|---|---|\n| 1 | 2 |\n")
    assert "| 1 | 2 |  |" in md and issues
    assert_valid_markdown(md)


def test_validate_adds_a_missing_separator_row():
    md, _ = validate_markdown("| a | b |\n| 1 | 2 |\n")
    assert "|---|---|" in md.splitlines()[1]


def test_validate_leaves_code_blocks_and_well_formed_tables_untouched():
    src = "| a | b |\n|---|---|\n| 1 | 2 |\n\n```text\n| no | es | tabla\n\n\n\nigual\n```\n"
    md, issues = validate_markdown(src)
    assert md == src and issues == []


def test_validate_removes_control_characters_only():
    md, issues = validate_markdown("hola\x00 mundo\x07\n")
    assert md == "hola mundo\n" and issues


def test_validate_empty_text():
    assert validate_markdown("") == ("", [])


# ---------------------------------------------------------------------------
# Cobertura de extracción
# ---------------------------------------------------------------------------
def test_coverage_is_complete_when_all_text_is_present():
    cov = extraction_coverage({0: "El importe es 1.200,00 EUR."}, ["El importe es **1.200,00 EUR**."])
    assert cov.ratio == 1.0 and coverage_warning(cov) == ""


def test_coverage_flags_possible_loss_with_examples_and_never_certainty():
    source = "Primera frase importante del contrato. Segunda frase con cifras 1234567 y fechas 20260420."
    cov = extraction_coverage({0: source}, ["Primera frase importante del contrato."])
    warning = coverage_warning(cov)
    assert cov.pages and cov.pages[0]["page"] == 1
    assert "Posible pérdida" in warning and "página" in warning
    assert "cifras" in warning or "1234567" in warning


def test_coverage_ignores_pages_without_source_text():
    cov = extraction_coverage({0: "   "}, [""])
    assert cov.checked_pages == 0 and cov.ratio == 1.0


def test_real_documents_report_no_loss_for_a_correct_extraction():
    for pdf in (fx.contract_pdf(5), fx.two_column_pdf(), fx.table_pdf(), fx.contract_form_pdf(), fx.widget_form_pdf()):
        res = main.convert_pdf(pdf)
        assert not any("Posible pérdida" in w for w in res["warnings"]), res["warnings"]
        assert res["report"]["coverage"]["coverage"] >= 0.98


def test_a_lossy_engine_triggers_the_loss_warning(monkeypatch):
    real = main.extract_pages

    def lossy(doc, opts, images_dir):
        ext = real(doc, opts, images_dir)
        ext.pages = [p[: len(p) // 3] for p in ext.pages]  # simula un motor que pierde dos tercios del texto
        return ext

    monkeypatch.setattr(main, "extract_pages", lossy)
    res = main.convert_pdf(fx.contract_pdf(3))
    assert any("Posible pérdida" in w for w in res["warnings"])


# ---------------------------------------------------------------------------
# Opciones
# ---------------------------------------------------------------------------
def test_content_discarding_options_are_off_in_every_profile():
    for mode in ("faithful", "clean", "compact"):
        o = ConvertOptions.for_mode(mode)
        assert not (o.omit_empty_fields or o.omit_redundant_headings or o.remove_repeated_blocks)


def test_default_profile_is_optimized_for_ai_and_images_are_marked():
    o = options_from_form({})
    assert o.mode == "clean" and o.image_mode == "markers" and o.table_format == "auto"


def test_faithful_profile_removes_nothing_by_default():
    o = ConvertOptions.for_mode("faithful")
    assert not (o.remove_headers_footers or o.remove_page_numbers or o.join_paragraphs or o.remove_duplicates)


def test_form_value_parsing():
    assert parse_bool("true") is True and parse_bool("off") is False and parse_bool("") is None
    assert parse_bool(None) is None
    with pytest.raises(OptionsError):
        parse_bool("tal vez")


def test_unknown_values_raise_options_error():
    with pytest.raises(OptionsError):
        normalize_mode("turbo")
    with pytest.raises(OptionsError):
        normalize_image_mode("magia")
    with pytest.raises(OptionsError):
        ConvertOptions.for_mode("clean", opcion_inventada=True)
    with pytest.raises(OptionsError):
        options_from_form({"table_format": "html"})


def test_every_boolean_option_can_be_set_from_a_form():
    for name in BOOL_OPTIONS:
        on = options_from_form({name: "true"})
        off = options_from_form({name: "false"})
        assert getattr(on, name) is True and getattr(off, name) is False


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def test_token_estimator_is_a_labelled_approximation():
    assert metrics.estimate_tokens("") == 0
    assert metrics.estimate_tokens("abcd") == 1 and metrics.estimate_tokens("abcde") == 2
    assert "caracteres / 4" in metrics.TOKEN_ESTIMATOR


def test_comparison_states_reference_and_formula_and_handles_zero_reference():
    c = metrics.comparison({"chars": 400, "words": 60, "tokens": 100}, {"chars": 300, "words": 45, "tokens": 75}, "ref")
    assert c["change_percent"] == -25.0 and c["reference"] == "ref" and c["formula"]
    zero = metrics.comparison({"chars": 0, "words": 0, "tokens": 0}, {"chars": 0, "words": 0, "tokens": 0}, "ref")
    assert zero["change_percent"] is None


# ---------------------------------------------------------------------------
# Garantías de fidelidad sobre PDFs reales: la limpieza nunca altera datos
# ---------------------------------------------------------------------------
_NUMBER_RE = re.compile(r"\d[\d.,/:+'\-]*\d|\d")
_NEGATION_RE = re.compile(r"\b(no|nunca|sin|ni|tampoco|jamas|NO)\b")


def _tokens(text: str, rx) -> Counter:
    return Counter(rx.findall(text))


@pytest.mark.parametrize(
    "factory",
    [
        lambda: fx.contract_form_pdf(),
        lambda: fx.table_pdf(),
        lambda: fx.two_tables_pdf(),
        lambda: fx.hyphenation_pdf(),
        lambda: fx.list_pdf(),
        lambda: fx.wrapped_pages_pdf(),
    ],
)
def test_cleaning_never_changes_figures_or_negations_compared_with_faithful(factory):
    pdf = factory()
    faithful = main.convert_pdf(pdf, mode="faithful", table_format="markdown")["markdown"]
    for mode in ("clean", "compact"):
        out = main.convert_pdf(pdf, mode=mode, table_format="markdown")["markdown"]
        # Las cifras son las mismas, sin añadidas ni perdidas (el guion de partición de palabra no es una cifra).
        assert _tokens(out, _NUMBER_RE) == _tokens(faithful, _NUMBER_RE)
        assert _tokens(out, _NEGATION_RE) == _tokens(faithful, _NEGATION_RE)


def test_cleaning_only_removes_what_it_reports_on_a_contract():
    pdf = fx.contract_pdf(5)
    faithful = main.convert_pdf(pdf, mode="faithful")
    clean = main.convert_pdf(pdf, mode="clean")
    removed_lines = [r["text"] for r in clean["report"]["removed"]]
    faithful_words = Counter(re.findall(r"\w+", faithful["markdown"]))
    clean_words = Counter(re.findall(r"\w+", clean["markdown"]))
    lost = faithful_words - clean_words
    reported = Counter(re.findall(r"\w+", " ".join(removed_lines)))
    # Todo lo que desaparece (más allá de rellenos de puntos) está en el informe de elementos eliminados.
    assert not (lost - reported), lost - reported
    # Y nada del contenido real del contrato desaparece.
    for must in ("CLAUSULA", "salario", "1.500,00", "20/04/2026", "sede"):
        assert clean_words[must] == faithful_words[must] or must not in faithful_words


def test_legal_references_survive_every_profile():
    pdf = fx.contract_form_pdf()
    for mode in ("faithful", "clean", "compact"):
        md = main.convert_pdf(pdf, mode=mode)["markdown"]
        assert "art. 17.2" in md and "Ley 29/1994" in md and "NO podra" in md
