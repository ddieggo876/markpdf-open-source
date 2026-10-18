# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Comprobaciones estáticas de la interfaz: todos los controles están cableados al backend."""

import json
import re
from pathlib import Path

import pytest

import main
import options as opts

HTML = (Path(__file__).resolve().parent.parent / "index.html").read_text(encoding="utf-8")


def js_const(name: str) -> str:
    m = re.search(rf"const {name} = (.+?);\n", HTML, re.S)
    assert m, f"no se encontró const {name} en index.html"
    return m.group(1)


def js_list(name: str) -> list[str]:
    return re.findall(r'"([^"]+)"', js_const(name))


def test_every_backend_boolean_option_has_a_control_and_is_sent():
    sent = set(js_list("BOOL_OPTIONS"))
    assert sent == set(opts.BOOL_OPTIONS)
    for name in sent:
        assert f'id="{name}"' in HTML, f"falta el control de {name}"


def test_every_select_option_is_sent_and_matches_backend_values():
    assert set(js_list("SELECT_OPTIONS")) == {"table_format", "image_mode"}
    table = re.search(r'<select id="table_format".*?</select>', HTML, re.S).group(0)
    assert re.findall(r'value="([^"]+)"', table) == list(opts.TABLE_FORMATS)
    images = re.search(r'<select id="image_mode".*?</select>', HTML, re.S).group(0)
    assert re.findall(r'value="([^"]+)"', images) == ["ignore", "markers", "extract", "describe"] == list(opts.IMAGE_MODES)


def test_form_fields_sent_by_the_ui_are_exactly_the_ones_the_endpoint_accepts():
    ui_fields = set(js_list("BOOL_OPTIONS")) | set(js_list("SELECT_OPTIONS"))
    endpoint = set(main._FORM_OPTION_NAMES) | {"image_mode"}
    assert ui_fields == endpoint


def test_profiles_in_the_ui_match_the_backend_profiles():
    radios = re.findall(r'name="mode" value="([^"]+)"', HTML)
    assert radios == list(opts.MODES)
    assert 'value="clean" class="peer sr-only" checked' in HTML  # «Optimizado para IA» por defecto
    ui_defaults = json.loads(js_const("PROFILE_DEFAULTS"))
    assert ui_defaults == opts._PROFILE_DEFAULTS
    for label in opts.PROFILE_LABELS.values():
        assert label in HTML


def test_content_discarding_options_start_unchecked_and_are_labelled():
    for name in ("omit_empty_fields", "omit_redundant_headings", "remove_repeated_blocks"):
        assert re.search(rf'id="{name}"[^>]*>', HTML) and "checked" not in re.search(rf'<input[^>]*id="{name}"[^>]*>', HTML).group(0)
    assert "solo si lo pides" in HTML


def test_vision_option_is_disabled_by_capabilities_not_hardcoded():
    assert "/capabilities" in HTML and "describeOption" in HTML and "no disponible" in HTML


def test_copy_download_and_preview_all_read_the_same_editor_content():
    assert "navigator.clipboard.writeText(text)" in HTML and "copyText(els.editor.value)" in HTML
    assert "markdown: els.editor.value" in HTML
    assert "marked.parse(md" in HTML and "const md = els.editor.value" in HTML


def test_ui_text_comes_from_the_pdf_only_via_textcontent():
    # Lo que procede del PDF (informe, avisos) nunca se inserta como HTML; solo la vista previa pasa por DOMPurify.
    assert HTML.count(".innerHTML") == 1 and "DOMPurify.sanitize" in HTML


def test_no_misleading_claims_remain():
    lowered = HTML.lower()
    for phrase in ("100% de precisión", "100 % de precisión", "sin errores de formato", "ahorro máximo", "gráfico/tabla visual detectada"):
        assert phrase not in lowered
    assert "no es un recuento exacto" in lowered or "no el recuento real" in lowered


def test_accessibility_basics_are_present():
    assert 'lang="es"' in HTML and 'role="alert"' in HTML and 'aria-live="polite"' in HTML
    assert "focus-visible" in HTML and "prefers-reduced-motion" in HTML
    for control in ("editor", "table_format", "image_mode"):
        assert f'for="{control}"' in HTML  # etiquetas asociadas
    assert 'name="viewport"' in HTML
    assert re.search(r'<fieldset[^>]*>\s*<legend', HTML)
