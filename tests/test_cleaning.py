# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Pruebas unitarias de cleaning.py (solo texto: no necesitan PDFs)."""

import re

import pytest

from options import ConvertOptions
from cleaning import (
    MODES,
    clean_markdown,
    clean_pages,
    count_words,
    estimate_tokens,
    normalize_chars,
    remove_margin_lines,
)

LONG_DOTS = re.compile(r"\.{6,}")


# ---------------------------------------------------------------------------
# Regresión: fragmento real de contrato
# ---------------------------------------------------------------------------
CONTRACT_INPUT = (
    "CUARTA : la duración del presente contrato será INDEFINIDA, iniciándose la relación laboral "
    "en fecha .............................., y se establece un período de 20/04/2026 prueba de (14) "
    "................................................................................ TRES MESES"
)


@pytest.mark.parametrize("mode", MODES)
def test_regression_contract_fragment(mode):
    out = clean_markdown(CONTRACT_INPUT, mode)

    # Las secuencias largas de puntos desaparecen.
    assert not LONG_DOTS.search(out)
    # Se conservan los datos importantes.
    for keep in ("CUARTA", "INDEFINIDA", "20/04/2026", "(14)", "TRES MESES"):
        assert keep in out
    # No se inventa ni se reordena nada: mismas palabras y en el mismo orden. Cada campo en blanco
    # (relleno de puntos) pasa a ser el marcador `___`: "fecha ....." ≠ "fecha" (hay un hueco sin rellenar).
    assert out.split() == (
        "CUARTA : la duración del presente contrato será INDEFINIDA, iniciándose la relación laboral "
        "en fecha ___, y se establece un período de 20/04/2026 prueba de (14) ___ TRES MESES"
    ).split()
    # Espacios y puntuación normalizados.
    assert "  " not in out
    assert " ," not in out
    assert "fecha ___, y se establece" in out
    assert "(14) ___ TRES MESES" in out


def test_regression_exact_output():
    assert clean_markdown(CONTRACT_INPUT, "clean") == (
        "CUARTA : la duración del presente contrato será INDEFINIDA, iniciándose la relación laboral "
        "en fecha ___, y se establece un período de 20/04/2026 prueba de (14) ___ TRES MESES\n"
    )


@pytest.mark.parametrize("mode", MODES)
def test_blank_fields_are_never_silently_dropped(mode):
    """'Nombre: ......' (campo vacío) no debe parecerse a 'Nombre:' seguido de otra etiqueta."""
    out = clean_markdown("Nombre: ........................\nDNI: 12345678A\n", mode)
    assert "Nombre: ___" in out
    assert "DNI: 12345678A" in out


def test_omit_empty_fields_is_explicit_and_reported():
    md = "Nombre: ........................\nDNI: 12345678A\nEn fecha ........, y firma.\n"
    kept = clean_pages([md], "compact")
    assert "___" in kept.text and kept.empty_fields_omitted == 0

    opts = ConvertOptions.for_mode("compact", omit_empty_fields=True)
    res = clean_pages([md], "compact", options=opts)
    assert "___" not in res.text
    assert "DNI: 12345678A" in res.text
    assert "En fecha, y firma." in res.text
    assert res.empty_fields_omitted == 2
    assert any(r["kind"] == "empty_field" for r in res.removed)


# ---------------------------------------------------------------------------
# 1) Líneas largas de puntos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_removes_dot_leaders_inside_and_between_lines(mode):
    md = (
        "Nombre: ........................................ DNI: 12345678A\n\n"
        "................................................\n\n"
        "________________________________\n\n"
        "Firma: ____________________\n"
    )
    out = clean_markdown(md, mode)
    assert not LONG_DOTS.search(out)
    assert "____" not in out
    assert "Nombre: ___ DNI: 12345678A" in out
    assert "Firma: ___" in out
    assert not re.search(r"^\s*[._]+\s*$", out, re.M)  # ni líneas de puntos ni un `___` suelto (sería una regla horizontal)
    assert out.count("[campo en blanco]") == 2  # las dos líneas enteras de relleno se conservan como campo en blanco


def test_normalizes_spaced_and_escaped_fillers():
    out = clean_markdown(". . . . . . . . . .\n\nTexto \\_\\_\\_\\_\\_\\_\\_\\_ fin\n", "clean")
    assert out == "[campo en blanco]\n\nTexto ___ fin\n"


def test_filler_line_in_the_middle_of_a_sentence_becomes_an_inline_blank():
    md = "El trabajador prestará servicios como\n......................\nen el centro de Madrid."
    out = clean_markdown(md, "clean")
    assert out == "El trabajador prestará servicios como ___ en el centro de Madrid.\n"


def test_decorative_rules_are_still_removed():
    assert clean_markdown("Título\n\n----------------------------\n\nTexto\n", "clean") == "Título\n\nTexto\n"


def test_table_of_contents_keeps_title_and_page():
    out = clean_markdown("Capítulo uno de la obra ................................ 12\n", "clean")
    assert "Capítulo uno de la obra" in out
    assert out.strip().endswith("12")
    assert not LONG_DOTS.search(out)


# ---------------------------------------------------------------------------
# 2) Fechas, importes, números de cláusula
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_preserves_dates_amounts_and_clause_numbers(mode):
    md = (
        "## CLÁUSULA 3.2.1\n\n"
        "El importe será de 1.500,00 € (mil quinientos euros) pagadero el 01/02/2026 según el "
        "art. 12.3 del Real Decreto 1/2020, de 7 de enero. Ref. ES91 2100 0418 4502 0005 1332.\n\n"
        "1. Primera obligación.\n2. Segunda obligación.\n"
    )
    out = clean_markdown(md, mode)
    for token in (
        "CLÁUSULA 3.2.1",
        "1.500,00 €",
        "01/02/2026",
        "art. 12.3",
        "Real Decreto 1/2020",
        "ES91 2100 0418 4502 0005 1332",
        "1. Primera obligación.",
        "2. Segunda obligación.",
    ):
        assert token in out


# ---------------------------------------------------------------------------
# 3) Puntos suspensivos legítimos
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_keeps_legitimate_ellipsis(mode):
    md = "Y entonces... no sé… quizá.... bueno, etc... ¿Seguro...?\n"
    assert clean_markdown(md, mode) == md


# ---------------------------------------------------------------------------
# 4) Espacios sin concatenar palabras
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("mode", MODES)
def test_normalizes_spaces_without_gluing_words(mode):
    out = clean_markdown("El   contrato\t\tde   trabajo\u00a0\u00a0es   indefinido   \n", mode)
    assert out == "El contrato de trabajo es indefinido\n"


def test_normalize_chars_removes_control_and_invisible_characters():
    out = normalize_chars("a\x00b\x07c\u200bd\ufeffe\u00a0f")
    assert out == "abcde f"


def test_keeps_indentation_of_nested_lists():
    md = "- uno\n  - dos   con   espacios\n"
    assert clean_markdown(md, "clean") == "- uno\n  - dos con espacios\n"


# ---------------------------------------------------------------------------
# 5) Reconstrucción de párrafos
# ---------------------------------------------------------------------------
WRAPPED = (
    "Este contrato se celebra entre las partes abajo firmantes y se\n"
    "regirá por las normas aplicables a la relación laboral que\n"
    "las une desde la fecha de inicio indicada.\n"
)


def test_rebuilds_wrapped_paragraph_in_clean_mode():
    assert clean_markdown(WRAPPED, "clean") == (
        "Este contrato se celebra entre las partes abajo firmantes y se regirá por las normas "
        "aplicables a la relación laboral que las une desde la fecha de inicio indicada.\n"
    )


def test_faithful_mode_does_not_join_lines():
    assert clean_markdown(WRAPPED, "faithful") == WRAPPED


def test_joins_hyphenated_words_only_with_evidence():
    assert clean_markdown("la contra-\ntación pública\n", "clean") == "la contratación pública\n"
    # Fragmento con mayúscula: es un compuesto, se conserva el guion.
    assert clean_markdown("el Real Decreto-\nley 1/2020\n", "clean") == "el Real Decreto-ley 1/2020\n"
    # El guion pegado a una palabra y seguido de un número/mayúscula es de la palabra: se conserva.
    assert clean_markdown("tras el COVID-\n19 en España\n", "clean") == "tras el COVID-19 en España\n"
    # Un guion con espacio delante es un separador, no una palabra cortada.
    assert clean_markdown("Madrid -\nBarcelona\n", "clean") == "Madrid -\nBarcelona\n"


def test_does_not_join_independent_lines():
    md = "Nombre: Juan Pérez\nDNI: 12345678A\nFirmado.\nMadrid, 1 de enero\n"
    assert clean_markdown(md, "clean") == md


def test_joins_paragraph_split_across_pages_only_when_lowercase_continuation():
    out = clean_pages(["Las partes acuerdan que el contrato\n", "continuará vigente hasta nueva orden.\n"], "clean").text
    assert out == "Las partes acuerdan que el contrato continuará vigente hasta nueva orden.\n"

    out = clean_pages(["Fin del apartado primero\n", "Segundo apartado del contrato.\n"], "clean").text
    assert out == "Fin del apartado primero\n\nSegundo apartado del contrato.\n"


def test_blocks_in_different_positions_are_not_concatenated():
    # Dos bloques separados (p. ej. dos columnas) no se funden aunque el segundo empiece en minúscula.
    md = "Columna izquierda sin punto final\n\ncolumna derecha en minúscula\n"
    assert clean_markdown(md, "clean") == md


def test_list_continuation_lines_are_merged_into_their_item():
    md = "- primer elemento que sigue\n  en otra línea\n- segundo\n"
    assert clean_markdown(md, "clean") == "- primer elemento que sigue en otra línea\n- segundo\n"


# ---------------------------------------------------------------------------
# 6) Tablas y bloques de código
# ---------------------------------------------------------------------------
TABLE = "| Concepto | Importe |\n|---|---|\n| Salario base | 1.200,00 € |\n| Plus | 150,00 € |\n"


@pytest.mark.parametrize("mode", MODES)
def test_preserves_markdown_tables(mode):
    out = clean_markdown("Tabla salarial:\n\n" + TABLE + "\nFin.\n", mode)
    assert TABLE in out


def test_table_rows_are_never_merged_with_neighbours():
    out = clean_markdown("texto previo\n| a | b |\n|---|---|\n| 1 | 2 |\n", "clean")
    assert out == "texto previo\n| a | b |\n|---|---|\n| 1 | 2 |\n"


def test_compact_mode_normalizes_table_padding_but_keeps_cells_and_rows():
    md = "| Concepto      | Importe   |\n| ------------- | --------: |\n| Salario  base | 1.200,00 € |\n|  |  |\n"
    out = clean_markdown(md, "compact")
    # Una fila vacía puede ser una fila en blanco de un formulario: solo se omite si se pide expresamente.
    assert out == "| Concepto | Importe |\n|---|---:|\n| Salario base | 1.200,00 € |\n|  |  |\n"
    opts = ConvertOptions.for_mode("compact", omit_empty_fields=True)
    out = clean_markdown(md, "compact", options=opts)
    assert out == "| Concepto | Importe |\n|---|---:|\n| Salario base | 1.200,00 € |\n"


def test_code_blocks_are_untouched():
    md = "```\na    b ........... c\n\n\n  indented\n```\n"
    assert clean_markdown(md, "compact") == md


# ---------------------------------------------------------------------------
# 7) Cabeceras y pies
# ---------------------------------------------------------------------------
def test_remove_margin_lines_only_removes_edge_lines_that_match_literally():
    page = "Contrato ACME S.L.\n\nTexto del cuerpo.\n\nContrato ACME S.L. aparece también en el cuerpo\n\nPágina 3 de 10\n"
    out, n = remove_margin_lines(page, {"Contrato ACME S.L.", "Página 3 de 10"})
    assert n == 2
    assert "Texto del cuerpo." in out
    assert "aparece también en el cuerpo" in out
    assert "Página 3 de 10" not in out


def test_margin_lines_are_ignored_in_faithful_mode_and_removed_in_clean_mode():
    pages = ["Cabecera X\n\nCuerpo uno.\n\n1\n"]
    literals = [{"Cabecera X", "1"}]
    assert "Cabecera X" in clean_pages(pages, "faithful", literals).text
    res = clean_pages(pages, "clean", literals)
    assert res.text == "Cuerpo uno.\n"
    assert res.margin_lines_removed == 2


# ---------------------------------------------------------------------------
# 10) Documento ya limpio: no se modifica
# ---------------------------------------------------------------------------
CLEAN_DOC = (
    "# Contrato de prestación de servicios\n\n"
    "## Cláusula primera\n\n"
    "Las partes acuerdan lo siguiente... y firman en Madrid, a 20/04/2026.\n\n"
    "- Primer punto\n"
    "- Segundo punto\n\n"
    "1. Uno\n"
    "2. Dos\n\n"
    "| Concepto | Importe |\n"
    "|---|---|\n"
    "| Base | 100,00 € |\n\n"
    "Fin del documento.\n"
)


@pytest.mark.parametrize("mode", MODES)
def test_clean_document_is_not_modified(mode):
    assert clean_markdown(CLEAN_DOC, mode) == CLEAN_DOC


@pytest.mark.parametrize("mode", MODES)
def test_cleaning_is_idempotent(mode):
    once = clean_markdown(CONTRACT_INPUT + "\n\n" + WRAPPED + "\n" + TABLE, mode)
    assert clean_markdown(once, mode) == once


def test_empty_input():
    for mode in MODES:
        assert clean_markdown("", mode) == ""
        assert clean_markdown("   \n\n  \n", mode) == ""


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        clean_markdown("x", "agresivo")


# ---------------------------------------------------------------------------
# Compacto: sin pérdida de contenido
# ---------------------------------------------------------------------------
def test_compact_reduces_formatting_without_dropping_words():
    md = "# **Título**\n\n**Cláusula** **primera**\n\n- a\n\n- b\n"
    out = clean_markdown(md, "compact")
    assert out == "# Título\n\n**Cláusula primera**\n\n- a\n- b\n"
    assert estimate_tokens(out) <= estimate_tokens(md)


@pytest.mark.parametrize("mode", MODES)
def test_adjacent_bold_segments_are_never_glued_together(mode):
    out = clean_markdown("**Cláusula** **primera** y **segunda**\n", mode)
    words = re.sub(r"\*", "", out).split()
    assert words == ["Cláusula", "primera", "y", "segunda"]


def test_emphasis_emptied_by_filler_removal_is_dropped_without_touching_neighbours():
    out = clean_markdown("Nombre: **.................** Apellidos: **Pérez**\n", "clean")
    assert out == "Nombre: ___ Apellidos: **Pérez**\n"


def test_clean_mode_never_loses_words():
    md = WRAPPED + "\n" + "Cláusula 1:   obligaciones ........ del trabajador.\n"
    words_in = ["___" if w.startswith("....") else w for w in md.split()]
    words_out = clean_markdown(md, "clean").split()
    assert words_out == words_in


def test_word_and_token_counters():
    assert count_words("uno dos  tres\ncuatro") == 4
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcde") == 2


# ---------------------------------------------------------------------------
# Formularios, listas, duplicados y opciones explícitas (auditoría)
# ---------------------------------------------------------------------------
FORM = "Nombre: Juan Pérez\nDNI: 12345678Z\nTeléfono (opcional): ___\nEmail:\nDirección: Calle Mayor 5, 3º B\n"


@pytest.mark.parametrize("mode", MODES)
def test_form_entries_are_never_merged_into_a_paragraph(mode):
    assert clean_markdown(FORM, mode) == FORM


def test_long_form_entries_are_not_merged_even_if_they_look_like_full_lines():
    md = (
        "Domicilio: Calle Mayor número cinco, tercero B, 28001 Madrid\n"
        "Localidad: San Sebastián de los Reyes, Comunidad de Madrid\n"
        "Provincia: Madrid, código postal veintiocho mil uno\n"
    )
    assert clean_markdown(md, "clean") == md


def test_checkbox_lines_are_independent_entries():
    md = "Acepto las condiciones del servicio\n[x] Sí\n[ ] No\n[?] Estado desconocido\n"
    assert clean_markdown(md, "clean") == md


def test_geometric_evidence_joins_full_lines_but_not_form_entries():
    full = "\u200b"
    md = f"Este contrato se celebra entre las partes y se{full}\nRegirá por las normas aplicables.\n"
    assert clean_markdown(md, "clean") == "Este contrato se celebra entre las partes y se Regirá por las normas aplicables.\n"
    # Sin la evidencia geométrica, una mayúscula tras una línea corta no se une.
    assert clean_markdown(md.replace(full, ""), "clean").count("\n") == 2
    # Una entrada de formulario nunca se une aunque la línea anterior estuviera llena.
    md = f"Observaciones del trabajador{full}\nDNI: 12345678Z\n"
    assert clean_markdown(md, "clean") == "Observaciones del trabajador\nDNI: 12345678Z\n"
    # El perfil Fiel conserva las líneas y retira siempre la marca.
    md = f"Este contrato se celebra entre las partes y se{full}\nRegirá por las normas.\n"
    assert clean_markdown(md, "faithful") == "Este contrato se celebra entre las partes y se\nRegirá por las normas.\n"


def test_soft_hyphen_at_line_end_is_joined_in_every_mode():
    for mode in MODES:
        assert clean_markdown("la contra\u00ad\ntación pública\n", mode) == "la contratación pública\n"


def test_exact_duplicate_consecutive_paragraphs_are_removed_and_reported():
    p = "El arrendatario no podrá subarrendar el inmueble sin consentimiento escrito."
    res = clean_pages([f"{p}\n\n{p}\n\nOtro párrafo distinto."], "clean")
    assert res.text == f"{p}\n\nOtro párrafo distinto.\n"
    assert res.duplicates_removed == 1
    assert res.removed[0]["kind"] == "duplicate_block"
    # El perfil Fiel no elimina nada.
    assert clean_pages([f"{p}\n\n{p}"], "faithful").text.count(p) == 2


def test_duplicates_with_possible_meaning_are_kept():
    p = "El arrendatario no podrá subarrendar el inmueble sin consentimiento escrito."
    # No consecutivos: puede ser una repetición con significado (cláusula repetida en otro apartado).
    out = clean_markdown(f"{p}\n\n## Otro apartado\n\n{p}\n", "clean")
    assert out.count(p) == 2
    # Textos cortos ("Sí", "No") nunca se tratan como duplicados.
    assert clean_markdown("No\n\nNo\n", "clean") == "No\n\nNo\n"
    # Bloques de firma: nunca se eliminan por parecer duplicados.
    sig = "Firmado digitalmente por GARCÍA LÓPEZ MARÍA con fecha 2026.04.20 10:33:12"
    assert clean_markdown(f"{sig}\n\n{sig}\n", "clean").count("Firmado digitalmente") == 2


def test_repeated_blocks_are_removed_only_when_explicitly_requested():
    p = "Esta cláusula se repite literalmente en varios apartados del documento contractual."
    md = f"{p}\n\n## B\n\n{p}\n"
    assert clean_markdown(md, "compact").count(p) == 2
    opts = ConvertOptions.for_mode("compact", remove_repeated_blocks=True)
    res = clean_pages([md], "compact", options=opts)
    assert res.text.count(p) == 1 and res.repeated_blocks_removed == 1


def test_redundant_headings_only_when_requested():
    md = "## Anexo I\n\n## Anexo I\n\nTexto.\n"
    assert clean_markdown(md, "compact").count("## Anexo I") == 2
    opts = ConvertOptions.for_mode("compact", omit_redundant_headings=True)
    assert clean_markdown(md, "compact", options=opts) == "## Anexo I\n\nTexto.\n"


def test_margin_removal_is_reported_per_page():
    res = clean_pages(["Cabecera X\n\nCuerpo.\n\nPágina 1 de 2\n"], "clean", [{"Cabecera X", "Página 1 de 2"}])
    assert [r["text"] for r in res.removed] == ["Cabecera X", "Página 1 de 2"]
    assert all(r["page"] == 1 and r["kind"] == "margin" for r in res.removed)


def test_profile_options_can_be_overridden():
    wrapped_in_two_pages = ["Las partes acuerdan que el contrato\n", "continuará vigente.\n"]
    assert "contrato continuará" in clean_pages(wrapped_in_two_pages, "clean").text
    opts = ConvertOptions.for_mode("clean", join_paragraphs=False)
    assert "contrato continuará" not in clean_pages(wrapped_in_two_pages, "clean", options=opts).text


def test_conflicting_mode_and_options_are_rejected():
    with pytest.raises(ValueError):
        clean_pages(["x"], "clean", options=ConvertOptions.for_mode("compact"))


@pytest.mark.parametrize("mode", MODES)
def test_cleaning_never_changes_names_figures_dates_or_negations(mode):
    md = (
        "El trabajador NO tendrá derecho a indemnización salvo lo dispuesto en el art. 49.1.c) del Estatuto.\n"
        "Importe: 1.234,56 € (IVA no incluido). Fecha: 31/12/2026. Ref: ES91 2100 0418 4502 0005 1332.\n"
        "Porcentaje: 21,5 %. Sin perjuicio de lo anterior, no se admitirán reclamaciones.\n"
    )
    out = clean_markdown(md, mode)
    for token in ("NO tendrá", "art. 49.1.c)", "1.234,56 €", "IVA no incluido", "31/12/2026", "ES91 2100 0418 4502 0005 1332",
                  "21,5 %", "no se admitirán"):
        assert token in out
    assert out.split() == md.split()