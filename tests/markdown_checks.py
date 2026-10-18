# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Comprobaciones reutilizables de que un Markdown está bien formado."""

from __future__ import annotations

import re

_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b]")


def table_groups(md: str) -> list[list[str]]:
    """Grupos de líneas consecutivas que empiezan por `|` (fuera de bloques de código)."""
    groups: list[list[str]] = []
    cur: list[str] = []
    in_fence = False
    for line in md.split("\n"):
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            if cur:
                groups.append(cur)
                cur = []
            continue
        if not in_fence and line.lstrip().startswith("|"):
            cur.append(line)
        elif cur:
            groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return groups


def columns(line: str) -> int:
    s = line.strip()
    s = s[1:] if s.startswith("|") else s
    s = s[:-1] if s.endswith("|") and not s.endswith("\\|") else s
    return len(_CELL_SPLIT_RE.split(s))


def assert_valid_markdown(md: str) -> None:
    assert not _CONTROL_RE.search(md), "caracteres de control o de marca interna en el Markdown"
    if md:
        assert md.endswith("\n") and not md.endswith("\n\n")
    fences = [ln for ln in md.split("\n") if ln.lstrip().startswith(("```", "~~~"))]
    assert len(fences) % 2 == 0, "bloque de código sin cerrar"
    for group in table_groups(md):
        counts = {columns(r) for r in group}
        assert len(counts) == 1, f"fila de tabla mal formada: {group}"
        if len(group) >= 2:
            assert re.fullmatch(r"\|\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?", group[1].strip()), f"falta la fila separadora: {group}"
