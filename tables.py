# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Tablas: lectura de las tablas Markdown del motor y serialización según la estructura real y la
configuración del usuario.

Formatos (`table_format`):
  * markdown : tabla Markdown convencional (conserva filas, columnas y celdas vacías).
  * linear   : representación lineal con etiqueta y valor explícitos por celda
               ("- Fila 1: Concepto: Salario base | Importe: 1200 | Notas: (vacío)").
  * literal  : las filas tal como se extrajeron, en un bloque de texto sin interpretar.
  * auto     : markdown si la estructura es fiable y manejable; lineal si la tabla es demasiado
               ancha o irregular. La decisión depende de la estructura, no del tamaño del resultado.

Honestidad: nunca se afirma que una tabla es fiel al original. Cuando hay un indicio concreto de que la
estructura es aproximada (filas con distinto número de columnas, tabla no interpretable) se antepone una
nota con el motivo. La ausencia de nota significa "sin indicios", no "verificada".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

EMPTY_CELL = "(vacío)"
WIDE_TABLE_COLUMNS = 10  # a partir de aquí `auto` prefiere la forma lineal
_SEP_CELL_RE = re.compile(r"^:?-{1,}:?$")
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")


@dataclass
class ParsedTable:
    rows: list[list[str]]
    cols: int
    notes: list[str] = field(default_factory=list)  # motivos de estructura aproximada

    @property
    def approximate(self) -> bool:
        return bool(self.notes)


@dataclass
class RenderedTable:
    text: str
    rows: int
    cols: int
    format: str
    approximate: bool
    notes: list[str]


def _split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [re.sub(r"[ \t]+", " ", c).strip() for c in _CELL_SPLIT_RE.split(s)]


def parse_markdown_table(md: str) -> Optional[ParsedTable]:
    """Lee una tabla Markdown. Devuelve None si el texto no es una tabla de filas `|...|`."""
    lines = [ln for ln in md.strip().split("\n") if ln.strip()]
    if not lines or not all(ln.lstrip().startswith("|") for ln in lines):
        return None
    rows: list[list[str]] = []
    for idx, ln in enumerate(lines):
        cells = _split_row(ln)
        if idx == 1 and cells and all(_SEP_CELL_RE.match(c) for c in cells):
            continue  # fila separadora de la cabecera
        rows.append(cells)
    if not rows:
        return None
    cols = max(len(r) for r in rows)
    notes: list[str] = []
    if any(len(r) != cols for r in rows):
        notes.append("filas con distinto número de columnas (se han completado con celdas vacías)")
        rows = [r + [""] * (cols - len(r)) for r in rows]
    # Nota: las celdas combinadas no se pueden detectar con fiabilidad desde el Markdown del motor (una fila
    # con una sola celda rellena es tan compatible con una celda combinada como con celdas vacías), así que
    # no se afirma nada al respecto: la fidelidad de la estructura de las tablas no está garantizada.
    return ParsedTable(rows, cols, notes)


def _flat_cell(cell: str) -> str:
    return re.sub(r"\s+", " ", _BR_RE.sub(" / ", cell)).strip()


def _is_blank(cell: str) -> bool:
    """Celda vacía, o solo con la marca de campo en blanco."""
    return cell.strip() in ("", "___", "[campo en blanco]")


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _render_markdown(t: ParsedTable, omit_empty: bool) -> str:
    rows = t.rows
    if omit_empty:
        rows = [rows[0]] + [r for r in rows[1:] if not all(_is_blank(x) for x in r)]
    out = [_md_row(rows[0]), "|" + "|".join(["---"] * t.cols) + "|"]
    out += [_md_row(r) for r in rows[1:]]
    return "\n".join(out)


def _render_linear(t: ParsedTable, omit_empty: bool) -> str:
    header = [_flat_cell(c) or f"Columna {j + 1}" for j, c in enumerate(t.rows[0])]
    out = [f"[Tabla: cabecera + {len(t.rows) - 1} filas × {t.cols} columnas — formato lineal; cabeceras: {' | '.join(header)}]"]
    for n, row in enumerate(t.rows[1:], start=1):
        parts = []
        for label, cell in zip(header, row):
            value = _flat_cell(cell)
            if _is_blank(value):
                if omit_empty:
                    continue
                value = EMPTY_CELL
            parts.append(f"{label}: {value}")
        if parts:
            out.append(f"- Fila {n}: " + " | ".join(parts))
    return "\n".join(out)


def _render_literal(t: ParsedTable, omit_empty: bool) -> str:
    rows = t.rows
    if omit_empty:
        rows = [rows[0]] + [r for r in rows[1:] if not all(_is_blank(x) for x in r)]
    body = "\n".join(" | ".join(_flat_cell(c) for c in r).rstrip() for r in rows)
    return f"[Tabla: {len(rows)} filas (la primera, cabecera) × {t.cols} columnas — texto literal]\n```text\n{body}\n```"


def render_table(md: str, table_format: str = "auto", omit_empty: bool = False) -> RenderedTable:
    """Serializa una tabla Markdown del motor según `table_format`."""
    parsed = parse_markdown_table(md)
    if parsed is None:
        # No se puede interpretar: se conserva el texto tal cual, avisando de que es aproximado.
        raw = md.strip()
        note = "> Nota: tabla no interpretable; se conserva el texto extraído sin estructura fiable."
        return RenderedTable(f"{note}\n\n```text\n{raw}\n```" if raw else "", 0, 0, "literal", True, ["no interpretable"])

    fmt = table_format
    if fmt == "auto":
        fmt = "linear" if (parsed.cols > WIDE_TABLE_COLUMNS or (parsed.approximate and parsed.cols > 4)) else "markdown"
    if fmt == "linear":
        text = _render_linear(parsed, omit_empty)
    elif fmt == "literal":
        text = _render_literal(parsed, omit_empty)
    else:
        fmt = "markdown"
        text = _render_markdown(parsed, omit_empty)
    if parsed.approximate:
        text = f"> Nota: estructura de tabla aproximada ({'; '.join(parsed.notes)}).\n\n{text}"
    return RenderedTable(text, len(parsed.rows), parsed.cols, fmt, parsed.approximate, parsed.notes)
