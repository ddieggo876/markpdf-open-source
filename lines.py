# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Reconstrucción de texto a partir de las líneas reales del PDF (PyMuPDF), caja a caja.

Por qué existe (observado en la auditoría): el motor de layout de pymupdf4llm escribe cada caja de
texto "sin saltos de línea", es decir, une TODAS sus líneas con espacios. En un formulario o una lista
eso destruye la estructura:

    PDF:      Nombre: Juan Perez            Motor:  Nombre: Juan Perez DNI: 12345678Z Telefono
              DNI: 12345678Z                        (opcional): Email: Direccion: Calle Mayor 5
              Telefono (opcional):
              Email:
              Direccion: Calle Mayor 5

y además descarta el guion final de una línea ("COVID-" + "19" -> "COVID19", "Decreto-" + "ley" ->
"Decretoley"), alterando el texto.

Estrategia: se conserva la clasificación del motor (qué es tabla, título, imagen, texto...) y su
orden de lectura, pero el texto de las cajas "de texto" se vuelve a leer del PDF línea a línea:

  * Las líneas se agrupan en filas visuales (etiqueta y valor dibujados por separado en la misma
    línea quedan juntos: "Nombre completo: Juan Perez").
  * Cada fila es una línea de salida. Un hueco vertical grande abre un párrafo nuevo (línea en blanco).
  * La unión de líneas partidas por maquetación NO se hace aquí: la decide `cleaning.py` según el
    perfil (el perfil Fiel conserva las líneas tal cual). Aquí solo se aporta evidencia geométrica:
    una fila que llega hasta el borde derecho de su columna lleva al final `FULL_MARK` (U+200B), que
    `cleaning.py` consume y elimina siempre.
  * Las listas conservan sus marcadores y su nivel de anidamiento (por sangría), aunque el motor las
    haya repartido en varias cajas.
  * Los glifos de casilla / símbolos se interpretan o se marcan como desconocidos (forms.py).
  * El texto duplicado por superposición de capas (misma cadena en el mismo rectángulo) se lee una vez.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Iterable

try:  # PyMuPDF >= 1.24.3
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from forms import map_span_text

FULL_MARK = "\u200b"
_TEXT_FLAGS = fitz.TEXTFLAGS_DICT & ~fitz.TEXT_PRESERVE_LIGATURES & ~fitz.TEXT_PRESERVE_IMAGES
_LIST_MARKER_RE = re.compile(r"^(?:[-*+•–]|\d{1,3}[.)]|[A-Za-z][.)]|\(\d{1,3}\)|\([A-Za-z]\))\s+\S")
BOX_TOLERANCE = 1.5  # puntos de holgura al asignar líneas a las cajas del motor
FULL_COLUMN_RATIO = 0.35  # una columna debe medir >= 35 % del ancho de página para fiarse de "fila llena"
FULL_ROW_TOLERANCE = 0.04  # una fila está "llena" si termina a menos del 4 % del borde derecho de la columna
OVERLAY_IOU = 0.85


@dataclass
class LineInfo:
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    size: float = 10.0
    unknown_symbols: int = 0
    indent: float = 0.0  # sangría efectiva (x0 + espacios iniciales del propio texto)
    full: bool = False

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2

    @property
    def height(self) -> float:
        return max(self.y1 - self.y0, 1.0)


@dataclass
class RebuildResult:
    texts: dict[int, str] = field(default_factory=dict)  # índice de caja -> texto
    unknown_symbols: int = 0


def _is_bold(span: dict[str, Any]) -> bool:
    name = (span.get("font") or "").lower()
    return bool(span.get("flags", 0) & 16) or "bold" in name or "black" in name or name.endswith(("-bd", ",bd"))


def _line_text(spans: list[dict[str, Any]]) -> tuple[str, int]:
    """Texto de una línea con negritas `**...**` agrupadas y símbolos interpretados."""
    segments: list[list[Any]] = []  # [texto, negrita]
    unknown = 0
    for span in spans:
        text, unk = map_span_text(span.get("text", ""), span.get("font", ""))
        unknown += unk
        if not text:
            continue
        bold = _is_bold(span)
        if not text.strip() and segments:
            bold = segments[-1][1]  # los espacios heredan la negrita del segmento anterior
        if segments and segments[-1][1] == bold:
            segments[-1][0] += text
        else:
            segments.append([text, bold])
    out: list[str] = []
    for text, bold in segments:
        if bold and text.strip():
            lead = len(text) - len(text.lstrip())
            trail = len(text) - len(text.rstrip())
            out.append(text[:lead] + "**" + text.strip() + "**" + (text[len(text) - trail:] if trail else ""))
        else:
            out.append(text)
    return "".join(out), unknown


def _iou(a: LineInfo, b: LineInfo) -> float:
    iw = min(a.x1, b.x1) - max(a.x0, b.x0)
    ih = min(a.y1, b.y1) - max(a.y0, b.y0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a.x1 - a.x0) * (a.y1 - a.y0) + (b.x1 - b.x0) * (b.y1 - b.y0) - inter
    return inter / union if union > 0 else 0.0


def _drop_overlay_duplicates(lines: list[LineInfo]) -> list[LineInfo]:
    """Quita líneas idénticas dibujadas una sobre otra (capas superpuestas, falsas negritas)."""
    kept: list[LineInfo] = []
    for ln in lines:
        key = ln.text.strip()
        if any(k.text.strip() == key and _iou(k, ln) >= OVERLAY_IOU for k in kept[-12:]):
            continue
        kept.append(ln)
    return kept


def page_lines(page: Any) -> list[LineInfo]:
    """Todas las líneas de texto de la página (sin imágenes), con su rectángulo."""
    lines: list[LineInfo] = []
    for block in page.get_text("dict", flags=_TEXT_FLAGS).get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue
            text, unknown = _line_text(spans)
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            size = float(spans[0].get("size", 10.0))
            lead = len(text) - len(text.lstrip(" "))
            lines.append(LineInfo(text, x0, y0, x1, y1, size, unknown, indent=x0 + lead * 0.28 * size))
    return _drop_overlay_duplicates(lines)


def plain_page_text(page: Any) -> str:
    """Texto de la página línea a línea (sin negritas) y sin duplicados por superposición; referencia de cobertura."""
    rows = group_rows(page_lines(page))
    return "\n".join(r.text.replace("**", "") for r in rows)


def _inside(line: LineInfo, bbox: Iterable[float]) -> bool:
    x0, y0, x1, y1 = bbox
    t = BOX_TOLERANCE
    return x0 - t <= line.cx <= x1 + t and y0 - t <= line.cy <= y1 + t


def group_rows(lines: list[LineInfo]) -> list[LineInfo]:
    """Une las líneas que están en la misma fila visual (misma línea base) y ordena por posición."""
    rows: list[list[LineInfo]] = []
    for ln in sorted(lines, key=lambda l: (l.cy, l.x0)):
        if rows:
            ref = rows[-1][0]
            if abs(ln.cy - ref.cy) <= 0.5 * min(ln.height, ref.height):
                rows[-1].append(ln)
                continue
        rows.append([ln])
    merged: list[LineInfo] = []
    for row in rows:
        row.sort(key=lambda l: l.x0)
        text = row[0].text
        for prev, cur in zip(row, row[1:]):
            gap = cur.x0 - prev.x1
            glue = "" if gap < 0.08 * max(cur.size, 1.0) and not (text.endswith(" ") or cur.text.startswith(" ")) else " "
            text += glue + cur.text
        merged.append(
            LineInfo(
                text.strip(),
                min(l.x0 for l in row),
                min(l.y0 for l in row),
                max(l.x1 for l in row),
                max(l.y1 for l in row),
                row[0].size,
                sum(l.unknown_symbols for l in row),
                indent=row[0].indent,
            )
        )
    return merged


def _paragraph_breaks(rows: list[LineInfo]) -> set[int]:
    """Índices de fila antes de los cuales hay un hueco vertical propio de un párrafo nuevo."""
    if len(rows) < 2:
        return set()
    gaps = [rows[i].y0 - rows[i - 1].y1 for i in range(1, len(rows))]
    h_med = median(r.height for r in rows)
    threshold = 0.9 * h_med
    if len(gaps) >= 3:
        threshold = max(threshold, 1.8 * median(max(g, 0.0) for g in gaps))
    return {i for i, g in enumerate(gaps, start=1) if g > threshold}


def _list_levels(rows: list[LineInfo]) -> dict[int, int]:
    """Nivel de anidamiento de las filas con marcador de lista, según su sangría."""
    marker_rows = [i for i, r in enumerate(rows) if _LIST_MARKER_RE.match(r.text)]
    if not marker_rows:
        return {}
    xs: list[float] = []
    for x in sorted(rows[i].indent for i in marker_rows):
        if not xs or x - xs[-1] > 4.0:
            xs.append(x)
    return {i: next(k for k, x in enumerate(xs) if rows[i].indent <= x + 4.0) for i in marker_rows}


def rows_to_text(rows: list[LineInfo]) -> str:
    levels = _list_levels(rows)
    breaks = _paragraph_breaks(rows)
    out: list[str] = []
    for i, row in enumerate(rows):
        if i in breaks:
            out.append("")
        out.append("  " * levels.get(i, 0) + row.text + (FULL_MARK if row.full else ""))
    return "\n".join(out)


def _mark_full_rows(per_box: dict[int, list[LineInfo]], page_width: float) -> None:
    """Marca las filas que llegan al borde derecho de su columna (evidencia de ajuste automático)."""
    spans = {i: (min(r.x0 for r in rows), max(r.x1 for r in rows)) for i, rows in per_box.items()}
    for i, rows in per_box.items():
        x0, x1 = spans[i]
        right = x1
        for j, (ox0, ox1) in spans.items():
            if j != i and min(x1, ox1) - max(x0, ox0) >= 0.5 * min(x1 - x0, ox1 - ox0):
                right = max(right, ox1)
        width = right - x0
        if width < FULL_COLUMN_RATIO * page_width:
            continue
        for r in rows:
            r.full = r.x1 >= right - FULL_ROW_TOLERANCE * width


def _is_list_start(rows: list[LineInfo]) -> bool:
    return bool(rows) and bool(_LIST_MARKER_RE.match(rows[0].text))


def rebuild_boxes(
    page: Any,
    boxes: list[dict[str, Any]],
    rebuild: set[int],
    excluded: set[int],
) -> RebuildResult:
    """
    Reconstruye el texto de las cajas `rebuild` desde las líneas del PDF.
    Las líneas que caen dentro de una caja `excluded` (tablas, ya serializadas aparte) se descartan.
    """
    result = RebuildResult()
    lines = page_lines(page)
    per_box_lines: dict[int, list[LineInfo]] = {i: [] for i in rebuild}
    areas = [max((b["bbox"][2] - b["bbox"][0]) * (b["bbox"][3] - b["bbox"][1]), 1) for b in boxes]
    for ln in lines:
        containing = [i for i, b in enumerate(boxes) if _inside(ln, b["bbox"])]
        if not containing or any(i in excluded for i in containing):
            continue
        owner = min(containing, key=lambda i: areas[i])
        if owner in per_box_lines:
            per_box_lines[owner].append(ln)
            result.unknown_symbols += ln.unknown_symbols

    per_box = {i: group_rows(lns) for i, lns in per_box_lines.items() if lns}
    _mark_full_rows(per_box, float(page.rect.width))

    # Las listas que el motor reparte en varias cajas consecutivas se serializan como un único bloque,
    # para calcular juntos los niveles de sangría y no insertar líneas en blanco entre elementos.
    groups: list[list[int]] = []
    for idx in sorted(per_box):
        rows = per_box[idx]
        if groups and _is_list_start(rows) and _is_list_start(per_box[groups[-1][0]]):
            prev_rows = per_box[groups[-1][-1]]
            gap = rows[0].y0 - prev_rows[-1].y1
            if gap <= 0.9 * median(r.height for r in prev_rows + rows):
                groups[-1].append(idx)
                continue
        groups.append([idx])
    for group in groups:
        merged_rows = [r for i in group for r in per_box[i]]
        result.texts[group[0]] = rows_to_text(merged_rows)
        for extra in group[1:]:
            result.texts[extra] = ""
    return result


def orphan_lines(page: Any, boxes: list[dict[str, Any]]) -> list[LineInfo]:
    """Líneas de texto que no caen en ninguna caja del motor (se usan solo para diagnóstico)."""
    return [ln for ln in page_lines(page) if not any(_inside(ln, b["bbox"]) for b in boxes)]
