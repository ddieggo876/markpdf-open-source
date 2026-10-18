# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Detección de cabeceras, pies de página y numeración por patrones repetidos entre páginas.

Principios (la prioridad es NO borrar contenido):
  * Solo se consideran bloques de texto completamente dentro de la franja superior/inferior.
  * Un texto se considera cabecera/pie únicamente si se repite en >= 40 % de las páginas (mín. 3),
    en una posición vertical similar, y no aparece también como bloque del cuerpo del documento.
  * Los dígitos que varían entre páginas deben seguir la numeración de página (p. ej. "Página 3 de
    10"); si varían de otra forma (capítulos, importes, referencias...) el texto se conserva.
  * Un número suelto solo se trata como numeración si la secuencia de números coincide con el
    índice de página (con un desfase constante) o si lleva una palabra clave ("Página 3", "3 de 10").
  * Ante la duda, se conserva. Lo que se detecta se informa en `MarginAnalysis.items`.

Esta detección no modifica el PDF: devuelve, por página, el texto literal de los bloques de margen
que `cleaning.remove_margin_lines()` elimina del Markdown solo si coincide con la propia página.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Optional

MARGIN_RATIO = 0.10  # franja superior/inferior considerada "margen"
REPEAT_RATIO = 0.40  # un texto debe repetirse en >= 40 % de las páginas (mín. 3)
MIN_REPEATS = 3
POSITION_TOLERANCE = 0.03  # variación vertical admitida (fracción de la altura de página)
NUMBER_SEQUENCE_MATCH = 0.8  # fracción de candidatos que debe seguir la secuencia de páginas

_KEYWORD_RE = re.compile(r"^\W*(?:p[áa]g(?:ina|\.)?|page|p\.)\s*\d{1,4}", re.IGNORECASE)
_OF_RE = re.compile(r"^\W*\d{1,4}\s*(?:/|de|of)\s*\d{1,4}\W*$", re.IGNORECASE)
_BARE_RE = re.compile(r"^\W*(\d{1,4})\W*$")
PAGE_NUM_RE = re.compile(
    r"^\W*(?:p[áa]g(?:ina|\.)?|page|p\.)?\s*\d{1,4}(?:\s*(?:/|de|of)\s*\d{1,4})?\W*$",
    re.IGNORECASE,
)


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def norm_margin_text(text: str) -> str:
    text = re.sub(r"\d+", "#", text.lower())
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class MarginItem:
    kind: str  # header | footer | page_number
    text: str  # texto de la primera aparición
    pages: list[int]  # páginas (1-based) donde se detectó


@dataclass
class MarginAnalysis:
    literals: list[set[str]]
    items: list[MarginItem] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)  # candidatos conservados por baja confianza

    def as_report(self) -> list[dict[str, Any]]:
        return [{"kind": i.kind, "text": i.text, "pages": i.pages} for i in self.items]


@dataclass
class _Block:
    page: int
    text: str  # texto aplanado
    lines: list[str]
    y0: float
    y1: float
    band: str  # top | bottom


def _collect(doc: Any) -> tuple[list[list[_Block]], set[str]]:
    per_page: list[list[_Block]] = []
    body_keys: set[str] = set()
    for idx, page in enumerate(doc):
        height = page.rect.height or 1.0
        top_limit, bottom_limit = height * MARGIN_RATIO, height * (1 - MARGIN_RATIO)
        blocks: list[_Block] = []
        for block in page.get_text("blocks"):
            _x0, y0, _x1, y1, text, _no, btype = block[:7]
            if btype != 0:
                continue
            flat = _flat(text)
            if not flat:
                continue
            if y1 <= top_limit or y0 >= bottom_limit:
                lines = [_flat(ln) for ln in text.split("\n") if ln.strip()]
                blocks.append(_Block(idx, flat, lines, y0, y1, "top" if y1 <= top_limit else "bottom"))
            else:
                body_keys.add(norm_margin_text(flat))
        per_page.append(blocks)
    return per_page, body_keys


def _digit_slots_ok(occurrences: list[_Block]) -> bool:
    """Los números del texto deben ser constantes o seguir el índice de página."""
    seqs = [re.findall(r"\d+", b.text) for b in occurrences]
    if len({len(s) for s in seqs}) != 1:
        return False
    for slot in range(len(seqs[0])):
        values = [int(s[slot]) for s, b in zip(seqs, occurrences)]
        if len(set(values)) == 1:
            continue
        offsets = Counter(v - b.page for v, b in zip(values, occurrences))
        if offsets.most_common(1)[0][1] / len(values) < NUMBER_SEQUENCE_MATCH:
            return False
    return True


def analyze_margins(
    doc: Any,
    *,
    remove_headers_footers: bool = True,
    remove_page_numbers: bool = True,
    repeat_ratio: float = REPEAT_RATIO,
) -> MarginAnalysis:
    n_pages = len(doc)
    per_page, body_keys = _collect(doc)
    literals: list[set[str]] = [set() for _ in range(n_pages)]
    analysis = MarginAnalysis(literals)
    if not (remove_headers_footers or remove_page_numbers):
        return analysis

    def mark(block: _Block) -> None:
        literals[block.page].add(block.text)
        literals[block.page].update(block.lines)

    # --- 1) Numeración de página -------------------------------------------------------
    accepted: set[int] = set()  # id() de los bloques ya tratados como numeración
    if remove_page_numbers:
        bare: list[tuple[_Block, int]] = []
        number_pages: list[_Block] = []
        for blocks in per_page:
            for b in blocks:
                if not PAGE_NUM_RE.match(b.text):
                    continue
                if _KEYWORD_RE.match(b.text) or _OF_RE.match(b.text):
                    mark(b)
                    number_pages.append(b)
                    accepted.add(id(b))
                elif (m := _BARE_RE.match(b.text)) is not None:
                    bare.append((b, int(m.group(1))))
        if len(bare) >= 2:
            offsets = Counter(v - b.page for b, v in bare)
            offset, hits = offsets.most_common(1)[0]
            if hits / len(bare) >= NUMBER_SEQUENCE_MATCH:
                for b, v in bare:
                    if v - b.page == offset:
                        mark(b)
                        number_pages.append(b)
                        accepted.add(id(b))
            else:
                analysis.uncertain.append("Números sueltos en el margen que no siguen la numeración de páginas.")
        elif bare:
            analysis.uncertain.append("Número suelto en el margen sin evidencia suficiente de ser numeración.")
        if number_pages:
            first = min(number_pages, key=lambda b: b.page)
            analysis.items.append(
                MarginItem("page_number", first.text, sorted({b.page + 1 for b in number_pages}))
            )

    # --- 2) Cabeceras y pies repetidos --------------------------------------------------
    if remove_headers_footers:
        min_repeats = max(MIN_REPEATS, math.ceil(n_pages * repeat_ratio))
        groups: dict[tuple[str, str], list[_Block]] = defaultdict(list)
        for blocks in per_page:
            for b in blocks:
                if id(b) in accepted:
                    continue
                key = norm_margin_text(b.text)
                if re.fullmatch(r"[#\W]*", key):  # solo dígitos/puntuación: se trata arriba
                    continue
                groups[(b.band, key)].append(b)
        for (band, key), occ in groups.items():
            if len({b.page for b in occ}) < min_repeats:
                continue
            height = doc[occ[0].page].rect.height or 1.0
            med = median(b.y0 for b in occ)
            near = [b for b in occ if abs(b.y0 - med) <= POSITION_TOLERANCE * height]
            if len({b.page for b in near}) < min_repeats:
                analysis.uncertain.append(f"Texto repetido en posiciones distintas, conservado: «{occ[0].text[:60]}»")
                continue
            if key in body_keys:
                analysis.uncertain.append(f"Texto repetido que también aparece en el cuerpo, conservado: «{occ[0].text[:60]}»")
                continue
            if not _digit_slots_ok(near):
                analysis.uncertain.append(f"Cabecera/pie con valores variables, conservado: «{occ[0].text[:60]}»")
                continue
            if not remove_page_numbers and len({tuple(re.findall(r"\d+", b.text)) for b in near}) > 1:
                continue  # la parte variable es la numeración de página y el usuario pidió conservarla
            for b in near:
                mark(b)
            first = min(near, key=lambda b: b.page)
            analysis.items.append(
                MarginItem("header" if band == "top" else "footer", first.text, sorted({b.page + 1 for b in near}))
            )
    return analysis


def detect_margin_noise(doc: Any, **kwargs: Any) -> list[set[str]]:
    """Compatibilidad: solo los literales por página."""
    return analyze_margins(doc, **kwargs).literals
