# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Validación: (1) el Markdown final está bien formado y (2) la extracción no ha perdido texto.

1) `validate_markdown()` repara solo problemas de forma seguros (bloque de código sin cerrar, filas de tabla
   con distinto número de columnas, líneas en blanco de más) y devuelve los problemas encontrados.
2) `extraction_coverage()` compara, por página, los caracteres alfanuméricos de la capa de texto del PDF con
   los del Markdown extraído (ANTES de la limpieza configurable). No es una prueba de fidelidad perfecta: no
   detecta texto cambiado de sitio ni errores dentro de una palabra. Detecta pérdidas, que es lo que más
   importa, y las comunica como "posible pérdida", nunca como certeza.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_SEP_CELL_RE = re.compile(r"^:?-+:?$")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _row_cells(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    return [c.strip() for c in _CELL_SPLIT_RE.split(s)]


def _fix_table(rows: list[str], issues: list[str]) -> list[str]:
    cells = [_row_cells(r) for r in rows]
    is_sep = [bool(c) and all(_SEP_CELL_RE.match(x) for x in c) for c in cells]
    cols = max(len(c) for c in cells)
    if all(len(c) == cols for c in cells) and any(is_sep):
        return rows
    issues.append("Se corrigió una tabla con filas de distinto número de columnas.")
    out = []
    for c, sep in zip(cells, is_sep):
        if sep:
            out.append("|" + "|".join(["---"] * cols) + "|")
        else:
            out.append("| " + " | ".join(c + [""] * (cols - len(c))) + " |")
    if not any(is_sep) and len(out) >= 1:
        out.insert(1, "|" + "|".join(["---"] * cols) + "|")
    return out


def validate_markdown(md: str) -> tuple[str, list[str]]:
    """Devuelve (markdown_validado, problemas_encontrados)."""
    issues: list[str] = []
    if _CONTROL_RE.search(md):
        md = _CONTROL_RE.sub("", md)
        issues.append("Se eliminaron caracteres de control.")
    out: list[str] = []
    in_fence = False
    fence_marker = ""
    table: list[str] = []

    def flush_table() -> None:
        nonlocal table
        if table:
            out.extend(_fix_table(table, issues))
            table = []

    for line in md.split("\n"):
        m = _FENCE_RE.match(line)
        if in_fence:
            out.append(line)
            if m and m.group(1) == fence_marker:
                in_fence = False
            continue
        if m:
            flush_table()
            in_fence, fence_marker = True, m.group(1)
            out.append(line)
            continue
        if line.lstrip().startswith("|"):
            table.append(line)
            continue
        flush_table()
        out.append(line.rstrip())
    flush_table()
    if in_fence:
        out.append(fence_marker)
        issues.append("Se cerró un bloque de código que quedó abierto.")
    text = "\n".join(out)
    text = re.sub(r"\n{3,}", "\n\n", text) if "```" not in text and "~~~" not in text else text
    text = text.strip("\n")
    return (text + "\n" if text else ""), issues


# ---------------------------------------------------------------------------
# Cobertura de la extracción
# ---------------------------------------------------------------------------
def _alnum(text: str) -> str:
    return "".join(ch for ch in unicodedata.normalize("NFKC", text).lower() if ch.isalnum())


@dataclass
class Coverage:
    checked_pages: int = 0
    source_chars: int = 0
    missing_chars: int = 0
    pages: list[dict[str, int]] = field(default_factory=list)  # páginas con posible pérdida
    examples: list[str] = field(default_factory=list)

    @property
    def ratio(self) -> float:
        return 1.0 if not self.source_chars else 1 - self.missing_chars / self.source_chars

    def as_dict(self) -> dict[str, Any]:
        return {
            "checked_pages": self.checked_pages,
            "source_chars": self.source_chars,
            "missing_chars": self.missing_chars,
            "coverage": round(self.ratio, 4),
            "pages_with_loss": self.pages,
            "examples": self.examples,
        }


PAGE_LOSS_MIN_CHARS = 8
PAGE_LOSS_MIN_RATIO = 0.03
DOC_WARN_RATIO = 0.98


def extraction_coverage(source_pages: dict[int, str], md_pages: list[str]) -> Coverage:
    """`source_pages`: {índice_página: texto de la capa de texto}. `md_pages`: Markdown por página."""
    cov = Coverage()
    for idx, source in source_pages.items():
        if idx >= len(md_pages):
            continue
        src, md = _alnum(source), _alnum(md_pages[idx])
        if not src:
            continue
        cov.checked_pages += 1
        cov.source_chars += len(src)
        missing = sum((Counter(src) - Counter(md)).values())
        cov.missing_chars += missing
        if missing >= PAGE_LOSS_MIN_CHARS and missing / len(src) >= PAGE_LOSS_MIN_RATIO:
            cov.pages.append({"page": idx + 1, "missing_chars": missing})
            if len(cov.examples) < 5:
                for tok in dict.fromkeys(
                    _alnum(t) for t in re.findall(r"\w{4,}", unicodedata.normalize("NFKC", source))
                ):
                    if tok and tok not in md and len(cov.examples) < 5:
                        cov.examples.append(tok)
    return cov


def coverage_warning(cov: Coverage) -> str:
    """Aviso legible si la cobertura sugiere pérdidas; cadena vacía si todo parece correcto."""
    if cov.ratio >= DOC_WARN_RATIO and not cov.pages:
        return ""
    pages = ", ".join(str(p["page"]) for p in cov.pages[:10])
    where = f" (páginas {pages})" if pages else ""
    examples = f" Ejemplos de texto que no aparece: {', '.join(cov.examples)}." if cov.examples else ""
    return (
        f"Posible pérdida de texto en la extracción{where}: aproximadamente {round((1 - cov.ratio) * 100, 1)} % de los "
        f"caracteres alfanuméricos del PDF no aparecen en el Markdown.{examples} Revisa esas páginas en el original."
    )
