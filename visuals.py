# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Imágenes y elementos gráficos: detección y tratamiento según `image_mode`.

Tratamientos (independientes del perfil de limpieza de texto):
  * ignore  : no se incorpora nada de las imágenes. Con `image_markers=True` se deja un marcador breve.
  * markers : marcador de posición con la página. "[Imagen omitida: página 3]"
  * extract : se renderiza cada imagen a un PNG y se referencia desde el Markdown
              (`![Imagen pág 3](images/page003-01.png)`). Solo se referencian archivos que existen.
  * describe: requiere un modelo de visión (ver vision.py). Sin él, la API rechaza la opción.

Honestidad de los marcadores: solo se dice lo que se sabe. Una imagen raster es una "imagen"; un dibujo
detectado por trazos vectoriales es un "dibujo vectorial"; un elemento que solo detectó el motor de layout
es un "elemento visual". Nunca se afirma que algo es un gráfico, una firma, un logotipo o una tabla.
Una imagen idéntica repetida en muchas páginas se trata como un único elemento ("imagen repetida").
"""

from __future__ import annotations

import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # PyMuPDF >= 1.24.3
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

log = logging.getLogger("markpdf.visuals")

MIN_IMAGE_PAGE_RATIO = 0.06  # imágenes menores al 6 % de la página se consideran decorativas
REPEAT_RATIO = 0.40
MIN_REPEATS = 3
SCAN_IMAGE_RATIO = 0.30
MAX_EXTRACTED_IMAGES = 200
EXTRACT_DPI = 150

MARKER_RE = re.compile(
    r"^\[(?:Imagen|Imagen repetida|Dibujo vectorial|Elemento visual|Página sin texto)[^\]\n]*\]$"
)


@dataclass
class Visual:
    page: int  # 0-based
    rect: Any  # fitz.Rect
    source: str  # image | layout | vector
    repeated: bool = False
    repeat_pages: list[int] = field(default_factory=list)  # 1-based, solo si repeated

    @property
    def label(self) -> str:
        if self.repeated:
            return "Imagen repetida"
        return {"image": "Imagen", "vector": "Dibujo vectorial"}.get(self.source, "Elemento visual")


@dataclass
class VisualScan:
    per_page: dict[int, list[Visual]] = field(default_factory=dict)
    small_ignored: int = 0  # imágenes decorativas por su tamaño (no se tratan)
    repeated_occurrences: int = 0  # apariciones repetidas absorbidas por la primera de su grupo

    @property
    def relevant(self) -> int:
        return sum(1 for v in self.all() if not v.repeated)

    @property
    def repeated(self) -> int:
        """Imágenes repetidas (todas las apariciones, incluidas las absorbidas por la primera)."""
        return sum(1 for v in self.all() if v.repeated) + self.repeated_occurrences

    def all(self) -> list[Visual]:
        return [v for vs in self.per_page.values() for v in vs]


@dataclass
class VisualStats:
    detected: int = 0  # relevantes + repetidas + decorativas pequeñas
    relevant: int = 0
    repeated: int = 0
    small_ignored: int = 0
    extracted: int = 0
    marked: int = 0
    omitted: int = 0  # relevantes sin marcador ni archivo
    described: int = 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def _vector_chart_rect(page: Any) -> Optional[Any]:
    """Heurística: muchas curvas/rellenos vectoriales => dibujo (no una tabla de líneas)."""
    page_area = page.rect.get_area()
    curves = filled = 0
    union: Optional[Any] = None
    try:
        drawings = page.get_drawings()
    except Exception:  # pragma: no cover
        return None
    for d in drawings:
        rect = d.get("rect")
        if rect is None or rect.get_area() > 0.8 * page_area:
            continue  # fondos de página
        is_curve = any(item[0] in ("c", "qu") for item in d.get("items", []))
        is_fill = d.get("fill") is not None and "f" in (d.get("type") or "")
        if is_curve:
            curves += 1
        elif is_fill and rect.get_area() > 4:
            filled += 1
        else:
            continue
        union = fitz.Rect(rect) if union is None else union.include_rect(rect)
    if union is None:
        return None
    if (curves >= 10 or filled >= 10) and union.get_area() >= 0.05 * page_area:
        return union
    return None


def _overlap(a: Any, b: Any) -> float:
    """Fracción del rectángulo menor cubierta por la intersección."""
    inter = fitz.Rect(a) & fitz.Rect(b)
    if inter.is_empty:
        return 0.0
    smaller = min(fitz.Rect(a).get_area(), fitz.Rect(b).get_area()) or 1.0
    return inter.get_area() / smaller


def detect_visuals(
    doc: Any,
    *,
    layout_pictures: Optional[dict[int, list[Any]]] = None,
    skip_pages: Optional[set[int]] = None,
) -> VisualScan:
    """
    Detecta imágenes y dibujos relevantes. `layout_pictures` son los rectángulos de "picture" del motor
    de layout (opcional). `skip_pages`: páginas escaneadas (la imagen es la propia página).
    """
    skip_pages = skip_pages or set()
    layout_pictures = layout_pictures or {}
    n_pages = len(doc)
    xref_pages: Counter[int] = Counter()
    for page in doc:
        for xref in {img[0] for img in page.get_images(full=True)}:
            xref_pages[xref] += 1
    repeat_threshold = max(MIN_REPEATS, math.ceil(n_pages * REPEAT_RATIO))

    scan = VisualScan()
    repeated_groups: dict[int, list[Visual]] = defaultdict(list)
    for idx, page in enumerate(doc):
        if idx in skip_pages:
            continue
        page_area = page.rect.get_area() or 1.0
        found: list[Visual] = []
        for xref in {img[0] for img in page.get_images(full=True)}:
            try:
                rects = page.get_image_rects(xref)
            except Exception:  # pragma: no cover
                continue
            for rect in rects:
                if rect.get_area() / page_area < MIN_IMAGE_PAGE_RATIO:
                    scan.small_ignored += 1
                    continue
                vis = Visual(idx, fitz.Rect(rect), "image", repeated=xref_pages[xref] >= repeat_threshold)
                found.append(vis)
                if vis.repeated:
                    repeated_groups[xref].append(vis)
        for rect in layout_pictures.get(idx, []):
            r = fitz.Rect(rect)
            if r.get_area() / page_area >= MIN_IMAGE_PAGE_RATIO and not any(_overlap(r, v.rect) >= 0.5 for v in found):
                found.append(Visual(idx, r, "layout"))
        if not found:
            chart_rect = _vector_chart_rect(page)
            if chart_rect is not None:
                found.append(Visual(idx, chart_rect, "vector"))
        if found:
            scan.per_page[idx] = found[:6]

    # Una imagen repetida es un único elemento: se conserva solo su primera aparición.
    for group in repeated_groups.values():
        pages = sorted({v.page + 1 for v in group})
        group.sort(key=lambda v: v.page)
        for v in group:
            v.repeat_pages = pages
        for dup in group[1:]:
            scan.per_page[dup.page] = [v for v in scan.per_page.get(dup.page, []) if v is not dup]
            if not scan.per_page[dup.page]:
                scan.per_page.pop(dup.page, None)
            scan.repeated_occurrences += 1
    return scan


def _fmt_pages(pages: list[int]) -> str:
    if len(pages) > 1 and pages == list(range(pages[0], pages[-1] + 1)):
        return f"{pages[0]}-{pages[-1]}"
    shown = ", ".join(str(p) for p in pages[:8])
    return shown + (f" y {len(pages) - 8} más" if len(pages) > 8 else "")


def marker_text(v: Visual) -> str:
    if v.repeated:
        return f"[Imagen repetida omitida: páginas {_fmt_pages(v.repeat_pages)}]"
    omitted = "omitida" if v.source == "image" else "omitido"  # concordancia: imagen / dibujo / elemento
    return f"[{v.label} {omitted}: página {v.page + 1}]"


def treat_visual(
    doc: Any,
    v: Visual,
    *,
    image_mode: str,
    image_markers: bool,
    images_dir: Path,
    counter: dict[int, int],
    stats: VisualStats,
    warnings: list[str],
) -> str:
    """Devuelve el texto Markdown que representa a `v` ('' si no se incorpora nada)."""
    if image_mode == "extract" and stats.extracted < MAX_EXTRACTED_IMAGES:
        counter[v.page] = counter.get(v.page, 0) + 1
        name = f"page{v.page + 1:03d}-{counter[v.page]:02d}.png"
        try:
            images_dir.mkdir(parents=True, exist_ok=True)
            pix = doc[v.page].get_pixmap(clip=v.rect, dpi=EXTRACT_DPI)
            pix.save(str(images_dir / name))
            if (images_dir / name).exists():
                stats.extracted += 1
                alt = "Imagen repetida" if v.repeated else f"Imagen pág {v.page + 1}"
                return f"![{alt}](images/{name})"
        except Exception:
            log.exception("No se pudo extraer una imagen de la página %s", v.page + 1)
            warnings.append(f"No se pudo extraer una imagen de la página {v.page + 1}; se dejó un marcador.")
        stats.marked += 1
        return marker_text(v)
    if image_mode == "extract":  # límite alcanzado
        if stats.extracted == MAX_EXTRACTED_IMAGES and not any("límite de imágenes" in w for w in warnings):
            warnings.append(
                f"Se alcanzó el límite de imágenes extraídas ({MAX_EXTRACTED_IMAGES}); el resto se marcan como omitidas."
            )
        stats.marked += 1
        return marker_text(v)
    if image_mode == "markers" or (image_mode == "ignore" and image_markers):
        stats.marked += 1
        return marker_text(v)
    stats.omitted += 1
    return ""


# ---------------------------------------------------------------------------
# Artefactos de imagen que escribe el motor en su salida Markdown
# ---------------------------------------------------------------------------
_PICTURE_TEXT_RE = re.compile(
    r"(?:<!--\s*Start of picture text\s*-->|\**-{3,}\s*Start of picture text\s*-{3,}\**)(?P<body>.*?)"
    r"(?:<!--\s*End of picture text\s*-->|\**-{3,}\s*End of picture text\s*-{3,}\**)",
    re.IGNORECASE | re.DOTALL,
)
_PICTURE_OMITTED_RE = re.compile(r"^.*==>\s*picture.*<==.*$", re.IGNORECASE | re.MULTILINE)
_IMAGE_LINE_RE = re.compile(r"^[ \t]*!\[[^\]]*\]\([^)]*\)[ \t]*$", re.MULTILINE)


def strip_engine_image_artifacts(md: str) -> str:
    """
    Quita las referencias a imágenes del motor (rutas que no existirían en la descarga). El texto
    que el motor leyó DENTRO de las imágenes se conserva como líneas normales.
    """

    def keep_text(match: "re.Match[str]") -> str:
        body = re.sub(r"<br\s*/?>", "\n", match.group("body"), flags=re.IGNORECASE)
        return "\n".join(ln.strip() for ln in body.split("\n") if ln.strip())

    md = _PICTURE_TEXT_RE.sub(keep_text, md)
    md = _PICTURE_OMITTED_RE.sub("", md)
    return _IMAGE_LINE_RE.sub("", md)
