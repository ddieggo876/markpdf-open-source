# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Etapa de extracción: PDF -> Markdown por página (sin limpiar).

Flujo por página:
  1. pymupdf4llm (motor de layout si está instalado `pymupdf-layout`) devuelve el Markdown de la página y,
     con `page_chunks=True`, las cajas que detectó (tabla, título, texto, imagen...) con su posición.
  2. Las tablas se serializan con `tables.render_table()` según `table_format`.
  3. El texto de las cajas se relee del PDF línea a línea (`lines.py`) para no perder la estructura
     de formularios, listas y guiones de fin de línea.
  4. Las imágenes y dibujos se tratan según `image_mode` (`visuals.py`) y se colocan en su posición.
  5. OCR solo para páginas sin texto seleccionable y solo si hay un motor instalado.

Si pymupdf4llm no devuelve cajas (motor clásico) se usa su texto tal cual, con los mismos tratamientos
de imágenes y casillas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:  # PyMuPDF >= 1.24.3
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

import pymupdf4llm

from forms import normalize_checkbox_glyphs
from lines import rebuild_boxes
import perf
from options import ConvertOptions
from tables import render_table
from visuals import VisualStats, detect_visuals, strip_engine_image_artifacts, treat_visual

log = logging.getLogger("markpdf.extraction")

MIN_TEXT_CHARS = 20  # una página con menos caracteres alfanuméricos no tiene "texto útil"
SCAN_IMAGE_RATIO = 0.30  # imagen que cubre >=30% de la página => probable escaneo
SCANNED_MARKER = "[Página sin texto extraíble (probable escaneo): página {n}]"
_HEADING_CLASSES = ("title", "section-header")


@dataclass
class Extraction:
    pages: list[str]
    warnings: list[str] = field(default_factory=list)
    tables: int = 0
    tables_approximate: int = 0
    table_formats: dict[str, int] = field(default_factory=dict)
    visuals: VisualStats = field(default_factory=VisualStats)
    scanned_pages: list[int] = field(default_factory=list)  # 0-based
    ocr_pages: list[int] = field(default_factory=list)  # 0-based
    unknown_symbols: int = 0
    layout_engine: bool = False

    @property
    def no_text_layer(self) -> set[int]:
        return set(self.scanned_pages) | set(self.ocr_pages)


# ---------------------------------------------------------------------------
# Utilidades del motor
# ---------------------------------------------------------------------------
def _call_to_markdown(doc: "fitz.Document", **opts: Any) -> list[dict[str, Any]]:
    """
    pymupdf4llm.to_markdown() es un envoltorio (*args, **kwargs): no se puede filtrar por firma. Se pasan
    todas las opciones y, si la versión instalada no las admite, se reintenta solo con las imprescindibles.
    """
    try:
        result = pymupdf4llm.to_markdown(doc, **opts)
    except TypeError:  # pragma: no cover - versiones antiguas con firma estricta
        basic = {k: opts[k] for k in ("pages", "page_chunks") if k in opts}
        result = pymupdf4llm.to_markdown(doc, **basic)
    if isinstance(result, str):
        return [{"text": result}]
    return list(result)


def _ocr_available() -> bool:
    """True si pymupdf4llm puede usar un motor de OCR instalado (Tesseract, RapidOCR...)."""
    if not getattr(pymupdf4llm, "_use_layout", False):
        return False
    try:
        from pymupdf4llm.helpers.document_layout import select_ocr_function

        return callable(select_ocr_function())
    except Exception:  # pragma: no cover
        return False


def layout_engine_available() -> bool:
    return bool(getattr(pymupdf4llm, "_use_layout", False))


def _alnum_count(text: str) -> int:
    return sum(ch.isalnum() for ch in text)


def _has_large_image(page: "fitz.Page") -> bool:
    area = page.rect.get_area() or 1.0
    for img in page.get_images(full=True):
        try:
            rects = page.get_image_rects(img[0])
        except Exception:  # pragma: no cover
            continue
        if any(r.get_area() / area >= SCAN_IMAGE_RATIO for r in rects):
            return True
    return False


def find_scanned_pages(doc: "fitz.Document") -> list[int]:
    """Páginas sin texto útil que contienen una imagen grande (probables escaneos)."""
    scanned = []
    for idx, page in enumerate(doc):
        if _alnum_count(page.get_text("text")) < MIN_TEXT_CHARS and _has_large_image(page):
            scanned.append(idx)
    return scanned


def _plain_page_text(page: "fitz.Page") -> str:
    """Texto simple de una página ordenado por bloques (respaldo si pymupdf4llm no devuelve nada)."""
    blocks = [b for b in page.get_text("blocks", sort=True) if b[6] == 0 and b[4].strip()]
    return "\n\n".join(b[4].strip() for b in blocks)


def _fmt_pages(indices: list[int]) -> str:
    nums = [str(i + 1) for i in indices]
    return ", ".join(nums[:10]) + (f" y {len(nums) - 10} más" if len(nums) > 10 else "")


def _pages_phrase(indices: list[int]) -> str:
    """'la página 3' / 'las páginas 3, 4'."""
    return f"la página {_fmt_pages(indices)}" if len(indices) == 1 else f"las páginas {_fmt_pages(indices)}"


# ---------------------------------------------------------------------------
# Ensamblado de una página a partir de las cajas del motor
# ---------------------------------------------------------------------------
@dataclass
class _Part:
    text: str
    bbox: Optional[Any] = None  # fitz.Rect


def _insert_index(parts: list[_Part], rect: Any) -> int:
    """Posición en la que colocar un elemento visual: tras el último bloque situado por encima de él."""
    same_col = [
        i
        for i, p in enumerate(parts)
        if p.bbox is not None
        and min(p.bbox.x1, rect.x1) - max(p.bbox.x0, rect.x0) > 0.3 * min(p.bbox.width, rect.width, 1e9)
    ]
    candidates = same_col or [i for i, p in enumerate(parts) if p.bbox is not None]
    idx = 0
    for i in candidates:
        if parts[i].bbox.y1 <= rect.y0 + 4:  # type: ignore[union-attr]
            idx = max(idx, i + 1)
    return idx


def _assemble_page(
    doc: "fitz.Document",
    idx: int,
    chunk: dict[str, Any],
    opts: ConvertOptions,
    ext: Extraction,
) -> list[_Part]:
    """Piezas (texto + posición) de una página con cajas de layout; lista vacía si no se puede."""
    boxes = chunk.get("page_boxes") or []
    text = chunk.get("text", "")
    if not boxes:
        return []
    page = doc[idx]
    for b in boxes:
        b["_rect"] = fitz.Rect(*b["bbox"])
    excluded = {i for i, b in enumerate(boxes) if b["class"] == "table"}
    rebuild = {i for i, b in enumerate(boxes) if b["class"] not in _HEADING_CLASSES and i not in excluded}
    rebuilt = rebuild_boxes(page, boxes, rebuild, excluded)
    ext.unknown_symbols += rebuilt.unknown_symbols

    parts: list[_Part] = []
    for i, box in enumerate(boxes):
        cls = box["class"]
        start, stop = box["pos"]
        piece = text[start:stop].strip("\n")
        if cls == "table":
            table = render_table(piece, opts.table_format, opts.omit_empty_fields)
            ext.tables += 1
            ext.tables_approximate += int(table.approximate)
            ext.table_formats[table.format] = ext.table_formats.get(table.format, 0) + 1
            out = table.text
        elif cls in _HEADING_CLASSES:
            out = strip_engine_image_artifacts(piece).strip()
        else:
            out = rebuilt.texts.get(i)
            if out is None:
                out = "" if cls == "picture" else strip_engine_image_artifacts(piece).strip()
            elif piece.lstrip().startswith("```"):
                out = piece.strip()  # bloque de código: se respeta el formato del motor
        if out.strip():
            parts.append(_Part(out.strip("\n"), box["_rect"]))
        elif cls == "picture":
            parts.append(_Part("", box["_rect"]))  # sin texto, pero conserva la posición
    return parts


# ---------------------------------------------------------------------------
# Extracción
# ---------------------------------------------------------------------------
def extract_pages(doc: "fitz.Document", opts: ConvertOptions, images_dir: Path) -> Extraction:
    n_pages = len(doc)
    ext = Extraction(pages=[""] * n_pages, layout_engine=layout_engine_available())
    warnings = ext.warnings

    engine_opts: dict[str, Any] = {
        "pages": list(range(n_pages)),
        "page_chunks": True,
        "show_progress": False,
        "use_ocr": False,  # el OCR se decide aquí, no de forma implícita
    }
    chunks: list[dict[str, Any]] = []
    try:
        with perf.stage("layout"):  # análisis de layout + generación del Markdown base (pymupdf4llm)
            raw_chunks = _call_to_markdown(doc, **engine_opts)
        chunks = [c if isinstance(c, dict) else {"text": str(c)} for c in raw_chunks]
        if len(chunks) != n_pages:
            raise ValueError(f"pymupdf4llm devolvió {len(chunks)} fragmentos para {n_pages} páginas")
    except Exception:
        log.exception("Fallo de pymupdf4llm; se usa extracción de texto simple")
        warnings.append("La extracción estructurada falló; se ha usado texto simple sin formato Markdown.")
        chunks = [{"text": _plain_page_text(p)} for p in doc]

    texts = [c.get("text", "") for c in chunks]
    # Respaldo página a página: si pymupdf4llm no devolvió nada pero la página sí tiene texto.
    for idx, page in enumerate(doc):
        if not texts[idx].strip() and _alnum_count(page.get_text("text")) >= MIN_TEXT_CHARS:
            texts[idx] = _plain_page_text(page)
            chunks[idx] = {"text": texts[idx]}

    # OCR solo para páginas sin texto útil, y solo si hay un motor disponible.
    scanned_all = find_scanned_pages(doc)
    ext.scanned_pages = [i for i in scanned_all if not texts[i].strip()]
    ocr_done: set[int] = set()
    if ext.scanned_pages:
        if _ocr_available():
            try:
                with perf.stage("ocr"):
                    ocr_chunks = _call_to_markdown(
                        doc, pages=ext.scanned_pages, page_chunks=True, show_progress=False, force_ocr=True
                    )
                for idx, chunk in zip(ext.scanned_pages, ocr_chunks):
                    texts[idx] = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
                    chunks[idx] = {"text": texts[idx]}
                    ocr_done.add(idx)
                warnings.append(
                    f"Se aplicó OCR en {_pages_phrase(ext.scanned_pages)} (sin texto seleccionable). "
                    "Revisa el resultado: el OCR puede cometer errores."
                )
            except Exception:  # pragma: no cover
                log.exception("Fallo del OCR")
                warnings.append(f"El OCR falló en {_pages_phrase(ext.scanned_pages)}.")
        else:
            many = len(ext.scanned_pages) > 1
            verb, looks = ("tienen", "parecen escaneadas") if many else ("tiene", "parece escaneada")
            warnings.append(
                f"{_pages_phrase(ext.scanned_pages).capitalize()} no {verb} texto seleccionable ({looks}) "
                "y no hay un motor de OCR instalado. Instala Tesseract o RapidOCR para extraer su texto."
            )
        ext.ocr_pages = sorted(ocr_done)
        ext.scanned_pages = [i for i in ext.scanned_pages if i not in ocr_done]

    # Imágenes: detección (independiente del perfil de texto) y tratamiento según image_mode.
    layout_pictures: dict[int, list[Any]] = {}
    for idx, chunk in enumerate(chunks):
        pics = [fitz.Rect(*b["bbox"]) for b in (chunk.get("page_boxes") or []) if b.get("class") == "picture"]
        if pics:
            layout_pictures[idx] = pics
    scan = detect_visuals(doc, layout_pictures=layout_pictures, skip_pages=set(scanned_all))
    stats = ext.visuals
    stats.small_ignored = scan.small_ignored
    stats.relevant, stats.repeated = scan.relevant, scan.repeated
    stats.detected = stats.relevant + stats.repeated + stats.small_ignored
    counter: dict[int, int] = {}

    for idx in range(n_pages):
        if idx in ocr_done or idx in ext.scanned_pages:
            parts: list[_Part] = []
        else:
            parts = _assemble_page(doc, idx, chunks[idx], opts, ext)
        use_parts = bool(parts)
        if not use_parts:
            base = strip_engine_image_artifacts(texts[idx]).strip()
            if chunks[idx].get("tables") and not chunks[idx].get("page_boxes"):
                ext.tables += len(chunks[idx]["tables"])  # motor clásico: solo se puede contar
            parts = [_Part(base)] if base else []
        for visual in scan.per_page.get(idx, []):
            out = treat_visual(
                doc,
                visual,
                image_mode=opts.image_mode,
                image_markers=opts.image_markers,
                images_dir=images_dir,
                counter=counter,
                stats=stats,
                warnings=warnings,
            )
            if not out:
                continue
            if use_parts:
                parts.insert(_insert_index(parts, visual.rect), _Part(out, visual.rect))
            else:
                parts.append(_Part(out))
        page_text = "\n\n".join(p.text for p in parts if p.text.strip())
        ext.pages[idx] = normalize_checkbox_glyphs(page_text)

    # Páginas escaneadas sin OCR: dejar constancia solo si el documento tiene texto en otras páginas.
    if any(p.strip() for p in ext.pages):
        for idx in ext.scanned_pages:
            ext.pages[idx] = SCANNED_MARKER.format(n=idx + 1)

    if ext.unknown_symbols:
        warnings.append(
            f"Se encontraron {ext.unknown_symbols} símbolos de fuente no interpretables (viñetas o casillas). "
            "Se marcaron como [?] o se descartaron; si eran casillas, su estado es desconocido."
        )
    if ext.tables_approximate:
        warnings.append(
            f"{ext.tables_approximate} de {ext.tables} tablas tienen una estructura aproximada "
            "(consulta la nota que precede a cada una)."
        )
    return ext
