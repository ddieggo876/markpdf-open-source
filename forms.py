# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Formularios: campos AcroForm y casillas de verificación.

Problema que resuelve (observado en la auditoría): al extraer un formulario con widgets, MuPDF
incluye en el texto el "aspecto" de cada widget. Una casilla marcada aparece como el carácter "3"
(glifo ZapfDingbats), una casilla vacía no deja rastro y un campo de texto vacío tampoco. Para una
IA eso significa perder el estado de las casillas y no distinguir "campo vacío" de "campo omitido".

Solución: antes de extraer, sobre una copia en memoria del documento (el archivo original no se
toca), cada widget se sustituye/complementa por texto estático explícito:

    casilla marcada       -> [x]
    casilla sin marcar    -> [ ]
    estado desconocido    -> [?]       (nunca se deduce: se avisa)
    campo de texto vacío  -> ___       (marcador de campo en blanco)
    campo cumplimentado   -> su valor  (ya aparece en el texto; si no, se añade)

Los marcadores `[x]`, `[ ]` y `___` son los mismos que usa el resto del pipeline, de modo que el
motor de extracción (tablas, párrafos) los trata como texto normal y la asociación etiqueta/valor
se conserva por posición.

También se interpretan glifos de casilla en texto estático (☐ ☑ ☒, Wingdings, ZapfDingbats...).
Los símbolos que no se pueden interpretar se marcan como `[?]` en lugar de adivinarse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

try:  # PyMuPDF >= 1.24.3
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

CHECKED = "[x]"
UNCHECKED = "[ ]"
UNKNOWN = "[?]"
BLANK = "___"

_OFF_VALUES = {"", "off", "false", "0", "no"}
_ON_VALUES = {"yes", "on", "true", "1", "sí", "si", "x"}


@dataclass
class FormReport:
    fields: int = 0
    text_filled: int = 0
    text_empty: int = 0
    checked: int = 0
    unchecked: int = 0
    unknown: int = 0
    details: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "fields": self.fields,
            "text_filled": self.text_filled,
            "text_empty": self.text_empty,
            "checked": self.checked,
            "unchecked": self.unchecked,
            "unknown": self.unknown,
        }


def _checkbox_state(widget: Any) -> Optional[bool]:
    """True/False si el estado se puede determinar con seguridad; None si no."""
    value = widget.field_value
    if isinstance(value, bool):
        return value
    text = "" if value is None else str(value).strip()
    on = None
    try:
        on = widget.on_state()
    except Exception:  # pragma: no cover
        pass
    if text.lower() in _OFF_VALUES:
        return False
    if isinstance(on, str) and on:
        return text == on  # radio: otro botón del grupo está seleccionado => False
    if text.lower() in _ON_VALUES:
        return True
    return None


def _burn(page: Any, rect: Any, text: str, *, fontsize: float) -> None:
    page.insert_text((rect.x0 + 1, rect.y0 + max(rect.height * 0.8, fontsize)), text, fontsize=fontsize, fontname="helv")


def prepare_widgets(doc: Any) -> FormReport:
    """Sustituye los widgets por texto explícito (ver docstring del módulo). Modifica `doc` en memoria."""
    report = FormReport()
    kinds = {
        "checkbox": fitz.PDF_WIDGET_TYPE_CHECKBOX,
        "radio": fitz.PDF_WIDGET_TYPE_RADIOBUTTON,
        "text": fitz.PDF_WIDGET_TYPE_TEXT,
        "combo": fitz.PDF_WIDGET_TYPE_COMBOBOX,
        "list": fitz.PDF_WIDGET_TYPE_LISTBOX,
    }
    try:
        pages = list(doc)
    except Exception:  # pragma: no cover
        return report
    for page in pages:
        try:
            widget = page.first_widget
        except Exception:  # pragma: no cover
            continue
        burns: list[tuple[Any, str, float]] = []
        while widget:
            ftype = widget.field_type
            rect = fitz.Rect(widget.rect)
            name = widget.field_name or ""
            keep_next = True
            if ftype in (kinds["checkbox"], kinds["radio"]):
                state = _checkbox_state(widget)
                marker = UNKNOWN if state is None else (CHECKED if state else UNCHECKED)
                report.fields += 1
                if state is None:
                    report.unknown += 1
                elif state:
                    report.checked += 1
                else:
                    report.unchecked += 1
                report.details.append({"page": page.number + 1, "name": name, "type": "checkbox", "marker": marker})
                burns.append((rect, marker, max(5.0, min(10.0, rect.width / 1.1))))
                keep_next = False
            elif ftype in (kinds["text"], kinds["combo"], kinds["list"]):
                value = widget.field_value
                value = "" if value is None else (", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value))
                report.fields += 1
                if value.strip():
                    report.text_filled += 1
                    shown = page.get_text("text", clip=rect).strip()
                    if not shown:  # el valor existe pero el widget no lo dibuja
                        burns.append((rect, value.strip(), max(6.0, min(10.0, rect.height * 0.7))))
                    report.details.append({"page": page.number + 1, "name": name, "type": "text", "marker": "value"})
                else:
                    report.text_empty += 1
                    burns.append((rect, BLANK, max(6.0, min(10.0, rect.height * 0.7))))
                    report.details.append({"page": page.number + 1, "name": name, "type": "text", "marker": BLANK})
            if keep_next:
                widget = widget.next
            else:
                widget = page.delete_widget(widget)  # devuelve el siguiente widget
        for rect, text, size in burns:
            try:
                _burn(page, rect, text, fontsize=size)
            except Exception:  # pragma: no cover
                pass
    return report


# ---------------------------------------------------------------------------
# Glifos de casilla en texto estático
# ---------------------------------------------------------------------------
_UNICODE_UNCHECKED = "☐□▢❏❐❑❒⬜◻⎕"
_UNICODE_CHECKED = "☑☒⌧✅✔✓✗✘🗹🗵"
_UNICODE_TABLE = {**{c: UNCHECKED for c in _UNICODE_UNCHECKED}, **{c: CHECKED for c in _UNICODE_CHECKED}}
_UNICODE_RE = re.compile("[" + re.escape(_UNICODE_UNCHECKED + _UNICODE_CHECKED) + "]")

_WINGDINGS = {
    0xA8: UNCHECKED, 0x6F: UNCHECKED, 0x71: UNCHECKED, 0x72: UNCHECKED,
    0xFE: CHECKED, 0xFD: CHECKED, 0xFC: CHECKED, 0xFB: CHECKED, 0x78: CHECKED,
    0x6C: "•", 0x6E: "•", 0xA7: "•", 0xD8: "•", 0x76: "•", 0x9F: "•",
}
_ZAPF = {
    "3": CHECKED, "4": CHECKED, "7": CHECKED, "8": CHECKED,
    "o": UNCHECKED, "p": UNCHECKED, "q": UNCHECKED, "r": UNCHECKED,
    "n": "•", "l": "•", "m": "•",
}
_SYMBOL = {0xB7: "•"}


def normalize_checkbox_glyphs(text: str) -> str:
    """Convierte glifos Unicode de casilla (☐ ☑ ☒ ...) en [ ] / [x]."""
    return _UNICODE_RE.sub(lambda m: _UNICODE_TABLE[m.group(0)], text)


def _is_pua(ch: str) -> bool:
    return 0xE000 <= ord(ch) <= 0xF8FF


def symbol_font_family(font_name: str) -> Optional[str]:
    name = (font_name or "").lower().replace(" ", "")
    if "wingdings" in name:
        return "wingdings"
    if "zapf" in name or "dingbats" in name:
        return "zapf"
    if name.endswith("symbol") or name.startswith("symbol") or "symbolmt" in name:
        return "symbol"
    return None


def map_span_text(text: str, font_name: str) -> tuple[str, int]:
    """
    Interpreta el texto de un span según su fuente. Devuelve (texto, nº de símbolos sin interpretar).
    Solo se interpretan fuentes de símbolos conocidas; el resto del texto se devuelve intacto.
    """
    family = symbol_font_family(font_name)
    if family is None:
        # Fuente normal: solo se retiran caracteres de uso privado aislados (no tienen significado).
        out = normalize_checkbox_glyphs(text)
        unknown = sum(1 for ch in out if _is_pua(ch))
        return "".join("" if _is_pua(ch) else ch for ch in out), unknown
    unknown = 0
    out: list[str] = []
    for ch in text:
        if ch.isspace():
            out.append(ch)
            continue
        code = (ord(ch) & 0xFF) if (_is_pua(ch) or ord(ch) < 256) else None
        mapped: Optional[str] = None
        if code is not None:
            if family == "wingdings":
                mapped = _WINGDINGS.get(code)
            elif family == "zapf":
                mapped = _ZAPF.get(chr(code))
            elif family == "symbol":
                mapped = _SYMBOL.get(code)
        if mapped is None and ch in _UNICODE_TABLE:
            mapped = _UNICODE_TABLE[ch]
        if mapped is None:
            if family == "symbol" and not _is_pua(ch) and ord(ch) >= 32 and ch.isalnum():
                mapped = ch  # texto griego/matemático normal escrito con la fuente Symbol
            else:
                mapped = UNKNOWN
                unknown += 1
        out.append(mapped)
    return "".join(out), unknown
