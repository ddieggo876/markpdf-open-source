# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Opciones de conversión de MarkPDF.

Un único objeto (`ConvertOptions`) describe TODA la configuración de una conversión. Los tres
perfiles (`faithful`, `clean`, `compact`) no son motores distintos: solo fijan valores por
defecto de estas opciones. Cualquier opción enviada explícitamente por el cliente prevalece
sobre el valor por defecto del perfil.

Compatibilidad: los identificadores históricos de `mode` ("faithful", "clean", "compact") y de
`image_mode` ("ignore", "extract", "summarize_charts") se siguen aceptando.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Optional

MODES = ("faithful", "clean", "compact")
DEFAULT_MODE = "clean"  # "Optimizado para IA"
MODE_ALIASES = {"optimized": "clean", "optimizado": "clean", "fiel": "faithful", "compacto": "compact"}

IMAGE_MODES = ("ignore", "markers", "extract", "describe")
DEFAULT_IMAGE_MODE = "markers"
IMAGE_MODE_ALIASES = {"summarize_charts": "markers", "references": "extract"}

TABLE_FORMATS = ("auto", "markdown", "linear", "literal")
DEFAULT_TABLE_FORMAT = "auto"

PROFILE_LABELS = {"faithful": "Fiel al original", "clean": "Optimizado para IA", "compact": "Compacto"}

# Valores por defecto de las opciones booleanas "de limpieza" según el perfil.
_PROFILE_DEFAULTS: dict[str, dict[str, bool]] = {
    "faithful": dict(
        remove_headers_footers=False,
        remove_page_numbers=False,
        join_paragraphs=False,
        remove_duplicates=False,
    ),
    "clean": dict(
        remove_headers_footers=True,
        remove_page_numbers=True,
        join_paragraphs=True,
        remove_duplicates=True,
    ),
    "compact": dict(
        remove_headers_footers=True,
        remove_page_numbers=True,
        join_paragraphs=True,
        remove_duplicates=True,
    ),
}

# Opciones que descartan contenido: desactivadas en todos los perfiles; solo se activan de forma explícita.
_EXPLICIT_ONLY = ("omit_empty_fields", "omit_redundant_headings", "remove_repeated_blocks")

BOOL_OPTIONS = (
    "remove_headers_footers",
    "remove_page_numbers",
    "join_paragraphs",
    "remove_duplicates",
    "image_markers",
) + _EXPLICIT_ONLY


class OptionsError(ValueError):
    """Opción inválida (se traduce en HTTP 400 con un mensaje comprensible)."""


@dataclass(frozen=True)
class ConvertOptions:
    mode: str = DEFAULT_MODE
    image_mode: str = DEFAULT_IMAGE_MODE
    image_markers: bool = False  # solo tiene efecto con image_mode="ignore"
    table_format: str = DEFAULT_TABLE_FORMAT
    remove_headers_footers: bool = True
    remove_page_numbers: bool = True
    join_paragraphs: bool = True
    remove_duplicates: bool = True
    omit_empty_fields: bool = False
    omit_redundant_headings: bool = False
    remove_repeated_blocks: bool = False

    @property
    def compact_format(self) -> bool:
        """El perfil compacto reduce además el ruido de formato Markdown."""
        return self.mode == "compact"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def for_mode(cls, mode: str = DEFAULT_MODE, **overrides: Any) -> "ConvertOptions":
        """Opciones por defecto del perfil `mode`, con `overrides` (los valores None se ignoran)."""
        mode = normalize_mode(mode)
        values: dict[str, Any] = {"mode": mode, **_PROFILE_DEFAULTS[mode]}
        known = {f.name for f in fields(cls)}
        for key, value in overrides.items():
            if key not in known:
                raise OptionsError(f"Opción desconocida: {key!r}")
            if value is not None:
                values[key] = value
        opts = cls(**values)
        opts.validate()
        return opts

    def validate(self) -> None:
        if self.mode not in MODES:
            raise OptionsError(f"mode inválido. Opciones: {list(MODES)}")
        if self.image_mode not in IMAGE_MODES:
            raise OptionsError(f"image_mode inválido. Opciones: {list(IMAGE_MODES)}")
        if self.table_format not in TABLE_FORMATS:
            raise OptionsError(f"table_format inválido. Opciones: {list(TABLE_FORMATS)}")


def normalize_mode(mode: Optional[str]) -> str:
    mode = (mode or DEFAULT_MODE).strip().lower()
    mode = MODE_ALIASES.get(mode, mode)
    if mode not in MODES:
        raise OptionsError(f"mode inválido. Opciones: {list(MODES)}")
    return mode


def normalize_image_mode(image_mode: Optional[str]) -> str:
    image_mode = (image_mode or DEFAULT_IMAGE_MODE).strip().lower()
    image_mode = IMAGE_MODE_ALIASES.get(image_mode, image_mode)
    if image_mode not in IMAGE_MODES:
        raise OptionsError(f"image_mode inválido. Opciones: {list(IMAGE_MODES)}")
    return image_mode


def parse_bool(value: Any) -> Optional[bool]:
    """Interpreta un valor de formulario ('true', '1', 'on'...). None = no especificado."""
    if value is None or isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("", "default", "auto", "none"):
        return None
    if text in ("1", "true", "on", "yes", "si", "sí"):
        return True
    if text in ("0", "false", "off", "no"):
        return False
    raise OptionsError(f"Valor booleano inválido: {value!r}")


def options_from_form(values: Mapping[str, Any]) -> ConvertOptions:
    """Construye las opciones desde los campos de formulario de POST /convert."""
    overrides: dict[str, Any] = {}
    for name in BOOL_OPTIONS:
        if name in values:
            overrides[name] = parse_bool(values[name])
    if values.get("table_format"):
        overrides["table_format"] = str(values["table_format"]).strip().lower()
    overrides["image_mode"] = normalize_image_mode(values.get("image_mode"))
    return ConvertOptions.for_mode(values.get("mode") or DEFAULT_MODE, **overrides)
