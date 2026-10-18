# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Descripción visual de imágenes: PUNTO DE INTEGRACIÓN (no implementado).

MarkPDF no incluye ningún modelo de visión, por lo que la opción `image_mode="describe"` NO está
disponible y la API la rechaza con un mensaje claro. No se simula nunca un análisis de imagen.

Para integrarlo, implementa `describe_image()` (p. ej. llamando a un modelo local o a un servicio
autorizado) y haz que `is_available()` devuelva True cuando esté configurado. La descripción debería
separar: (1) texto realmente legible, (2) contenido visual observado, (3) información incierta.
"""

from __future__ import annotations

UNAVAILABLE_MESSAGE = (
    "La descripción visual no está disponible: no hay ningún modelo de visión configurado en este servidor. "
    "Usa «Ignorar», «Marcar posición» o «Extraer referencias»."
)


def is_available() -> bool:
    return False


def describe_image(png_bytes: bytes) -> str:  # pragma: no cover - sin implementación a propósito
    raise NotImplementedError(UNAVAILABLE_MESSAGE)
