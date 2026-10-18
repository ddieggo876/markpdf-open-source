# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Métricas: definiciones únicas y documentadas.

  * Caracteres  = puntos de código Unicode del texto (`len(texto)`), incluidos saltos de línea.
  * Palabras    = secuencias de caracteres no blancos separadas por espacios en blanco (`len(texto.split())`).
                  Los símbolos Markdown (`|`, `#`, `-`) cuentan como palabras si van separados por espacios.
  * Tokens est. = ceil(caracteres / 4). ESTIMACIÓN aproximada y barata: no es un recuento exacto de ningún
                  modelo (cada tokenizador cuenta distinto). Sirve para comparar textos entre sí, no para
                  predecir el coste exacto de una llamada.

Se comparan siempre magnitudes de la MISMA fuente y con el MISMO estimador:

  * `source`     : texto plano de la capa de texto del PDF (PyMuPDF `get_text`), sin ninguna estructura.
  * `extraction` : Markdown tal como lo entrega la extracción, ANTES de la limpieza configurable.
  * `final`      : Markdown final (el que se copia/descarga; en el navegador se recalcula si se edita).

La reducción se define con denominador explícito:  (extracción − final) / extracción.
Una cifra negativa significa que el Markdown final es MÁS GRANDE que la referencia (por ejemplo por los
marcadores o notas añadidos): no se disfraza de ahorro.
"""

from __future__ import annotations

from typing import Any, Optional

TOKEN_ESTIMATOR = "ceil(caracteres / 4)"


def estimate_tokens(text: str) -> int:
    """Estimación aproximada (~4 caracteres por token). No es un recuento exacto."""
    return -(-len(text) // 4)


def count_words(text: str) -> int:
    return len(text.split())


def text_stats(text: str) -> dict[str, int]:
    return {"chars": len(text), "words": count_words(text), "tokens": estimate_tokens(text)}


def comparison(reference: dict[str, int], final: dict[str, int], reference_name: str) -> dict[str, Any]:
    """Compara el texto final con una referencia concreta, indicando fórmula y denominador."""
    ref_tokens, fin_tokens = reference["tokens"], final["tokens"]
    change: Optional[float] = round((fin_tokens - ref_tokens) / ref_tokens * 100, 1) if ref_tokens else None
    return {
        "reference": reference_name,
        "reference_tokens": ref_tokens,
        "final_tokens": fin_tokens,
        "change_percent": change,  # negativo = reducción respecto de la referencia
        "formula": "(tokens_final − tokens_referencia) / tokens_referencia × 100",
        "estimator": TOKEN_ESTIMATOR,
    }
