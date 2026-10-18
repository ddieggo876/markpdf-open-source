# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Límite opcional de hilos de ONNX Runtime (variable de entorno MARKPDF_ORT_THREADS).

Por defecto ONNX Runtime usa tantos hilos como CPUs ve el proceso. En contenedores con un tope de CPU (Render:
0,5 CPU con 16 CPUs visibles) esos hilos compiten por una cuota minúscula y gastan CPU sin hacer trabajo útil.

Sin la variable NO se toca nada: `onnxruntime.InferenceSession` sigue siendo la clase original. Con la variable
(entero >= 1) se sustituye por una subclase que, al crear una sesión:
  - conserva las opciones que pasó quien la crea (p. ej. `enable_cpu_mem_arena=False` de pymupdf-layout);
  - solo si esas opciones no fijan hilos (valor por defecto 0) pone `intra_op_num_threads=N` e `inter_op_num_threads=1`;
  - respeta los hilos que alguien haya fijado de forma explícita.
No altera el modelo, los proveedores ni los datos: el Markdown resultante es el mismo.

Se parchea la clase `InferenceSession` (no `make_session`) porque pymupdf-layout hace `from .common_util import
make_session` en varios módulos, y reasignar esa función no llegaría a todos. Debe aplicarse ANTES de importar
`pymupdf.layout` (que crea sus sesiones al importarse); `main.py` lo hace al principio.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

log = logging.getLogger("markpdf.ort_threads")

ENV_VAR = "MARKPDF_ORT_THREADS"

_original: Optional[type] = None  # InferenceSession original mientras el parche está activo


def threads_from_env() -> Optional[int]:
    """Valor de MARKPDF_ORT_THREADS; None si falta o no es un entero >= 1 (en ese caso no se cambia nada)."""
    raw = os.environ.get(ENV_VAR, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        value = 0
    if value < 1:
        log.warning("%s=%r no es válida (entero >= 1): se ignora y ONNX Runtime usa sus hilos por defecto.", ENV_VAR, raw)
        return None
    return value


def is_active() -> bool:
    return _original is not None


def apply(threads: int) -> bool:
    """Activa el límite. Devuelve False si onnxruntime no está instalado. Es idempotente."""
    global _original
    try:
        import onnxruntime as ort
    except ImportError:  # pragma: no cover
        return False
    if _original is not None:
        revert()
    base = ort.InferenceSession

    class _LimitedInferenceSession(base):  # type: ignore[valid-type, misc]
        def __init__(self, path_or_bytes: Any, sess_options: Any = None, *args: Any, **kwargs: Any) -> None:
            if sess_options is None:
                sess_options = ort.SessionOptions()
            if sess_options.intra_op_num_threads == 0:  # no fijado por quien crea la sesión
                sess_options.intra_op_num_threads = threads
                sess_options.inter_op_num_threads = 1
            super().__init__(path_or_bytes, sess_options, *args, **kwargs)

    _LimitedInferenceSession.__name__ = base.__name__
    _LimitedInferenceSession.__qualname__ = base.__qualname__
    _original = base
    ort.InferenceSession = _LimitedInferenceSession
    return True


def revert() -> None:
    """Quita el parche (para pruebas): vuelve a la clase original."""
    global _original
    if _original is None:
        return
    import onnxruntime as ort

    ort.InferenceSession = _original
    _original = None


def apply_from_env() -> Optional[int]:
    """Aplica el límite si MARKPDF_ORT_THREADS es válida. Devuelve los hilos configurados o None."""
    threads = threads_from_env()
    if threads is None or not apply(threads):
        return None
    import perf  # su logger ya escribe en stderr (uvicorn no configura el logger raíz)

    perf.logger.info("ort_threads active intra_op_num_threads=%d inter_op_num_threads=1 (%s)", threads, ENV_VAR)
    return threads
