# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Utilidades compartidas por los tests de cuotas."""

from __future__ import annotations

import datetime as dt
import importlib.util
import os
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_main_copy(monkeypatch, **env):
    """
    Carga una copia independiente de main.py con las variables de entorno indicadas.
    Se borran antes todas las MARKPDF_* y DATABASE_URL para que el entorno de quien ejecuta los tests no influya.
    """
    for key in [k for k in os.environ if k.startswith("MARKPDF_") or k == "DATABASE_URL"]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    name = f"main_quota_copy_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class FakeClock:
    """Reloj controlable (siempre UTC) para probar el reinicio diario sin esperar."""

    def __init__(self, start: dt.datetime) -> None:
        self.current = start

    def __call__(self) -> dt.datetime:
        return self.current

    def set(self, *args) -> None:
        self.current = dt.datetime(*args, tzinfo=dt.timezone.utc)

    def advance(self, **kwargs) -> None:
        self.current += dt.timedelta(**kwargs)
