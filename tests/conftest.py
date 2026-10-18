# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Configuración común de las pruebas."""

import os

import pytest

import main


@pytest.fixture(autouse=True)
def _rate_limit_off_for_legacy_tests(monkeypatch):
    """
    Las pruebas de conversión hacen muchas peticiones desde la misma IP de pruebas, y el límite por hora (10) las
    bloquearía. Se desactiva aquí; `tests/test_security.py` lo activa de forma explícita con su propio limitador.
    """
    monkeypatch.setattr(main.RATE_LIMITER, "max_hits", 0)
    main.RATE_LIMITER.reset()
    yield
    main.RATE_LIMITER.reset()


# ---------------------------------------------------------------------------
# Almacenamiento de cuotas: la misma batería se ejecuta contra memoria y contra PostgreSQL real.
# PostgreSQL solo se usa si TEST_DATABASE_URL apunta a una base DESECHABLE (las tablas se vacían en cada test).
# Sin esa variable (o sin psycopg instalado) los tests de PostgreSQL se omiten.
# ---------------------------------------------------------------------------
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", "").strip()


def truncate_postgres(url: str) -> None:
    import psycopg

    with psycopg.connect(url) as conn:
        conn.execute("TRUNCATE users, daily_usage RESTART IDENTITY CASCADE")


@pytest.fixture(params=["memory", "postgres"])
def store(request):
    from db import MemoryStore, PostgresStore

    if request.param == "memory":
        return MemoryStore()
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL no definida: se omiten las pruebas con PostgreSQL real")
    pytest.importorskip("psycopg")
    pg = PostgresStore(TEST_DATABASE_URL)
    pg.init()
    truncate_postgres(TEST_DATABASE_URL)
    return pg
