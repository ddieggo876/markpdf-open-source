# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Cuotas y protección de POST /convert.

* Con MARKPDF_AUTH_ENABLED=false (por defecto): comportamiento de siempre, incluido el límite por IP.
* Con login activo: sin sesión 401, cuota agotada 429, sin límite por IP, y la cuota se reserva antes de
  convertir y se devuelve si la conversión falla.

Aquí la identidad se simula sustituyendo `main._current_user_id`; el flujo real con Google (sesión incluida)
está en test_auth.py. Las pruebas con `store` se ejecutan contra memoria y, si hay TEST_DATABASE_URL, PostgreSQL.
"""

from __future__ import annotations

import datetime as dt
import threading
import time

import pytest
from fastapi.testclient import TestClient

import main
from db import MemoryStore, QuotaStoreError
from quotas import CODE_HEADER, QuotaService
from tests import pdf_fixtures as fx
from tests.helpers import FakeClock, load_main_copy

UTC = dt.timezone.utc


def post(client, pdf=None, **data):
    pdf = fx.clean_pdf() if pdf is None else pdf
    return client.post("/convert", files={"file": ("doc.pdf", pdf, "application/pdf")}, data=data)


def new_client():
    return TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture
def client():
    return new_client()


def _service(store):
    clock = FakeClock(dt.datetime(2026, 10, 18, 12, 0, tzinfo=UTC))
    return QuotaService(store, daily_limit=3, clock=clock)


@pytest.fixture
def service(monkeypatch, store):
    svc = _service(store)
    monkeypatch.setattr(main, "QUOTAS", svc)
    return svc


@pytest.fixture
def memory_service(monkeypatch):
    """Como `service`, pero siempre en memoria: para las pruebas que sustituyen el almacenamiento por uno roto."""
    svc = _service(MemoryStore())
    monkeypatch.setattr(main, "QUOTAS", svc)
    return svc


@pytest.fixture
def fake_engine(monkeypatch):
    """Motor instantáneo; devuelve la lista de llamadas (para saber si llegó a ejecutarse)."""
    calls = []

    def engine(data, options=None):
        calls.append(1)
        return {"markdown": "# ok", "warnings": []}

    monkeypatch.setattr(main, "convert_pdf", engine)
    return calls


@pytest.fixture
def broken_engine(monkeypatch):
    def engine(data, options=None):
        raise RuntimeError("fallo inesperado del motor")

    monkeypatch.setattr(main, "convert_pdf", engine)


@pytest.fixture
def as_user(monkeypatch, service):
    """Las peticiones son del usuario devuelto (simula una sesión válida)."""
    uid = service.store.upsert_user("google-sub-1", "ana@example.com", "Ana")
    monkeypatch.setattr(main, "_current_user_id", lambda request: uid)
    return uid


def usage(service, uid):
    return service.store.usage_user(uid, service.today())


# ===========================================================================
# Sin autenticación (por defecto): comportamiento actual de la web publicada
# ===========================================================================
def test_auth_is_disabled_by_default(monkeypatch):
    m = load_main_copy(monkeypatch)
    assert m.AUTH_ENABLED is False and m.QUOTAS is None
    assert m.DAILY_FREE_CONVERSIONS == 3
    with TestClient(m.app):
        assert m.QUOTAS is None  # arrancar tampoco crea nada


def test_the_real_module_has_no_quota_service():
    assert main.QUOTAS is None


def test_disabled_flag_converts_without_login_or_cookies(client, fake_engine):
    for _ in range(6):  # más que cualquier cuota
        r = post(client)
        assert r.status_code == 200
        assert "set-cookie" not in r.headers
    assert len(fake_engine) == 6


def test_disabled_flag_keeps_the_per_ip_limit(client, fake_engine, monkeypatch):
    monkeypatch.setattr(main.RATE_LIMITER, "max_hits", 2)
    main.RATE_LIMITER.reset()
    assert [post(client).status_code for _ in range(3)] == [200, 200, 429]


def test_disabled_flag_ignores_session_cookies(client, fake_engine):
    client.cookies.set("mp_session", "basura")
    assert post(client).status_code == 200


def test_auth_routes_do_not_exist_when_disabled(client):
    assert client.get("/auth/me").json() == {"enabled": False}
    assert client.get("/auth/login", follow_redirects=False).status_code == 404
    assert client.get("/auth/callback", follow_redirects=False).status_code == 404
    assert client.post("/auth/logout").status_code == 404


def test_capabilities_expose_nothing_about_accounts(client):
    body = client.get("/capabilities").json()
    assert not {"auth", "quota", "quotas", "user", "login"} & set(body)


# ===========================================================================
# Con login: acceso directo a POST /convert
# ===========================================================================
def test_convert_without_session_is_401_and_never_runs_the_engine(client, service, fake_engine, monkeypatch):
    monkeypatch.setattr(main, "_current_user_id", lambda request: None)
    r = post(client)
    assert r.status_code == 401
    assert r.headers[CODE_HEADER] == "login_required"
    assert "Inicia sesión" in r.json()["detail"]
    assert not fake_engine


def test_401_comes_before_any_other_validation(client, service, fake_engine, monkeypatch):
    monkeypatch.setattr(main, "_current_user_id", lambda request: None)
    assert post(client, b"esto no es un pdf").status_code == 401
    assert post(client, mode="no-existe").status_code == 401


def test_authenticated_conversion_works_with_the_real_engine(client, service, as_user):
    r = post(client)
    assert r.status_code == 200 and r.json()["markdown"].strip()
    assert "set-cookie" not in r.headers
    assert usage(service, as_user) == 1


# ===========================================================================
# Cuota: 3 al día, reinicio a las 00:00 UTC
# ===========================================================================
def test_registered_user_gets_three_then_429(client, service, as_user, fake_engine):
    for _ in range(3):
        assert post(client).status_code == 200
    denied = post(client)
    assert denied.status_code == 429
    assert denied.headers[CODE_HEADER] == "daily_quota_exhausted"
    assert denied.headers["Retry-After"] == str(12 * 3600)
    assert "3 conversiones" in denied.json()["detail"]
    assert len(fake_engine) == 3
    assert usage(service, as_user) == 3


def test_quota_resets_at_midnight_utc(client, service, as_user, fake_engine):
    service._clock.set(2026, 10, 18, 23, 59, 58)
    for _ in range(3):
        assert post(client).status_code == 200
    assert post(client).status_code == 429
    service._clock.set(2026, 10, 19, 0, 0, 0)
    assert post(client).status_code == 200
    assert usage(service, as_user) == 1


def test_quota_is_per_user(client, service, as_user, fake_engine, monkeypatch):
    for _ in range(3):
        post(client)
    other = service.store.upsert_user("google-sub-2", "luis@example.com", "Luis")
    monkeypatch.setattr(main, "_current_user_id", lambda request: other)
    assert post(client).status_code == 200


def test_disabled_user_gets_403(client, service, as_user, fake_engine):
    service.store.set_user_active(as_user, False)
    r = post(client)
    assert r.status_code == 403 and r.headers[CODE_HEADER] == "account_disabled"
    assert not fake_engine


def test_too_many_pages_does_not_cost_quota(client, service, as_user, monkeypatch):
    monkeypatch.setattr(main, "MAX_PAGES", 2)
    assert post(client, fx.contract_pdf(5)).status_code == 413  # lo detecta el motor, ya con la cuota reservada
    assert usage(service, as_user) == 0


def test_failed_conversions_do_not_cost_quota(client, service, as_user):
    assert post(client, fx.corrupt_pdf()).status_code == 422  # PDF dañado
    assert post(client, fx.encrypted_pdf()).status_code == 422  # con contraseña
    assert usage(service, as_user) == 0
    for _ in range(3):  # sigue teniendo sus 3 conversiones
        assert post(client).status_code == 200
    assert post(client).status_code == 429


def test_unexpected_engine_crash_does_not_cost_quota(client, service, as_user, broken_engine):
    for _ in range(5):
        assert post(client).status_code == 500
    assert usage(service, as_user) == 0


def test_requests_rejected_before_conversion_never_touch_the_quota(client, service, as_user, fake_engine, monkeypatch):
    assert post(client, b"esto no es un pdf").status_code == 400
    assert post(client, b"").status_code == 400
    assert post(client, mode="no-existe").status_code == 400
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    assert post(client).status_code == 413
    assert usage(service, as_user) == 0 and not fake_engine


def test_busy_server_503_does_not_cost_quota(client, service, as_user, fake_engine, monkeypatch):
    slots = threading.BoundedSemaphore(1)
    monkeypatch.setattr(main, "CONVERT_SLOTS", slots)
    slots.acquire()
    r = post(client)
    assert r.status_code == 503 and "ocupado" in r.json()["detail"]
    slots.release()
    assert usage(service, as_user) == 0 and not fake_engine
    assert post(client).status_code == 200


# ===========================================================================
# El límite por IP solo se omite con autenticación activada
# ===========================================================================
def test_ip_limit_is_skipped_when_auth_is_enabled(client, service, as_user, fake_engine, monkeypatch):
    monkeypatch.setattr(main.RATE_LIMITER, "max_hits", 1)  # con el límite activo, la 2.ª petición sería 429
    main.RATE_LIMITER.reset()
    service.daily_limit = 5
    assert [post(client).status_code for _ in range(5)] == [200] * 5
    assert post(client).status_code == 429 and main.RATE_LIMITER.max_hits == 1  # ahora corta la CUOTA, no la IP
    assert post(client).headers[CODE_HEADER] == "daily_quota_exhausted"


def test_ip_limiter_is_not_even_consulted_when_auth_is_enabled(client, service, as_user, fake_engine, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("el límite por IP no debe consultarse con login activo")

    monkeypatch.setattr(main.RATE_LIMITER, "retry_after", boom)
    monkeypatch.setattr(main.RATE_LIMITER, "hit", boom)
    assert post(client).status_code == 200


def test_ip_limit_still_applies_to_everybody_when_auth_is_disabled(client, fake_engine, monkeypatch):
    monkeypatch.setattr(main.RATE_LIMITER, "max_hits", 1)
    main.RATE_LIMITER.reset()
    assert post(client).status_code == 200
    r = post(client)
    assert r.status_code == 429 and "por hora" in r.json()["detail"]


# ===========================================================================
# Errores de la base de datos
# ===========================================================================
class DownStore(MemoryStore):
    def reserve_user(self, *a, **k):
        raise QuotaStoreError("base de datos caída")


class ForgetfulStore(MemoryStore):
    """Reserva bien pero no consigue devolver (por ejemplo, la base de datos cae a mitad de conversión)."""

    def release_user(self, *a, **k):
        raise QuotaStoreError("base de datos caída")


def test_database_down_fails_closed_and_frees_the_slot(client, memory_service, fake_engine, monkeypatch):
    monkeypatch.setattr(main, "_current_user_id", lambda request: 1)
    healthy = memory_service.store
    uid = healthy.upsert_user("google-sub-1", "ana@example.com", "Ana")
    assert uid == 1
    memory_service.store = DownStore()
    r = post(client)
    assert r.status_code == 503
    assert r.headers[CODE_HEADER] == "quota_unavailable" and r.headers["Retry-After"] == "30"
    assert "límite de conversiones" in r.json()["detail"]
    assert "caída" not in r.text  # sin detalles internos
    assert not fake_engine  # nunca se convierte sin poder contar la cuota
    memory_service.store = healthy  # la base de datos vuelve
    assert post(client).status_code == 200  # y el hueco de conversión se había liberado


def test_failed_refund_does_not_hide_the_original_error(client, memory_service, broken_engine, caplog, monkeypatch):
    memory_service.store = ForgetfulStore()
    uid = memory_service.store.upsert_user("google-sub-1", "ana@example.com", "Ana")
    monkeypatch.setattr(main, "_current_user_id", lambda request: uid)
    r = post(client)
    assert r.status_code == 500 and "código" in r.json()["detail"]  # el error real sigue siendo el visible
    assert "No se pudo devolver una reserva" in caplog.text


# ===========================================================================
# Concurrencia a través de HTTP
# ===========================================================================
def _slow_engine(monkeypatch, delay=0.25):
    def engine(data, options=None):
        time.sleep(delay)  # mantiene las peticiones solapadas
        return {"markdown": "# ok"}

    monkeypatch.setattr(main, "convert_pdf", engine)


def _fire(count):
    barrier = threading.Barrier(count)
    responses = [None] * count

    def worker(i):
        c = new_client()
        barrier.wait(timeout=30)
        responses[i] = post(c)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert all(r is not None for r in responses)
    return responses


def test_simultaneous_requests_cannot_exceed_the_daily_quota(service, as_user, monkeypatch):
    monkeypatch.setattr(main, "CONVERT_SLOTS", threading.BoundedSemaphore(8))  # que de verdad se solapen
    _slow_engine(monkeypatch)
    statuses = sorted(r.status_code for r in _fire(8))
    assert statuses == [200] * 3 + [429] * 5
    assert usage(service, as_user) == 3


def test_simultaneous_failures_return_every_reservation(service, as_user, monkeypatch):
    monkeypatch.setattr(main, "CONVERT_SLOTS", threading.BoundedSemaphore(8))

    def engine(data, options=None):
        time.sleep(0.25)
        raise RuntimeError("fallo")

    monkeypatch.setattr(main, "convert_pdf", engine)
    statuses = sorted(r.status_code for r in _fire(8))
    assert statuses.count(500) >= 3  # al menos las 3 que cupieron; las demás pudieron entrar al ser devueltas
    assert 429 not in statuses or statuses.count(429) <= 5
    assert usage(service, as_user) == 0  # ninguna conversión fallida queda contada
