# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Base de datos y cuotas (sin HTTP; los tests de la API están en test_quotas_api.py y test_auth.py).

Las pruebas que reciben `store` se ejecutan dos veces: contra MemoryStore y contra PostgreSQL real
(solo si TEST_DATABASE_URL apunta a una base desechable; si no, esas ejecuciones se omiten).
"""

from __future__ import annotations

import datetime as dt
import logging
import threading

import pytest

import db
from db import MemoryStore, PostgresStore, QuotaStoreError
from quotas import CODE_HEADER, QuotaDenied, QuotaService, Reservation, build_quota_service
from tests.conftest import TEST_DATABASE_URL, truncate_postgres
from tests.helpers import FakeClock

DAY = dt.date(2026, 10, 18)
NEXT_DAY = DAY + dt.timedelta(days=1)


def run_concurrently(count, work):
    """Lanza `count` hilos a la vez (barrera) y devuelve el resultado de cada uno."""
    barrier = threading.Barrier(count)
    results = [None] * count

    def worker(index):
        barrier.wait(timeout=30)
        try:
            results[index] = work(index)
        except BaseException as exc:  # noqa: BLE001 - se devuelve para que el test lo vea
            results[index] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert all(not t.is_alive() for t in threads), "algún hilo no terminó"
    errors = [r for r in results if isinstance(r, BaseException)]
    assert not errors, f"excepciones en hilos: {errors[:3]}"
    return results


def make_user(store, n=1):
    return store.upsert_user(f"google-sub-{n}", f"user{n}@example.com", f"Usuario {n}")


def make_service(store, clock=None, **kwargs):
    clock = clock or FakeClock(dt.datetime(2026, 10, 18, 12, 0, tzinfo=dt.timezone.utc))
    return QuotaService(store, clock=clock, **kwargs), clock


# ===========================================================================
# Contrato del almacenamiento
# ===========================================================================
def test_upsert_user_is_idempotent(store):
    first = store.upsert_user("g-1", "a@example.com", "A")
    again = store.upsert_user("g-1", "nuevo@example.com", "A2")
    other = store.upsert_user("g-2", "b@example.com", None)
    assert first == again
    assert other != first


def test_reserve_user_counts_up_to_the_limit(store):
    uid = make_user(store)
    assert [store.reserve_user(uid, DAY, 3) for _ in range(3)] == [1, 2, 3]
    assert store.reserve_user(uid, DAY, 3) is None
    assert store.reserve_user(uid, DAY, 3) is None  # y sigue sin pasar
    assert store.usage_user(uid, DAY) == 3


@pytest.mark.parametrize("limit", [0, -1])
def test_non_positive_limit_never_reserves(store, limit):
    uid = make_user(store)
    assert store.reserve_user(uid, DAY, limit) is None
    assert store.usage_user(uid, DAY) == 0


def test_users_and_days_are_independent(store):
    a, b = make_user(store, 1), make_user(store, 2)
    for _ in range(3):
        assert store.reserve_user(a, DAY, 3)
    assert store.reserve_user(a, DAY, 3) is None
    assert store.reserve_user(b, DAY, 3) == 1  # otro usuario
    assert store.reserve_user(a, NEXT_DAY, 3) == 1  # mismo usuario, otro día (reinicio)


def test_unknown_and_inactive_users_cannot_reserve(store):
    uid = make_user(store)
    assert store.reserve_user(uid + 1000, DAY, 3) is None
    assert not store.user_is_active(uid + 1000)
    store.set_user_active(uid, False)
    assert not store.user_is_active(uid)
    assert store.reserve_user(uid, DAY, 3) is None
    store.set_user_active(uid, True)
    assert store.reserve_user(uid, DAY, 3) == 1


def test_release_user_refunds_and_never_goes_negative(store):
    uid = make_user(store)
    store.release_user(uid, DAY)  # sin reservas previas: no hace nada
    assert store.usage_user(uid, DAY) == 0
    store.reserve_user(uid, DAY, 3)
    store.release_user(uid, DAY)
    store.release_user(uid, DAY)
    assert store.usage_user(uid, DAY) == 0
    assert store.reserve_user(uid, DAY, 3) == 1


def test_release_user_applies_to_the_given_day_only(store):
    uid = make_user(store)
    store.reserve_user(uid, DAY, 3)
    store.reserve_user(uid, NEXT_DAY, 3)
    store.release_user(uid, DAY)
    assert store.usage_user(uid, DAY) == 0
    assert store.usage_user(uid, NEXT_DAY) == 1


def test_purge_removes_only_old_usage(store):
    uid = make_user(store)
    old = DAY - dt.timedelta(days=500)
    store.reserve_user(uid, old, 3)
    store.reserve_user(uid, DAY, 3)
    store.purge(DAY - dt.timedelta(days=400))
    assert store.usage_user(uid, old) == 0
    assert store.usage_user(uid, DAY) == 1
    assert store.user_is_active(uid)  # los usuarios no se tocan


# ===========================================================================
# Concurrencia: nunca se supera el límite
# ===========================================================================
def test_concurrent_reservations_never_exceed_the_limit(store):
    uid = make_user(store)
    results = run_concurrently(40, lambda i: store.reserve_user(uid, DAY, 3))
    assert sorted(r for r in results if r is not None) == [1, 2, 3]
    assert store.usage_user(uid, DAY) == 3


def test_concurrent_reservations_for_two_users_are_independent(store):
    a, b = make_user(store, 1), make_user(store, 2)
    results = run_concurrently(40, lambda i: (i % 2, store.reserve_user([a, b][i % 2], DAY, 3)))
    for who in (0, 1):
        assert sum(1 for w, r in results if w == who and r is not None) == 3
    assert store.usage_user(a, DAY) == 3 and store.usage_user(b, DAY) == 3


def test_concurrent_reserve_and_release_keep_the_counter_exact(store):
    uid = make_user(store)

    def work(i):
        used = store.reserve_user(uid, DAY, 5)
        if used is not None and i % 2 == 0:
            store.release_user(uid, DAY)
        return used is not None, used is not None and i % 2 == 0

    results = run_concurrently(40, work)
    granted = sum(1 for g, _ in results if g)
    refunded = sum(1 for _, r in results if r)
    assert store.usage_user(uid, DAY) == granted - refunded
    assert 0 <= store.usage_user(uid, DAY) <= 5


# ===========================================================================
# QuotaService: reglas de negocio
# ===========================================================================
def test_user_gets_three_per_day_then_429_with_retry_after(store):
    service, _ = make_service(store)  # 12:00 UTC
    uid = make_user(store)
    for _ in range(3):
        service.reserve_user(uid)
    with pytest.raises(QuotaDenied) as denied:
        service.reserve_user(uid)
    assert denied.value.status == 429
    assert denied.value.code == "daily_quota_exhausted"
    assert denied.value.retry_after == 12 * 3600
    assert denied.value.headers() == {CODE_HEADER: "daily_quota_exhausted", "Retry-After": str(12 * 3600)}
    assert "3 conversiones" in denied.value.message and "00:00 UTC" in denied.value.message


def test_quota_resets_at_midnight_utc(store):
    service, clock = make_service(store)
    uid = make_user(store)
    clock.set(2026, 10, 18, 23, 59, 30)
    for _ in range(3):
        service.reserve_user(uid)
    with pytest.raises(QuotaDenied) as denied:
        service.reserve_user(uid)
    assert denied.value.retry_after == 30

    clock.set(2026, 10, 18, 23, 59, 59)
    with pytest.raises(QuotaDenied):
        service.reserve_user(uid)  # un segundo antes de medianoche aún no se reinicia

    clock.set(2026, 10, 19, 0, 0, 0)
    service.reserve_user(uid)
    assert service.user_status(uid)["used"] == 1
    assert service.user_status(uid)["remaining"] == 2
    for _ in range(2):
        service.reserve_user(uid)
    with pytest.raises(QuotaDenied):
        service.reserve_user(uid)


def test_day_boundary_is_utc_whatever_the_timezone_of_the_clock(store):
    service, clock = make_service(store)
    uid = make_user(store)
    plus_two = dt.timezone(dt.timedelta(hours=2))
    clock.current = dt.datetime(2026, 10, 19, 1, 30, tzinfo=plus_two)  # = 18 oct 23:30 UTC
    assert service.today() == dt.date(2026, 10, 18)
    assert service.seconds_until_reset() == 30 * 60
    service.reserve_user(uid)
    assert store.usage_user(uid, dt.date(2026, 10, 18)) == 1
    clock.current = dt.datetime(2026, 10, 18, 23, 30)  # sin zona: se interpreta como UTC
    assert service.today() == dt.date(2026, 10, 18)


def test_released_reservation_can_be_used_again(store):
    service, _ = make_service(store)
    uid = make_user(store)
    reservations = [service.reserve_user(uid) for _ in range(3)]
    assert service.release(reservations[0]) is True
    service.reserve_user(uid)  # la devuelta vuelve a estar disponible
    with pytest.raises(QuotaDenied):
        service.reserve_user(uid)


def test_release_through_the_service_is_idempotent(store):
    service, _ = make_service(store)
    uid = make_user(store)
    first, _second = service.reserve_user(uid), service.reserve_user(uid)
    service.release(first)
    service.release(first)  # el doble reembolso no puede regalar una conversión extra
    assert store.usage_user(uid, DAY) == 1


def test_release_after_midnight_refunds_the_day_it_was_reserved(store):
    service, clock = make_service(store)
    uid = make_user(store)
    clock.set(2026, 10, 18, 23, 59, 59)
    late = service.reserve_user(uid)
    clock.set(2026, 10, 19, 0, 0, 5)
    service.reserve_user(uid)  # ya cuenta para el día siguiente
    service.release(late)
    assert store.usage_user(uid, DAY) == 0
    assert store.usage_user(uid, NEXT_DAY) == 1


def test_disabled_and_unknown_users_get_403_not_429(store):
    service, _ = make_service(store)
    uid = make_user(store)
    store.set_user_active(uid, False)
    for user_id in (uid, uid + 999):
        with pytest.raises(QuotaDenied) as denied:
            service.reserve_user(user_id)
        assert denied.value.status == 403
        assert denied.value.code == "account_disabled"
        assert "Retry-After" not in denied.value.headers()


def test_zero_daily_limit_denies_everyone(store):
    service, _ = make_service(store, daily_limit=0)
    with pytest.raises(QuotaDenied):
        service.reserve_user(make_user(store))


# ===========================================================================
# Errores
# ===========================================================================
class BrokenStore(MemoryStore):
    """Todas las operaciones de cuota fallan como si la base de datos estuviera caída."""

    def reserve_user(self, *a, **k):
        raise QuotaStoreError("base de datos caída")

    def release_user(self, *a, **k):
        raise QuotaStoreError("base de datos caída")


def test_store_failure_propagates_so_the_caller_can_fail_closed():
    service, _ = make_service(BrokenStore())
    with pytest.raises(QuotaStoreError):
        service.reserve_user(1)


def test_release_never_raises_and_reports_the_failure(caplog):
    service, _ = make_service(BrokenStore())
    with caplog.at_level(logging.ERROR, logger="markpdf.quotas"):
        assert service.release(Reservation(1, DAY)) is False
    assert "No se pudo devolver una reserva" in caplog.text


def test_postgres_unreachable_raises_quota_store_error():
    pytest.importorskip("psycopg")
    unreachable = PostgresStore("postgresql://nobody:nopass@127.0.0.1:1/nodb")
    with pytest.raises(QuotaStoreError):
        unreachable.reserve_user(1, DAY, 3)
    with pytest.raises(QuotaStoreError):
        unreachable.upsert_user("sub", "a@example.com")
    with pytest.raises(QuotaStoreError):
        unreachable.init()


def test_postgres_error_message_does_not_leak_the_password():
    pytest.importorskip("psycopg")
    unreachable = PostgresStore("postgresql://nobody:SUPERSECRET@127.0.0.1:1/nodb")
    with pytest.raises(QuotaStoreError) as error:
        unreachable.reserve_user(1, DAY, 3)
    assert "SUPERSECRET" not in str(error.value)
    assert "SUPERSECRET" not in repr(error.value.__cause__ or "")


def test_postgres_store_requires_a_url():
    with pytest.raises(ValueError):
        PostgresStore("")


# ===========================================================================
# build_quota_service: configuración al arrancar
# ===========================================================================
def test_build_requires_a_database_in_production():
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        build_quota_service(database_url="", daily_limit=3, production=True)


def test_build_uses_memory_in_development_and_warns(caplog):
    with caplog.at_level(logging.WARNING, logger="markpdf.quotas"):
        service = build_quota_service(database_url="", daily_limit=3, production=False)
    assert isinstance(service.store, MemoryStore)
    assert service.daily_limit == 3
    assert "memoria" in caplog.text


# ===========================================================================
# Migraciones (sin base de datos y con PostgreSQL real)
# ===========================================================================
def test_bundled_migrations_are_numbered_and_ordered():
    migrations = db.load_migrations()
    assert [v for v, _, _ in migrations] == list(range(1, len(migrations) + 1))
    assert "daily_usage" in migrations[0][2] and "google_sub" in migrations[0][2]
    assert "anon" not in migrations[0][2]  # ya no hay prueba anónima


def test_load_migrations_rejects_bad_names_and_duplicates(tmp_path):
    (tmp_path / "001_a.sql").write_text("SELECT 1;")
    (tmp_path / "1_b.sql").write_text("SELECT 2;")
    with pytest.raises(RuntimeError, match="mismo número"):
        db.load_migrations(tmp_path)
    (tmp_path / "1_b.sql").unlink()
    (tmp_path / "sin_numero.sql").write_text("SELECT 3;")
    with pytest.raises(RuntimeError, match="no válido"):
        db.load_migrations(tmp_path)


@pytest.fixture
def pg_url():
    if not TEST_DATABASE_URL:
        pytest.skip("TEST_DATABASE_URL no definida")
    pytest.importorskip("psycopg")
    PostgresStore(TEST_DATABASE_URL).init()
    truncate_postgres(TEST_DATABASE_URL)
    return TEST_DATABASE_URL


def test_postgres_init_is_idempotent_and_records_versions(pg_url):
    import psycopg

    store = PostgresStore(pg_url)
    store.init()
    store.init()
    with psycopg.connect(pg_url) as conn:
        versions = [r[0] for r in conn.execute("SELECT version FROM schema_migrations ORDER BY version")]
        tables = {
            r[0]
            for r in conn.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
        }
    assert versions == [v for v, _, _ in db.load_migrations()]
    assert {"users", "daily_usage", "schema_migrations"} <= tables


def test_postgres_simultaneous_init_does_not_fail(pg_url):
    store = PostgresStore(pg_url)
    run_concurrently(5, lambda i: store.init())


def test_postgres_check_constraint_forbids_negative_counters(pg_url):
    import psycopg

    uid = PostgresStore(pg_url).upsert_user("sub", "a@example.com")
    with psycopg.connect(pg_url) as conn:
        with pytest.raises(psycopg.errors.CheckViolation):
            conn.execute("INSERT INTO daily_usage (user_id, day, used) VALUES (%s, %s, -1)", (uid, DAY))


def test_postgres_deleting_a_user_removes_their_usage(pg_url):
    import psycopg

    store = PostgresStore(pg_url)
    uid = store.upsert_user("sub", "a@example.com")
    store.reserve_user(uid, DAY, 3)
    with psycopg.connect(pg_url) as conn:
        conn.execute("DELETE FROM users WHERE id = %s", (uid,))
    assert store.usage_user(uid, DAY) == 0


def test_postgres_upsert_updates_profile_and_keeps_the_id(pg_url):
    import psycopg

    store = PostgresStore(pg_url)
    uid = store.upsert_user("sub", "viejo@example.com", "Viejo")
    assert store.upsert_user("sub", "nuevo@example.com", "Nuevo") == uid
    with psycopg.connect(pg_url) as conn:
        row = conn.execute("SELECT email, name FROM users WHERE id = %s", (uid,)).fetchone()
    assert row == ("nuevo@example.com", "Nuevo")
