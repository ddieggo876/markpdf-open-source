# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Almacenamiento de usuarios y cuotas.

Dos implementaciones con el mismo contrato:

* ``MemoryStore``   – en memoria, para desarrollo y tests. Se pierde al reiniciar: NO usar en producción.
* ``PostgresStore`` – PostgreSQL persistente (psycopg 3, SQL plano, migraciones numeradas en ``migrations/``).

Garantía clave: reservar cuota es UNA operación atómica («comprobar y sumar» a la vez). Así, aunque lleguen
muchas peticiones simultáneas, nadie puede superar el límite. En PostgreSQL lo da el bloqueo de fila de
``INSERT ... ON CONFLICT DO UPDATE ... WHERE used < límite``; en memoria, un ``threading.Lock``.

``psycopg`` solo se importa cuando se usa ``PostgresStore``: con la autenticación desactivada (por defecto)
la aplicación no lo necesita ni lo carga.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("markpdf.db")

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"
_MIGRATION_LOCK_ID = 7_302_024_001  # clave del bloqueo consultivo: evita migraciones simultáneas entre procesos
_CONNECT_TIMEOUT_SECONDS = 5
_STATEMENT_TIMEOUT_MS = 5000


class QuotaStoreError(RuntimeError):
    """El almacenamiento no está disponible o falló (la aplicación debe fallar cerrada: sin cuota no se convierte)."""


# --------------------------------------------------------------------------------------------------
# Memoria (desarrollo y tests)
# --------------------------------------------------------------------------------------------------
class MemoryStore:
    """Misma semántica que PostgresStore, con un único candado que hace atómica cada operación."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_id = 1
        self._users: dict[int, dict[str, Any]] = {}
        self._by_sub: dict[str, int] = {}
        self._usage: dict[tuple[int, dt.date], int] = {}

    def init(self) -> None:
        """Sin migraciones: no hace nada."""

    def close(self) -> None:
        """Sin recursos que liberar."""

    # -- usuarios
    def upsert_user(self, google_sub: str, email: str, name: Optional[str] = None) -> int:
        with self._lock:
            user_id = self._by_sub.get(google_sub)
            if user_id is None:
                user_id = self._next_id
                self._next_id += 1
                self._by_sub[google_sub] = user_id
                self._users[user_id] = {"email": email, "name": name, "active": True}
            else:
                self._users[user_id].update(email=email, name=name)
            return user_id

    def set_user_active(self, user_id: int, active: bool) -> None:
        with self._lock:
            if user_id in self._users:
                self._users[user_id]["active"] = bool(active)

    def user_is_active(self, user_id: int) -> bool:
        with self._lock:
            user = self._users.get(user_id)
            return bool(user and user["active"])

    # -- cuota
    def reserve_user(self, user_id: int, day: dt.date, limit: int) -> Optional[int]:
        """Suma 1 si hay cupo. Devuelve el nuevo total, o None si el límite está agotado / usuario no válido."""
        if limit <= 0:
            return None
        with self._lock:
            user = self._users.get(user_id)
            if not user or not user["active"]:
                return None
            used = self._usage.get((user_id, day), 0)
            if used >= limit:
                return None
            self._usage[(user_id, day)] = used + 1
            return used + 1

    def release_user(self, user_id: int, day: dt.date) -> None:
        with self._lock:
            used = self._usage.get((user_id, day), 0)
            if used > 0:
                self._usage[(user_id, day)] = used - 1

    def usage_user(self, user_id: int, day: dt.date) -> int:
        with self._lock:
            return self._usage.get((user_id, day), 0)

    def purge(self, before_day: dt.date) -> None:
        """Borra datos de uso anteriores a `before_day` (no toca a los usuarios)."""
        with self._lock:
            self._usage = {k: v for k, v in self._usage.items() if k[1] >= before_day}


# --------------------------------------------------------------------------------------------------
# PostgreSQL
# --------------------------------------------------------------------------------------------------
def load_migrations(directory: Path = MIGRATIONS_DIR) -> list[tuple[int, str, str]]:
    """Lee ``NNN_nombre.sql`` ordenados por número: [(versión, nombre_de_archivo, sql)]."""
    found: list[tuple[int, str, str]] = []
    for path in sorted(directory.glob("*.sql")):
        match = re.match(r"^(\d+)_", path.name)
        if not match:
            raise RuntimeError(f"Nombre de migración no válido (debe empezar por un número y «_»): {path.name}")
        found.append((int(match.group(1)), path.name, path.read_text(encoding="utf-8")))
    versions = [v for v, _, _ in found]
    if len(versions) != len(set(versions)):
        raise RuntimeError("Hay dos migraciones con el mismo número.")
    return sorted(found)


class PostgresStore:
    """
    PostgreSQL con una conexión corta por operación (sin pool): cada conversión hace 1-2 operaciones y la
    aplicación convierte de una en una, así que la sobrecarga es irrelevante y se evitan conexiones caducadas.
    """

    def __init__(self, dsn: str) -> None:
        if not dsn:
            raise ValueError("Falta la URL de la base de datos (DATABASE_URL).")
        self._dsn = dsn

    # -- infraestructura
    @staticmethod
    def _psycopg():
        try:
            import psycopg  # noqa: PLC0415  (import perezoso a propósito)
        except ImportError as exc:
            raise RuntimeError(
                "Falta la dependencia «psycopg»: instala «psycopg[binary]» para usar PostgreSQL."
            ) from exc
        return psycopg

    def _connect(self):
        psycopg = self._psycopg()
        try:
            return psycopg.connect(
                self._dsn,
                connect_timeout=_CONNECT_TIMEOUT_SECONDS,
                options=f"-c statement_timeout={_STATEMENT_TIMEOUT_MS}",
            )
        except psycopg.Error as exc:
            raise QuotaStoreError(f"No se pudo conectar con la base de datos ({type(exc).__name__}).") from exc

    def _run(self, work):
        """Ejecuta `work(cursor)` en una transacción (commit si va bien, rollback si falla)."""
        psycopg = self._psycopg()
        try:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    return work(cur)
        except psycopg.Error as exc:
            raise QuotaStoreError(f"Error de base de datos ({type(exc).__name__}).") from exc

    def close(self) -> None:
        """Sin recursos persistentes: las conexiones se cierran tras cada operación."""

    # -- migraciones
    def init(self) -> None:
        """Aplica las migraciones pendientes (idempotente y segura ante arranques simultáneos)."""
        psycopg = self._psycopg()
        migrations = load_migrations()
        try:
            with self._connect() as conn:
                conn.autocommit = True
                conn.execute("SELECT pg_advisory_lock(%s)", (_MIGRATION_LOCK_ID,))
                try:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS schema_migrations ("
                        " version INTEGER PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
                    )
                    applied = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
                    for version, name, sql in migrations:
                        if version in applied:
                            continue
                        with conn.transaction():
                            conn.execute(sql)
                            conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))
                        log.info("Migración %s aplicada.", name)
                finally:
                    conn.execute("SELECT pg_advisory_unlock(%s)", (_MIGRATION_LOCK_ID,))
        except psycopg.Error as exc:
            raise QuotaStoreError(f"Error aplicando migraciones ({type(exc).__name__}).") from exc

    # -- usuarios
    def upsert_user(self, google_sub: str, email: str, name: Optional[str] = None) -> int:
        def work(cur):
            cur.execute(
                "INSERT INTO users (google_sub, email, name, last_login_at) VALUES (%s, %s, %s, now()) "
                "ON CONFLICT (google_sub) DO UPDATE "
                "SET email = EXCLUDED.email, name = EXCLUDED.name, last_login_at = now() "
                "RETURNING id",
                (google_sub, email, name),
            )
            return cur.fetchone()[0]

        return self._run(work)

    def set_user_active(self, user_id: int, active: bool) -> None:
        self._run(lambda cur: cur.execute("UPDATE users SET is_active = %s WHERE id = %s", (bool(active), user_id)))

    def user_is_active(self, user_id: int) -> bool:
        def work(cur):
            cur.execute("SELECT is_active FROM users WHERE id = %s", (user_id,))
            row = cur.fetchone()
            return bool(row and row[0])

        return self._run(work)

    # -- cuota
    def reserve_user(self, user_id: int, day: dt.date, limit: int) -> Optional[int]:
        if limit <= 0:
            return None

        def work(cur):
            # Una sola sentencia: crea la fila del día o suma 1 SOLO si used < límite. Dos peticiones simultáneas
            # sobre la misma fila se serializan por su bloqueo, así que el límite nunca se supera.
            # El SELECT sobre users impide reservar para usuarios inexistentes o desactivados.
            cur.execute(
                "INSERT INTO daily_usage (user_id, day, used) "
                "SELECT u.id, %(day)s, 1 FROM users u WHERE u.id = %(uid)s AND u.is_active "
                "ON CONFLICT (user_id, day) DO UPDATE SET used = daily_usage.used + 1 "
                "WHERE daily_usage.used < %(limit)s "
                "RETURNING used",
                {"day": day, "uid": user_id, "limit": limit},
            )
            row = cur.fetchone()
            return row[0] if row else None

        return self._run(work)

    def release_user(self, user_id: int, day: dt.date) -> None:
        self._run(
            lambda cur: cur.execute(
                "UPDATE daily_usage SET used = used - 1 WHERE user_id = %s AND day = %s AND used > 0",
                (user_id, day),
            )
        )

    def usage_user(self, user_id: int, day: dt.date) -> int:
        def work(cur):
            cur.execute("SELECT used FROM daily_usage WHERE user_id = %s AND day = %s", (user_id, day))
            row = cur.fetchone()
            return row[0] if row else 0

        return self._run(work)

    def purge(self, before_day: dt.date) -> None:
        self._run(lambda cur: cur.execute("DELETE FROM daily_usage WHERE day < %s", (before_day,)))
