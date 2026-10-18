# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Cuotas de conversión.

Reglas:

* Solo convierten los usuarios registrados (Google): ``daily_limit`` (3) conversiones gratuitas por día natural UTC;
  se reinician a las 00:00 UTC.
* Solo cuentan las conversiones completadas con éxito: la cuota se RESERVA (atómicamente) antes de convertir y se
  DEVUELVE si la conversión no termina bien. Así, peticiones simultáneas no pueden superar el límite y un fallo
  del servidor no le cuesta nada al usuario.
* Si no se puede comprobar la cuota (base de datos caída) se falla cerrado: no se convierte.

Este módulo no sabe nada de HTTP ni de FastAPI: ``main.py`` traduce ``QuotaDenied`` a respuestas.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Callable, Optional

from db import MemoryStore, PostgresStore, QuotaStoreError

log = logging.getLogger("markpdf.quotas")

RETENTION_DAYS = 400  # el uso más antiguo se borra al arrancar

CODE_HEADER = "X-MarkPDF-Code"  # código legible por máquina; el texto para la persona va en `detail`

Clock = Callable[[], dt.datetime]


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class QuotaDenied(Exception):
    """La petición no puede consumir cuota. `status` es el código HTTP que corresponde."""

    def __init__(self, status: int, code: str, message: str, retry_after: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after = retry_after

    def headers(self) -> dict[str, str]:
        headers = {CODE_HEADER: self.code}
        if self.retry_after is not None:
            headers["Retry-After"] = str(self.retry_after)
        return headers


@dataclass
class Reservation:
    """Una unidad de cuota reservada. Si la conversión falla hay que devolverla con `QuotaService.release`."""

    user_id: int
    day: dt.date  # día UTC en que se reservó (la devolución se aplica a ESE día aunque pase la medianoche)
    released: bool = False


class QuotaService:
    def __init__(self, store, *, daily_limit: int = 3, clock: Clock = _utc_now) -> None:
        self.store = store
        self.daily_limit = daily_limit
        self._clock = clock

    # -- tiempo (UTC)
    def now(self) -> dt.datetime:
        now = self._clock()
        return now.astimezone(dt.timezone.utc) if now.tzinfo else now.replace(tzinfo=dt.timezone.utc)

    def today(self) -> dt.date:
        return self.now().date()

    def seconds_until_reset(self) -> int:
        now = self.now()
        midnight = dt.datetime.combine(now.date() + dt.timedelta(days=1), dt.time(), tzinfo=dt.timezone.utc)
        return max(1, int((midnight - now).total_seconds() + 0.999))

    # -- reservas
    def reserve_user(self, user_id: int) -> Reservation:
        """Reserva 1 conversión del día. Lanza QuotaDenied (403/429) o QuotaStoreError (base de datos caída)."""
        day = self.today()
        used = self.store.reserve_user(user_id, day, self.daily_limit)
        if used is None:
            if not self.store.user_is_active(user_id):
                raise QuotaDenied(403, "account_disabled", "Tu cuenta no está activa.")
            raise QuotaDenied(
                429,
                "daily_quota_exhausted",
                f"Has usado tus {self.daily_limit} conversiones gratuitas de hoy. "
                "Se renuevan a las 00:00 UTC.",
                retry_after=self.seconds_until_reset(),
            )
        return Reservation(user_id, day)

    def release(self, reservation: Reservation) -> bool:
        """
        Devuelve una reserva (la conversión falló). Idempotente. Nunca lanza: si la base de datos falla se registra
        y se devuelve False (el usuario pierde una unidad, pero el error original de la conversión no se oculta).
        """
        if reservation.released:
            return True
        reservation.released = True
        try:
            self.store.release_user(reservation.user_id, reservation.day)
        except QuotaStoreError as exc:
            log.error("No se pudo devolver una reserva de cuota: %s", exc)
            return False
        return True

    def user_status(self, user_id: int) -> dict[str, int]:
        used = self.store.usage_user(user_id, self.today())
        return {
            "limit": self.daily_limit,
            "used": used,
            "remaining": max(0, self.daily_limit - used),
            "resets_in_seconds": self.seconds_until_reset(),
        }

    def purge_old(self) -> None:
        self.store.purge(self.today() - dt.timedelta(days=RETENTION_DAYS))


def build_quota_service(*, database_url: str, daily_limit: int, production: bool) -> QuotaService:
    """Crea el servicio con la configuración del entorno. Falla al arrancar (en alto) si está mal configurado."""
    if database_url:
        store = PostgresStore(database_url)
    elif production:
        raise RuntimeError(
            "DATABASE_URL es obligatoria en producción con MARKPDF_AUTH_ENABLED=true: "
            "sin ella los usuarios y las cuotas se perderían en cada reinicio."
        )
    else:
        log.warning("Sin DATABASE_URL: usuarios y cuotas se guardan en memoria y se perderán al reiniciar (solo desarrollo).")
        store = MemoryStore()
    store.init()
    service = QuotaService(store, daily_limit=daily_limit)
    try:
        service.purge_old()
    except QuotaStoreError as exc:  # la limpieza es opcional: no impide arrancar
        log.warning("No se pudo limpiar el uso antiguo: %s", exc)
    return service
