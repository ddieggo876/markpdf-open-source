# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Diagnóstico de rendimiento de POST /convert: tiempo y memoria por etapa, escritos en el log.

Solo mide; no cambia el resultado de la conversión. Por diseño NO registra contenido del PDF, nombres de archivo,
datos de usuario ni secretos: solo nombres de etapa fijos y números (segundos, MB, páginas, tamaños).

Uso (todo es inocuo si no hay una medición activa, p. ej. al llamar a `convert_pdf` desde los tests):

    prof = perf.start()                  # al empezar la petición
    with perf.stage("layout"): ...       # alrededor de cada etapa (pueden anidarse: el tiempo se resta al padre)
    perf.note(pages=12)                  # datos numéricos de contexto
    perf.finish(prof, status="ok")       # escribe el resumen en el log

Se desactiva con MARKPDF_PERF_LOG=false. El log sale por stderr (Render lo recoge en "Logs").
"""

from __future__ import annotations

import contextlib
import contextvars
import logging
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

try:
    import psutil
except ImportError:  # pragma: no cover - psutil va en requirements.txt
    psutil = None  # type: ignore[assignment]

try:
    import resource  # solo Unix
except ImportError:  # pragma: no cover - Windows
    resource = None  # type: ignore[assignment]

MB = 1024 * 1024
_PROCESS_START = time.monotonic()
_ENABLED = os.environ.get("MARKPDF_PERF_LOG", "").strip().lower() not in ("0", "false", "no", "off")

logger = logging.getLogger("markpdf.perf")
if not logger.handlers:  # uvicorn no configura el logger raíz: sin esto los INFO se perderían
    _handler = logging.StreamHandler()  # stderr
    _handler.setFormatter(logging.Formatter("%(levelname)s markpdf.perf %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

# Agrupación de etapas en las cuatro fases que interesa distinguir (el resto, como "quota", va aparte).
GROUPS: dict[str, tuple[str, ...]] = {
    "read": ("read_upload", "open_pdf"),
    "layout": ("layout", "ocr"),
    "convert": ("forms", "assemble", "margins", "source_text", "clean", "validate"),
    "response": ("metrics",),
}

_conversions_started = 0
_conversions_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Lecturas del sistema (todas "best effort": si no existen, devuelven None)
# ---------------------------------------------------------------------------
def _read_text(path: str) -> Optional[str]:
    try:
        return Path(path).read_text().strip()
    except (OSError, ValueError):
        return None


def _rss() -> Optional[int]:
    if psutil is None:
        return None
    try:
        return psutil.Process().memory_info().rss
    except Exception:  # pragma: no cover
        return None


def _peak_rss() -> Optional[int]:
    """Pico de RSS del proceso desde que arrancó (no por etapa). Solo Unix."""
    if resource is None:
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)  # Linux: KB; macOS: bytes


def _cgroup_throttled_us() -> Optional[int]:
    """Microsegundos acumulados en que el cgroup estuvo frenado por su límite de CPU (cgroup v2, luego v1)."""
    text = _read_text("/sys/fs/cgroup/cpu.stat")
    if text:
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key == "throttled_usec" and value.isdigit():
                return int(value)
    v1 = _read_text("/sys/fs/cgroup/cpu/cpu.stat")
    if v1:
        for line in v1.splitlines():
            key, _, value = line.partition(" ")
            if key == "throttled_time" and value.isdigit():
                return int(value) // 1000  # ns -> µs
    return None


def _cgroup_memory() -> tuple[Optional[int], Optional[int]]:
    """(uso actual, límite) de memoria del contenedor, en bytes."""

    def _num(raw: Optional[str]) -> Optional[int]:
        return int(raw) if raw and raw.isdigit() and int(raw) < 1 << 60 else None  # "max" / sin límite -> None

    cur = _num(_read_text("/sys/fs/cgroup/memory.current"))
    lim = _num(_read_text("/sys/fs/cgroup/memory.max"))
    if cur is None and lim is None:  # cgroup v1
        cur = _num(_read_text("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        lim = _num(_read_text("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
    return cur, lim


def _cgroup_oom_kills() -> Optional[int]:
    text = _read_text("/sys/fs/cgroup/memory.events")
    if text:
        for line in text.splitlines():
            key, _, value = line.partition(" ")
            if key == "oom_kill" and value.isdigit():
                return int(value)
    return None


def _cpu_limit() -> Optional[float]:
    """Límite de CPU del contenedor en núcleos (cgroup v2 `cpu.max`); None si no hay límite o no se puede leer."""
    text = _read_text("/sys/fs/cgroup/cpu.max")
    if text:
        quota, _, period = text.partition(" ")
        if quota.isdigit() and period.isdigit() and int(period) > 0:
            return int(quota) / int(period)
        return None
    quota_v1, period_v1 = _read_text("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), _read_text("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota_v1 and period_v1 and quota_v1.lstrip("-").isdigit() and period_v1.isdigit() and int(quota_v1) > 0:
        return int(quota_v1) / int(period_v1)
    return None


def _visible_cpus() -> Optional[int]:
    try:
        return len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        return os.cpu_count()


def _mb(value: Optional[int], signed: bool = False) -> str:
    if value is None:
        return "na"
    return f"{value / MB:+.0f}MB" if signed else f"{value / MB:.0f}MB"


class _Snap:
    __slots__ = ("wall", "cpu", "rss", "throttled_us")

    def __init__(self) -> None:
        self.wall = time.perf_counter()
        self.cpu = time.process_time()  # CPU de TODOS los hilos del proceso (incluye onnxruntime)
        self.rss = _rss()
        self.throttled_us = _cgroup_throttled_us()


# ---------------------------------------------------------------------------
# Medición de una petición
# ---------------------------------------------------------------------------
class Profile:
    def __init__(self) -> None:
        global _conversions_started
        with _conversions_lock:
            _conversions_started += 1
            self.conversion_number = _conversions_started  # 1 = primera conversión de este proceso (modelos en frío)
        self.request_id = uuid.uuid4().hex[:8]
        self.start = _Snap()
        self.stages: dict[str, dict[str, Any]] = {}
        self.notes: dict[str, Any] = {}
        self._stack: list[list[float]] = []  # por nivel: [wall, cpu, throttled_us] ya contados por etapas hijas

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        before = _Snap()
        children = [0.0, 0.0, 0.0, 0.0]
        self._stack.append(children)
        try:
            yield
        finally:
            after = _Snap()
            self._stack.pop()
            wall, cpu = after.wall - before.wall, after.cpu - before.cpu
            rss_growth = after.rss - before.rss if after.rss is not None and before.rss is not None else 0
            throttled = (
                (after.throttled_us - before.throttled_us) / 1e6
                if after.throttled_us is not None and before.throttled_us is not None
                else None
            )
            if self._stack:  # el padre no cuenta lo que ya midieron sus hijas
                parent = self._stack[-1]
                parent[0] += wall
                parent[1] += cpu
                parent[2] += throttled or 0.0
                parent[3] += rss_growth
            rec = self.stages.setdefault(
                name, {"wall": 0.0, "cpu": 0.0, "throttled": None, "rss": None, "rss_delta": 0, "peak": None}
            )
            rec["wall"] += wall - children[0]
            rec["cpu"] += cpu - children[1]
            if throttled is not None:
                rec["throttled"] = (rec["throttled"] or 0.0) + throttled - children[2]
            rec["rss"] = after.rss
            rec["rss_delta"] += rss_growth - children[3]
            rec["peak"] = _peak_rss()

    def _line(self, name: str, rec: dict[str, Any]) -> str:
        wall, cpu = max(rec["wall"], 0.0), max(rec["cpu"], 0.0)
        ratio = f"{cpu / wall:.2f}" if wall >= 0.05 else "na"
        throttled = "na" if rec["throttled"] is None else f"{max(rec['throttled'], 0.0):.2f}s"
        return (
            f"id={self.request_id} stage={name} wall={wall:.2f}s cpu={cpu:.2f}s cpu_ratio={ratio} "
            f"throttled={throttled} rss={_mb(rec['rss'])} rss_delta={_mb(rec['rss_delta'], True)} "
            f"peak={_mb(rec['peak'])}"
        )

    def report(self, status: str) -> None:
        end = _Snap()
        total_wall, total_cpu = end.wall - self.start.wall, end.cpu - self.start.cpu
        groups = {
            g: sum(max(self.stages[s]["wall"], 0.0) for s in names if s in self.stages) for g, names in GROUPS.items()
        }
        known = {s for names in GROUPS.values() for s in names}
        other = sum(max(r["wall"], 0.0) for s, r in self.stages.items() if s not in known)
        cg_cur, cg_lim = _cgroup_memory()
        cpu_limit = _cpu_limit()
        notes = " ".join(f"{k}={v}" for k, v in self.notes.items())
        throttled_total = (
            (end.throttled_us - self.start.throttled_us) / 1e6
            if end.throttled_us is not None and self.start.throttled_us is not None
            else None
        )
        logger.info(
            "id=%s done status=%s total=%.2fs total_cpu=%.2fs | read=%.2fs layout=%.2fs convert=%.2fs "
            "response=%.2fs other=%.2fs | throttled=%s | rss_end=%s peak=%s cgroup_mem=%s/%s oom_kills=%s | "
            "cpus_visible=%s cpu_limit=%s | conversion_n=%d cold=%s proc_uptime=%.0fs %s",
            self.request_id,
            status,
            total_wall,
            total_cpu,
            groups["read"],
            groups["layout"],
            groups["convert"],
            groups["response"],
            other,
            "na" if throttled_total is None else f"{throttled_total:.2f}s",
            _mb(end.rss),
            _mb(_peak_rss()),
            _mb(cg_cur),
            _mb(cg_lim),
            _cgroup_oom_kills(),
            _visible_cpus(),
            "na" if cpu_limit is None else f"{cpu_limit:.2f}",
            self.conversion_number,
            "yes" if self.conversion_number == 1 else "no",
            time.monotonic() - _PROCESS_START,
            notes,
        )
        for name, rec in self.stages.items():
            logger.info("%s", self._line(name, rec))


_current: contextvars.ContextVar[Optional[Profile]] = contextvars.ContextVar("markpdf_perf", default=None)


def start() -> Optional[Profile]:
    """Empieza a medir la petición actual (None si el diagnóstico está desactivado)."""
    if not _ENABLED:
        return None
    prof = Profile()
    _current.set(prof)
    return prof


def stage(name: str) -> contextlib.AbstractContextManager[None]:
    prof = _current.get()
    return prof.stage(name) if prof is not None else contextlib.nullcontext()


def note(**values: Any) -> None:
    """Datos de contexto NUMÉRICOS o de opciones fijas (páginas, tamaños, perfil). Nunca contenido ni nombres."""
    prof = _current.get()
    if prof is not None:
        prof.notes.update(values)


def finish(prof: Optional[Profile], status: str) -> None:
    """Escribe el resumen y las etapas. Nunca lanza: medir no debe romper una conversión."""
    if prof is None:
        return
    _current.set(None)
    try:
        prof.report(status)
    except Exception:  # pragma: no cover
        logger.exception("No se pudo escribir el diagnóstico de rendimiento")
