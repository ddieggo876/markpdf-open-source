# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
MarkPDF - Convierte PDFs en Markdown fiel, estructurado y eficiente en tokens, pensado como contexto para LLMs.

Pipeline (determinista, sin llamadas a ninguna IA ni servicio externo), en etapas claras:

  1. Validación del archivo (tipo, tamaño, páginas, contraseña, corrupción).
  2. Extracción de texto y estructura (`extraction.py`): pymupdf4llm + relectura línea a línea (`lines.py`),
     formularios AcroForm/casillas (`forms.py`).
  3. Detección y tratamiento de tablas (`tables.py`).
  4. Tratamiento de imágenes según `image_mode` (`visuals.py`).
  5. Limpieza configurable (`cleaning.py`, `margins.py`): cabeceras/pies/numeración, párrafos, duplicados.
  6. Normalización y validación del Markdown (`validation.py`).
  7. Métricas (`metrics.py`).
  8. Respuesta JSON.

Los tres perfiles ("faithful", "clean", "compact") son un único motor con políticas distintas: ver `options.py`.

Ejecutar:
    pip install -r requirements.txt
    uvicorn main:app --reload
y abrir http://127.0.0.1:8000
"""

from __future__ import annotations

import ort_threads

# Límite opcional de hilos de ONNX Runtime (MARKPDF_ORT_THREADS). Sin la variable no hace nada. Debe ir ANTES de
# importar pymupdf.layout / pymupdf4llm, que crean sus sesiones ONNX al importarse.
ort_threads.apply_from_env()

import contextlib
import functools
import io
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote

try:  # PyMuPDF >= 1.24.3
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # type: ignore

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import extraction
import metrics
import perf
import vision
from cleaning import clean_pages
from extraction import extract_pages
from db import QuotaStoreError
from forms import prepare_widgets
from lines import FULL_MARK, plain_page_text
from margins import analyze_margins, detect_margin_noise  # noqa: F401  (detect_margin_noise: compatibilidad)
from options import (
    DEFAULT_IMAGE_MODE,
    DEFAULT_MODE,
    IMAGE_MODES,
    MODES,
    PROFILE_LABELS,
    TABLE_FORMATS,
    ConvertOptions,
    OptionsError,
    normalize_image_mode,
    options_from_form,
)
from auth import (
    NO_STORE,
    SESSION_COOKIE,
    SESSION_MAX_AGE,
    build_oauth,
    build_router,
    session_user_id,
    validate_settings,
)
from quotas import (
    CODE_HEADER,
    QuotaDenied,
    QuotaService,
    Reservation,
    build_quota_service,
)
from security import (
    BodyLimitMiddleware,
    RateLimiter,
    SecurityHeadersMiddleware,
    build_security_headers,
    client_ip,
)
from validation import coverage_warning, extraction_coverage, validate_markdown

log = logging.getLogger("markpdf")


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Entero de una variable de entorno; si falta o no es válido se usa `default` (y se avisa en el log)."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Variable %s no válida (%r): se usa el valor por defecto %s.", name, raw, default)
        return default
    if value < minimum:
        log.warning("Variable %s=%s fuera de rango (mínimo %s): se usa %s.", name, value, minimum, default)
        return default
    return value


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in ("1", "true", "yes", "on", "si", "sí"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    return default


BASE_DIR = Path(__file__).resolve().parent
VERSION = "0.4.0"

# Entorno: con MARKPDF_ENV=production se ocultan /docs, /redoc y /openapi.json, se añade HSTS y la extracción
# de imágenes (.zip) queda desactivada salvo que MARKPDF_ENABLE_IMAGE_EXTRACT=true.
IS_PRODUCTION = os.environ.get("MARKPDF_ENV", "").strip().lower() == "production"

# Límites (todos configurables por variables de entorno).
MAX_UPLOAD_BYTES = _env_int("MARKPDF_MAX_UPLOAD_MB", 10) * 1024 * 1024  # tamaño máximo de cada PDF
MAX_PAGES = _env_int("MARKPDF_MAX_PAGES", 100)  # páginas máximas de cada PDF
MAX_DOWNLOAD_BYTES = _env_int("MARKPDF_MAX_DOWNLOAD_MB", 5) * 1024 * 1024  # texto máximo enviado a /download
MAX_CONCURRENT_CONVERSIONS = _env_int("MARKPDF_MAX_CONCURRENT", 1)  # conversiones simultáneas
RATE_LIMIT_PER_HOUR = _env_int("MARKPDF_RATE_LIMIT_PER_HOUR", 10, minimum=0)  # por IP y hora; 0 = sin límite
# Nº de proxies de confianza delante de la app (0 = ninguno: se ignora X-Forwarded-For). Ver security.client_ip.
TRUSTED_PROXY_HOPS = _env_int("MARKPDF_TRUSTED_PROXY_HOPS", 0, minimum=0)
ALLOW_IMAGE_EXTRACT = _env_flag("MARKPDF_ENABLE_IMAGE_EXTRACT", default=not IS_PRODUCTION)

# Cuentas (Google Login) y cuotas. Con MARKPDF_AUTH_ENABLED=false (por defecto) NADA de esto se activa y la
# aplicación se comporta como siempre (incluido el límite por IP). Con true: hace falta sesión para convertir,
# cada usuario tiene DAILY_FREE_CONVERSIONS al día (UTC) y el límite por IP se desactiva (en Render varias
# personas pueden compartir IP).
AUTH_ENABLED = _env_flag("MARKPDF_AUTH_ENABLED", default=False)
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()  # PostgreSQL; obligatoria en producción si AUTH_ENABLED
SESSION_SECRET = os.environ.get("MARKPDF_SESSION_SECRET", "")  # firma de la cookie de sesión (≥ 32 caracteres)
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "").strip()
PUBLIC_URL = os.environ.get("MARKPDF_PUBLIC_URL", "").strip().rstrip("/")  # p. ej. https://markpdf.onrender.com
DAILY_FREE_CONVERSIONS = _env_int("MARKPDF_DAILY_FREE_CONVERSIONS", 3)  # por usuario y día UTC

MAX_REPORT_ITEMS = 200  # tope de elementos en el informe de limpieza
FORM_OVERHEAD_BYTES = 1024 * 1024  # margen del formulario multipart (campos y delimitadores) sobre el PDF
JSON_OVERHEAD_BYTES = 64 * 1024  # margen del JSON de /download (nombre de archivo, comillas...)

BUSY_MESSAGE = "El servidor está ocupado convirtiendo otro PDF. Inténtalo de nuevo en unos segundos."
EXTRACT_UNAVAILABLE_MESSAGE = (
    "La extracción de imágenes (.zip) no está disponible en este servidor. "
    "Usa «Ignorar» o «Marcar posición»."
)

# Una sola conversión (o las que indique MARKPDF_MAX_CONCURRENT) a la vez; las demás reciben 503 al instante.
CONVERT_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_CONVERSIONS)
RATE_LIMITER = RateLimiter(RATE_LIMIT_PER_HOUR, 3600)

# Servicio de cuotas: None mientras AUTH_ENABLED=false. Se crea al arrancar (_lifespan) para que una mala
# configuración (sin base de datos o sin secreto) detenga el arranque en lugar de fallar con el primer usuario.
QUOTAS: Optional[QuotaService] = None


@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    global QUOTAS
    if AUTH_ENABLED:
        QUOTAS = build_quota_service(
            database_url=DATABASE_URL,
            daily_limit=DAILY_FREE_CONVERSIONS,
            production=IS_PRODUCTION,
        )
    yield
    with JOBS_LOCK:  # al apagar el servidor se borran las imágenes extraídas que sigan en disco
        for job in JOBS.values():
            shutil.rmtree(job["dir"], ignore_errors=True)
        JOBS.clear()


# En producción no se publican la documentación interactiva ni el esquema de la API.
_docs_off = {"docs_url": None, "redoc_url": None, "openapi_url": None} if IS_PRODUCTION else {}
app = FastAPI(title="MarkPDF", version=VERSION, lifespan=_lifespan, **_docs_off)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Middlewares (el último que se añade es el más externo: las cabeceras llegan también a los 413).
app.add_middleware(
    BodyLimitMiddleware,
    limits=lambda: {
        "/convert": MAX_UPLOAD_BYTES + FORM_OVERHEAD_BYTES,
        # El JSON escapa saltos de línea y comillas, así que puede pesar más que el texto: margen x2.
        "/download": 2 * MAX_DOWNLOAD_BYTES + JSON_OVERHEAD_BYTES,
    },
)

if AUTH_ENABLED:
    # Una mala configuración detiene el arranque aquí (no falla más tarde con el primer usuario).
    validate_settings(
        session_secret=SESSION_SECRET,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        public_url=PUBLIC_URL,
        production=IS_PRODUCTION,
    )
    from starlette.middleware.sessions import SessionMiddleware  # noqa: E402  (solo se necesita con login)

    app.include_router(build_router(build_oauth(GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET), PUBLIC_URL, lambda: QUOTAS))
    # Cookie de sesión firmada: HttpOnly siempre; Secure solo en producción (en local se usa http); SameSite=Lax.
    app.add_middleware(
        SessionMiddleware,
        secret_key=SESSION_SECRET,
        session_cookie=SESSION_COOKIE,
        max_age=SESSION_MAX_AGE,
        same_site="lax",
        https_only=IS_PRODUCTION,
    )
else:

    @app.get("/auth/me")
    def auth_me_disabled() -> JSONResponse:
        """La interfaz pregunta aquí si debe mostrar el login; sin autenticación responde que no."""
        return JSONResponse({"enabled": False}, headers=NO_STORE)


app.add_middleware(SecurityHeadersMiddleware, headers=build_security_headers(IS_PRODUCTION))


# ---------------------------------------------------------------------------
# Jobs (solo para image_mode="extract": guardamos las imágenes para el ZIP de descarga)
# ---------------------------------------------------------------------------
JOBS: dict[str, dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 60 * 60
MAX_JOBS = 50


def _gc_jobs() -> None:
    now = time.time()
    with JOBS_LOCK:
        for job_id, job in list(JOBS.items()):
            if now - job["ts"] > JOB_TTL_SECONDS:
                shutil.rmtree(job["dir"], ignore_errors=True)
                JOBS.pop(job_id, None)
        while len(JOBS) > MAX_JOBS:  # evita acumular disco si llegan muchas conversiones con imágenes
            oldest = min(JOBS, key=lambda k: JOBS[k]["ts"])
            shutil.rmtree(JOBS[oldest]["dir"], ignore_errors=True)
            JOBS.pop(oldest, None)


# ---------------------------------------------------------------------------
# Compatibilidad: nombres que antes vivían aquí
# ---------------------------------------------------------------------------
_call_to_markdown = extraction._call_to_markdown
_ocr_available = extraction._ocr_available
find_scanned_pages = extraction.find_scanned_pages


@functools.lru_cache(maxsize=1)
def _ocr_available_cached() -> bool:
    return bool(extraction._ocr_available())


# ---------------------------------------------------------------------------
# Conversión
# ---------------------------------------------------------------------------
def _open_document(pdf_path: Path) -> "fitz.Document":
    """Etapa 1: abre y valida el PDF con mensajes comprensibles (sin rutas ni trazas internas)."""
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise HTTPException(422, "No se pudo abrir el archivo: no parece un PDF válido o está dañado.") from exc
    try:
        if doc.needs_pass:
            raise HTTPException(422, "El PDF está protegido con contraseña.")
        n_pages = len(doc)
        if n_pages == 0:
            raise HTTPException(422, "El PDF no contiene páginas legibles (puede estar vacío o dañado).")
        if n_pages > MAX_PAGES:
            raise HTTPException(413, f"El PDF tiene {n_pages} páginas; el límite es {MAX_PAGES}.")
    except HTTPException:
        doc.close()
        raise
    return doc


def _source_pages(doc: "fitz.Document", skip: set[int]) -> dict[int, str]:
    """Capa de texto del PDF por página (referencia para métricas y cobertura)."""
    out: dict[int, str] = {}
    for idx, page in enumerate(doc):
        if idx in skip:
            continue
        try:
            out[idx] = plain_page_text(page)  # sin duplicados por superposición de capas
        except Exception:  # pragma: no cover
            continue
    return out


def convert_pdf(
    pdf_bytes: bytes,
    *,
    mode: str = DEFAULT_MODE,
    image_mode: str = DEFAULT_IMAGE_MODE,
    options: Optional[ConvertOptions] = None,
    **overrides: Any,
) -> dict[str, Any]:
    if options is None:
        options = ConvertOptions.for_mode(mode, image_mode=normalize_image_mode(image_mode), **overrides)
    opts = options
    if opts.image_mode == "describe" and not vision.is_available():
        raise HTTPException(400, vision.UNAVAILABLE_MESSAGE)

    work_dir = Path(tempfile.mkdtemp(prefix="markpdf_"))
    images_dir = work_dir / "images"
    pdf_path = work_dir / "input.pdf"
    doc: Optional["fitz.Document"] = None
    keep_dir = False

    try:
        with perf.stage("open_pdf"):
            pdf_path.write_bytes(pdf_bytes)
            doc = _open_document(pdf_path)  # 1) validación
        n_pages = len(doc)
        perf.note(pages=n_pages)

        form_report = None
        with perf.stage("forms"):
            try:
                if doc.is_form_pdf:
                    form_report = prepare_widgets(doc)  # copia en memoria: el archivo original no se toca
            except Exception:
                log.exception("No se pudieron procesar los campos de formulario")

        # "layout" y "ocr" se miden dentro de extraction; lo que queda aquí (tablas, imágenes, ensamblado) es "assemble".
        with perf.stage("assemble"):
            ext = extract_pages(doc, opts, images_dir)  # 2-4) extracción, tablas, imágenes
        warnings = list(ext.warnings)

        margins = None
        if opts.remove_headers_footers or opts.remove_page_numbers:
            with perf.stage("margins"):
                margins = analyze_margins(
                    doc,
                    remove_headers_footers=opts.remove_headers_footers,
                    remove_page_numbers=opts.remove_page_numbers,
                )

        with perf.stage("source_text"):
            source = _source_pages(doc, ext.no_text_layer)
            coverage = extraction_coverage(source, ext.pages)
            cov_warning = coverage_warning(coverage)
        if cov_warning:
            warnings.append(cov_warning)

        with perf.stage("clean"):
            result = clean_pages(ext.pages, opts.mode, margins.literals if margins else None, opts)  # 5) limpieza
        with perf.stage("validate"):
            markdown, validation_issues = validate_markdown(result.text)  # 6) validación del Markdown
        perf.note(md_chars=len(markdown), ocr_pages=len(ext.ocr_pages))

        # Imágenes extraídas: solo se conservan (y se ofrecen en el ZIP) las que el Markdown referencia.
        job_id: Optional[str] = None
        extracted_files = 0
        if opts.image_mode == "extract" and images_dir.exists():
            all_files = [f for f in images_dir.iterdir() if f.is_file()]
            files = [f for f in all_files if f"images/{quote(f.name)}" in markdown]
            for f in all_files:
                if f not in files:
                    f.unlink(missing_ok=True)
            extracted_files = len(files)
            if files:
                job_id = uuid.uuid4().hex
                keep_dir = True
                doc.close()  # en Windows no se puede borrar un archivo abierto
                doc = None
                pdf_path.unlink(missing_ok=True)
                with JOBS_LOCK:
                    JOBS[job_id] = {"dir": work_dir, "ts": time.time()}
        vstats = ext.visuals
        vstats.extracted = extracted_files if opts.image_mode == "extract" else 0

        if not markdown.strip() and not any("escaneada" in w or "OCR" in w for w in warnings):
            warnings.append("No se encontró texto en el PDF.")
        if form_report and form_report.unknown:
            warnings.append(
                f"{form_report.unknown} casillas de formulario tienen un estado indeterminable y se marcaron como [?]."
            )

        # 7) métricas
        with perf.stage("metrics"):
            source_text = "\n\n".join(t.strip() for _i, t in sorted(source.items()) if t.strip())
            raw_text = "\n\n".join(p.strip() for p in ext.pages if p.strip()).replace(FULL_MARK, "")
            m_final = metrics.text_stats(markdown)
            m_extraction = metrics.text_stats(raw_text)
            m_source = metrics.text_stats(source_text)
            comparison = metrics.comparison(
                m_extraction, m_final, "Markdown de la extracción antes de la limpieza configurable"
            )

        report_removed = result.removed[:MAX_REPORT_ITEMS]
        return {
            "markdown": markdown,
            "mode": opts.mode,
            "profile": PROFILE_LABELS[opts.mode],
            "options": opts.as_dict(),
            "metrics": {
                # Claves históricas (compatibilidad): tokens estimados = ceil(caracteres / 4).
                "chars": m_final["chars"],
                "words": m_final["words"],
                "tokens": m_final["tokens"],
                "tokens_before": m_extraction["tokens"],
                "tokens_change_percent": comparison["change_percent"],
                # Detalle: cada magnitud con su fuente.
                "tokens_exact": False,
                "token_estimator": metrics.TOKEN_ESTIMATOR,
                "source": m_source,
                "extraction": m_extraction,
                "final": m_final,
                "comparison": comparison,
            },
            "stats": {
                "pages": n_pages,
                "pages_processed": n_pages,
                "margin_lines_removed": result.margin_lines_removed,
                "fillers_removed": result.fillers_removed,
                "duplicates_removed": result.duplicates_removed,
                "repeated_blocks_removed": result.repeated_blocks_removed,
                "redundant_headings_removed": result.redundant_headings_removed,
                "empty_fields_omitted": result.empty_fields_omitted,
                "images": vstats.extracted + vstats.marked,
                "images_detail": vstats.as_dict(),
                "tables": ext.tables,
                "tables_approximate": ext.tables_approximate,
                "table_formats": ext.table_formats,
                "forms": form_report.as_dict() if form_report else None,
                "ocr_pages": len(ext.ocr_pages),
                "scanned_pages_without_text": len(ext.scanned_pages),
                "layout_engine": ext.layout_engine,
            },
            "report": {
                "removed": report_removed,
                "removed_truncated": len(result.removed) > MAX_REPORT_ITEMS,
                "margins": margins.as_report() if margins else [],
                "margins_uncertain": margins.uncertain if margins else [],
                "coverage": coverage.as_dict(),
                "validation": validation_issues,
            },
            "warnings": warnings,
            "job_id": job_id,
        }
    finally:
        if doc is not None:
            doc.close()
        if not keep_dir:
            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(BASE_DIR / "index.html")


@app.get("/capabilities")
def capabilities() -> dict[str, Any]:
    """Lo que este servidor puede hacer de verdad (la interfaz deshabilita lo que no esté disponible)."""
    vision_ok = vision.is_available()
    return {
        "version": VERSION,
        "layout_engine": extraction.layout_engine_available(),
        "ocr": _ocr_available_cached(),
        "vision": vision_ok,
        "modes": list(MODES),
        "image_modes": {
            m: (m != "describe" or vision_ok) and (m != "extract" or ALLOW_IMAGE_EXTRACT) for m in IMAGE_MODES
        },
        "table_formats": list(TABLE_FORMATS),
        "limits": {"max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024), "max_pages": MAX_PAGES},
    }


_FORM_OPTION_NAMES = (
    "remove_headers_footers",
    "remove_page_numbers",
    "join_paragraphs",
    "remove_duplicates",
    "omit_empty_fields",
    "omit_redundant_headings",
    "remove_repeated_blocks",
    "image_markers",
    "table_format",
)


def _upload_size(upload: UploadFile) -> int:
    """Tamaño del archivo subido SIN leerlo: Starlette ya lo tiene en un temporal, basta con ir al final."""
    stream = upload.file
    stream.seek(0, os.SEEK_END)
    size = stream.tell()
    stream.seek(0)
    return size


def _client_ip(request: Request) -> str:
    peer = request.client.host if request.client else None
    forwarded = ", ".join(request.headers.getlist("x-forwarded-for"))
    return client_ip(peer, forwarded, TRUSTED_PROXY_HOPS)


def _rate_limited(wait_seconds: int) -> HTTPException:
    minutes = max(1, -(-wait_seconds // 60))
    return HTTPException(
        429,
        f"Has alcanzado el límite de {RATE_LIMITER.max_hits} conversiones por hora. "
        f"Inténtalo de nuevo en {minutes} min.",
        headers={"Retry-After": str(wait_seconds)},
    )


LOGIN_REQUIRED_MESSAGE = "Inicia sesión con Google para convertir PDF."


def _current_user_id(request: Request) -> Optional[int]:
    """Usuario de la sesión (cookie firmada por SessionMiddleware, que solo existe con MARKPDF_AUTH_ENABLED=true)."""
    return session_user_id(request)


def _reserve_quota(user_id: int) -> Reservation:
    """Reserva 1 conversión (atómico). Traduce los rechazos a HTTP; ante un fallo de la base de datos, falla cerrado."""
    assert QUOTAS is not None
    try:
        return QUOTAS.reserve_user(user_id)
    except QuotaDenied as exc:
        raise HTTPException(exc.status, exc.message, headers=exc.headers()) from exc
    except QuotaStoreError as exc:
        log.error("No se pudo comprobar la cuota: %s", exc)
        raise HTTPException(
            503,
            "No se puede comprobar tu límite de conversiones en este momento. Inténtalo de nuevo en unos segundos.",
            headers={CODE_HEADER: "quota_unavailable", "Retry-After": "30"},
        ) from exc


@app.post("/convert")
def convert(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form(DEFAULT_MODE),
    image_mode: str = Form(DEFAULT_IMAGE_MODE),
    table_format: Optional[str] = Form(None),
    remove_headers_footers: Optional[str] = Form(None),
    remove_page_numbers: Optional[str] = Form(None),
    join_paragraphs: Optional[str] = Form(None),
    remove_duplicates: Optional[str] = Form(None),
    omit_empty_fields: Optional[str] = Form(None),
    omit_redundant_headings: Optional[str] = Form(None),
    remove_repeated_blocks: Optional[str] = Form(None),
    image_markers: Optional[str] = Form(None),
) -> dict[str, Any]:
    # Con login activo, sin sesión válida no se convierte (la comprobación es del servidor, no de la interfaz).
    user_id: Optional[int] = None
    if QUOTAS is not None:
        user_id = _current_user_id(request)
        if user_id is None:
            raise HTTPException(401, LOGIN_REQUIRED_MESSAGE, headers={CODE_HEADER: "login_required"})

    values = {"mode": mode, "image_mode": image_mode}
    local = locals()
    values.update({name: local[name] for name in _FORM_OPTION_NAMES if local[name] is not None})
    try:
        opts = options_from_form(values)
    except OptionsError as exc:
        raise HTTPException(400, str(exc)) from exc
    if opts.image_mode == "describe" and not vision.is_available():
        raise HTTPException(400, vision.UNAVAILABLE_MESSAGE)
    if opts.image_mode == "extract" and not ALLOW_IMAGE_EXTRACT:
        raise HTTPException(400, EXTRACT_UNAVAILABLE_MESSAGE)

    # 1) Comprobaciones baratas del archivo, sin cargarlo entero en memoria (solo se lee el principio).
    size = _upload_size(file)
    if size > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
    head = file.file.read(1024)
    file.file.seek(0)
    if not head.strip():
        raise HTTPException(400, "El archivo está vacío.")
    if b"%PDF-" not in head:
        raise HTTPException(400, "El archivo no es un PDF.")

    # 2) Admisión: límite por IP (429) y una conversión a la vez (503). Los rechazos no gastan cuota.
    #    El límite por IP solo rige SIN login: con login se limita por usuario (en Render varias personas comparten IP).
    ip = _client_ip(request)
    if QUOTAS is None:
        wait = RATE_LIMITER.retry_after(ip)
        if wait:
            raise _rate_limited(wait)
    if not CONVERT_SLOTS.acquire(blocking=False):
        raise HTTPException(503, BUSY_MESSAGE, headers={"Retry-After": "10"})
    prof = perf.start()  # diagnóstico de rendimiento: solo tiempos y MB, nunca contenido ni datos personales
    perf.note(mode=opts.mode, image_mode=opts.image_mode)
    reservation: Optional[Reservation] = None
    completed = False
    status = "error"  # solo para el diagnóstico de rendimiento (ok / http_<código> / error)
    try:
        # Cuota (solo con login): se reserva antes de convertir (atómico) y se devuelve en el `finally` si la
        # conversión no termina bien.
        with perf.stage("quota"):
            if QUOTAS is not None:
                assert user_id is not None
                reservation = _reserve_quota(user_id)
            else:
                wait = RATE_LIMITER.hit(ip)  # atómico: la consulta de arriba solo evita trabajo inútil
                if wait:
                    raise _rate_limited(wait)

        # 3) Solo ahora se lee el PDF completo (ya sabemos que cabe en el límite).
        with perf.stage("read_upload"):
            data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"El archivo supera el límite de {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
        perf.note(pdf_kb=len(data) // 1024)

        _gc_jobs()
        try:
            result = convert_pdf(data, options=opts)
        except HTTPException:
            raise
        except Exception as exc:
            error_id = uuid.uuid4().hex[:8]
            log.exception("Error convirtiendo PDF (id %s)", error_id)
            raise HTTPException(
                500, f"No se pudo procesar el PDF. Prueba con otro archivo o inténtalo de nuevo (código {error_id})."
            ) from exc
        completed = True  # a partir de aquí la conversión cuenta
        status = "ok"
    except HTTPException as exc:
        status = f"http_{exc.status_code}"
        raise
    finally:
        if reservation is not None and not completed and QUOTAS is not None:
            QUOTAS.release(reservation)  # no lanza: un fallo aquí no debe tapar el error original
        CONVERT_SLOTS.release()
        perf.finish(prof, status)

    result["filename"] = _safe_md_name(file.filename)
    return result


class DownloadRequest(BaseModel):
    markdown: str
    filename: str = "documento.md"
    job_id: Optional[str] = None


def _safe_md_name(name: Optional[str]) -> str:
    stem = Path(name or "documento").stem
    stem = re.sub(r"[^\w\-. ]", "_", stem, flags=re.UNICODE).strip(" .") or "documento"
    return f"{stem}.md"


def _content_disposition(filename: str) -> str:
    ascii_name = filename.encode("ascii", "ignore").decode() or "documento.md"
    ascii_name = ascii_name.replace('"', "")
    return f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}"


@app.post("/download")
def download(req: DownloadRequest) -> Response:
    """
    Devuelve el Markdown (posiblemente editado por el usuario) como archivo .md.
    Si la conversión extrajo imágenes (job_id), devuelve un .zip con el .md y la carpeta images/.
    """
    body = req.markdown.encode("utf-8")  # se codifica una sola vez; el límite se mide en bytes, no en caracteres
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(413, f"El texto supera el límite de {MAX_DOWNLOAD_BYTES // (1024 * 1024)} MB.")
    filename = _safe_md_name(req.filename)
    job = None
    if req.job_id:
        with JOBS_LOCK:
            job = JOBS.get(req.job_id)

    if job is not None:
        images_dir = Path(job["dir"]) / "images"
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, body)
            if images_dir.exists():
                for img in sorted(images_dir.iterdir()):
                    if img.is_file():
                        zf.write(img, f"images/{img.name}")
        zip_name = Path(filename).with_suffix(".zip").name
        return Response(
            buffer.getvalue(),
            media_type="application/zip",
            headers={"Content-Disposition": _content_disposition(zip_name)},
        )

    return Response(
        body,
        media_type="text/markdown; charset=utf-8",
        headers={"Content-Disposition": _content_disposition(filename)},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
