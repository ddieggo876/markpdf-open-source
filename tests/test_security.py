# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Seguridad básica: límites de uso, tamaño, concurrencia, /download, documentación, cabeceras e interfaz."""

import asyncio
import importlib.util
import io
import os
import re
import sys
import threading
import types
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import extraction
import main
import security
from security import RateLimiter, client_ip, normalize_ip
from tests import pdf_fixtures as fx

ROOT = Path(__file__).resolve().parent.parent
HTML = (ROOT / "index.html").read_text(encoding="utf-8")

client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def no_ocr_engine(monkeypatch):
    monkeypatch.setattr(extraction, "_ocr_available", lambda: False)


def post(pdf: bytes, *, c=client, filename="doc.pdf", headers=None, **data):
    return c.post("/convert", files={"file": (filename, pdf, "application/pdf")}, data=data, headers=headers)


def load_main_copy(monkeypatch, **env):
    """Carga una copia independiente de main.py con las variables de entorno indicadas (las MARKPDF_* previas se borran)."""
    for key in [k for k in os.environ if k.startswith("MARKPDF_")]:
        monkeypatch.delenv(key)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    name = f"main_copy_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def copy_client(module):
    return TestClient(module.app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Valores por defecto y configuración por variables de entorno
# ---------------------------------------------------------------------------
def test_defaults_are_the_public_limits(monkeypatch):
    m = load_main_copy(monkeypatch)
    assert m.MAX_UPLOAD_BYTES == 10 * 1024 * 1024
    assert m.MAX_PAGES == 100
    assert m.MAX_DOWNLOAD_BYTES == 5 * 1024 * 1024
    assert m.MAX_CONCURRENT_CONVERSIONS == 1
    assert m.RATE_LIMIT_PER_HOUR == 10 and m.RATE_LIMITER.max_hits == 10 and m.RATE_LIMITER.window == 3600
    assert m.TRUSTED_PROXY_HOPS == 0
    assert m.IS_PRODUCTION is False and m.ALLOW_IMAGE_EXTRACT is True
    limits = copy_client(m).get("/capabilities").json()["limits"]
    assert limits == {"max_upload_mb": 10, "max_pages": 100}


def test_limits_are_configurable_by_environment_variables(monkeypatch):
    m = load_main_copy(
        monkeypatch,
        MARKPDF_MAX_UPLOAD_MB="3",
        MARKPDF_MAX_PAGES="7",
        MARKPDF_MAX_DOWNLOAD_MB="2",
        MARKPDF_MAX_CONCURRENT="2",
        MARKPDF_RATE_LIMIT_PER_HOUR="4",
        MARKPDF_TRUSTED_PROXY_HOPS="1",
    )
    assert (m.MAX_UPLOAD_BYTES, m.MAX_PAGES, m.MAX_DOWNLOAD_BYTES) == (3 * 1024 * 1024, 7, 2 * 1024 * 1024)
    assert m.MAX_CONCURRENT_CONVERSIONS == 2 and m.RATE_LIMITER.max_hits == 4 and m.TRUSTED_PROXY_HOPS == 1
    assert copy_client(m).get("/capabilities").json()["limits"] == {"max_upload_mb": 3, "max_pages": 7}


@pytest.mark.parametrize("bad", ["abc", "", "0", "-5", "1.5"])
def test_invalid_environment_values_fall_back_to_safe_defaults(monkeypatch, bad):
    m = load_main_copy(monkeypatch, MARKPDF_MAX_UPLOAD_MB=bad, MARKPDF_MAX_PAGES=bad, MARKPDF_MAX_DOWNLOAD_MB=bad)
    assert (m.MAX_UPLOAD_BYTES, m.MAX_PAGES, m.MAX_DOWNLOAD_BYTES) == (10 * 1024 * 1024, 100, 5 * 1024 * 1024)


def test_rate_limit_can_be_disabled_with_zero(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_RATE_LIMIT_PER_HOUR="0")
    assert m.RATE_LIMITER.enabled is False


# ---------------------------------------------------------------------------
# RateLimiter (unidad)
# ---------------------------------------------------------------------------
class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def test_rate_limiter_allows_up_to_the_limit_then_blocks_with_a_wait_time():
    clock = FakeClock()
    rl = RateLimiter(3, 3600, clock=clock)
    assert [rl.hit("a") for _ in range(3)] == [0, 0, 0]
    wait = rl.hit("a")
    assert 3590 <= wait <= 3600
    assert rl.retry_after("a") == wait


def test_rate_limiter_window_slides_and_keys_are_independent():
    clock = FakeClock()
    rl = RateLimiter(2, 100, clock=clock)
    assert rl.hit("a") == 0
    clock.now += 60
    assert rl.hit("a") == 0
    assert rl.hit("a") > 0 and rl.hit("b") == 0  # otra IP no se ve afectada
    clock.now += 41  # caduca solo el primer uso (hace 101 s)
    assert rl.hit("a") == 0
    assert rl.hit("a") > 0


def test_retry_after_does_not_consume_quota():
    rl = RateLimiter(1, 100, clock=FakeClock())
    for _ in range(5):
        assert rl.retry_after("a") == 0
    assert rl.hit("a") == 0 and rl.hit("a") > 0


def test_rate_limiter_disabled_when_max_is_zero():
    rl = RateLimiter(0, 100, clock=FakeClock())
    assert all(rl.hit("a") == 0 for _ in range(50)) and rl.retry_after("a") == 0


def test_rate_limiter_memory_is_capped():
    rl = RateLimiter(1, 100, max_tracked=5, clock=FakeClock())
    for i in range(50):
        rl.hit(f"ip{i}")
    assert len(rl._hits) <= 5


# ---------------------------------------------------------------------------
# IP del cliente: no fiarse de lo que envía el usuario
# ---------------------------------------------------------------------------
def test_forwarded_header_is_ignored_by_default():
    assert client_ip("203.0.113.9", "1.2.3.4", 0) == "203.0.113.9"
    assert client_ip("203.0.113.9", "1.2.3.4, 5.6.7.8", 0) == "203.0.113.9"


def test_with_trusted_proxies_only_the_entry_added_by_them_counts():
    # El usuario escribe "6.6.6.6" a la izquierda; el proxy de confianza añade la IP real al final.
    assert client_ip("10.0.0.1", "6.6.6.6, 198.51.100.7", 1) == "198.51.100.7"
    # Con dos proxies de confianza se toma la penúltima.
    assert client_ip("10.0.0.1", "6.6.6.6, 198.51.100.7, 10.0.0.2", 2) == "198.51.100.7"


def test_forwarded_header_that_is_missing_short_or_invalid_falls_back_to_the_socket():
    assert client_ip("10.0.0.1", None, 1) == "10.0.0.1"
    assert client_ip("10.0.0.1", "", 1) == "10.0.0.1"
    assert client_ip("10.0.0.1", "198.51.100.7", 2) == "10.0.0.1"  # menos entradas que proxies
    assert client_ip("10.0.0.1", "6.6.6.6, no-es-una-ip", 1) == "10.0.0.1"


def test_ipv6_is_grouped_by_64_and_mapped_ipv4_is_unwrapped():
    assert normalize_ip("2001:db8:1:2:aaaa:bbbb:cccc:dddd") == normalize_ip("2001:db8:1:2:1::1") == "2001:db8:1:2::/64"
    assert normalize_ip("2001:db8:1:3::1") != normalize_ip("2001:db8:1:2::1")
    assert normalize_ip("::ffff:203.0.113.9") == "203.0.113.9"
    assert normalize_ip("testclient") == "testclient" and normalize_ip(None) == "unknown"


# ---------------------------------------------------------------------------
# 429: conversiones por IP y hora
# ---------------------------------------------------------------------------
def test_eleventh_conversion_from_the_same_ip_is_rejected_with_429(monkeypatch):
    m = load_main_copy(monkeypatch)  # límites reales por defecto: 10 por hora
    c = copy_client(m)
    pdf = fx.clean_pdf()
    assert [post(pdf, c=c).status_code for _ in range(10)] == [200] * 10
    r = post(pdf, c=c)
    assert r.status_code == 429
    assert "10 conversiones por hora" in r.json()["detail"] and int(r.headers["retry-after"]) > 0


def test_rate_limit_is_per_ip_and_x_forwarded_for_cannot_dodge_it(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_RATE_LIMIT_PER_HOUR="1")  # sin proxies de confianza
    c, pdf = copy_client(m), fx.clean_pdf()
    assert post(pdf, c=c, headers={"X-Forwarded-For": "1.1.1.1"}).status_code == 200
    # Cambiar la cabecera no da una cuota nueva: se cuenta la IP real del socket.
    assert post(pdf, c=c, headers={"X-Forwarded-For": "2.2.2.2"}).status_code == 429
    assert post(pdf, c=c, headers={"X-Real-IP": "3.3.3.3", "Forwarded": "for=4.4.4.4"}).status_code == 429


def test_behind_a_trusted_proxy_the_client_is_taken_from_the_right_of_the_header(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_RATE_LIMIT_PER_HOUR="1", MARKPDF_TRUSTED_PROXY_HOPS="1")
    c, pdf = copy_client(m), fx.clean_pdf()
    assert post(pdf, c=c, headers={"X-Forwarded-For": "6.6.6.6, 198.51.100.7"}).status_code == 200
    # Misma IP real, distinta "IP inventada" a la izquierda: sigue siendo el mismo usuario.
    assert post(pdf, c=c, headers={"X-Forwarded-For": "7.7.7.7, 198.51.100.7"}).status_code == 429
    # Otra IP real distinta: otra cuota.
    assert post(pdf, c=c, headers={"X-Forwarded-For": "6.6.6.6, 198.51.100.8"}).status_code == 200


def test_rejected_requests_do_not_consume_the_rate_limit(monkeypatch):
    limiter = RateLimiter(2, 3600)
    monkeypatch.setattr(main, "RATE_LIMITER", limiter)
    assert post(b"esto no es un pdf").status_code == 400
    assert post(b"").status_code == 400
    assert post(fx.clean_pdf(), image_mode="describe").status_code == 400
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10)
    assert post(fx.clean_pdf()).status_code == 413
    assert limiter._hits == {}  # ningún rechazo gastó cuota
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 10 * 1024 * 1024)
    assert post(fx.clean_pdf()).status_code == 200
    assert len(limiter._hits["testclient"]) == 1  # solo la conversión real


# ---------------------------------------------------------------------------
# 503: una sola conversión a la vez
# ---------------------------------------------------------------------------
def test_busy_server_returns_503_with_a_clear_message_and_does_not_consume_quota(monkeypatch):
    limiter = RateLimiter(5, 3600)
    monkeypatch.setattr(main, "RATE_LIMITER", limiter)
    assert main.CONVERT_SLOTS.acquire(blocking=False)
    try:
        r = post(fx.clean_pdf())
    finally:
        main.CONVERT_SLOTS.release()
    assert r.status_code == 503
    assert "ocupado" in r.json()["detail"] and r.headers["retry-after"] == "10"
    assert limiter._hits == {}  # el rechazo por ocupado no gasta cuota
    assert post(fx.clean_pdf()).status_code == 200  # y al liberarse el hueco vuelve a funcionar


def test_second_conversion_while_one_is_running_gets_503_and_the_first_finishes(monkeypatch):
    started, release = threading.Event(), threading.Event()
    real = main.convert_pdf

    def slow(*args, **kwargs):
        started.set()
        assert release.wait(30)
        return real(*args, **kwargs)

    monkeypatch.setattr(main, "convert_pdf", slow)
    first: dict = {}
    worker = threading.Thread(
        target=lambda: first.setdefault("r", post(fx.clean_pdf(), c=TestClient(main.app, raise_server_exceptions=False)))
    )
    worker.start()
    try:
        assert started.wait(30)
        second = post(fx.clean_pdf())
        assert second.status_code == 503
    finally:
        release.set()
        worker.join(60)
    assert first["r"].status_code == 200
    monkeypatch.setattr(main, "convert_pdf", real)
    assert post(fx.clean_pdf()).status_code == 200


@pytest.mark.parametrize("failure", ["validation", "crash"])
def test_the_slot_is_always_released_even_when_the_conversion_fails(monkeypatch, failure):
    if failure == "crash":
        monkeypatch.setattr(main, "extract_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
        assert post(fx.clean_pdf()).status_code == 500
    else:
        assert post(fx.corrupt_pdf()).status_code == 422
    assert main.CONVERT_SLOTS.acquire(blocking=False)
    main.CONVERT_SLOTS.release()


# ---------------------------------------------------------------------------
# Tamaño y páginas del PDF
# ---------------------------------------------------------------------------
def test_upload_size_is_measured_without_reading_the_file():
    class NoRead(io.BytesIO):
        def read(self, *args, **kwargs):  # pragma: no cover - solo debe fallar si alguien lee
            raise AssertionError("no se debe leer el archivo para medir su tamaño")

    upload = types.SimpleNamespace(file=NoRead(b"x" * 1234))
    assert main._upload_size(upload) == 1234
    assert upload.file.tell() == 0  # queda al principio, listo para leerse


def test_oversized_pdf_is_rejected_before_anything_is_read_or_converted(monkeypatch):
    limit = 4096
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", limit)
    limiter = RateLimiter(5, 3600)
    monkeypatch.setattr(main, "RATE_LIMITER", limiter)
    called = []
    monkeypatch.setattr(main, "convert_pdf", lambda *a, **k: called.append(1))
    r = post(b"%PDF-1.4\n" + b"x" * limit)  # un poco más que el límite
    assert r.status_code == 413 and "límite" in r.json()["detail"]
    assert not called and limiter._hits == {}
    assert main.CONVERT_SLOTS.acquire(blocking=False)  # no se llegó a ocupar el hueco
    main.CONVERT_SLOTS.release()


def test_pdf_exactly_at_the_limit_is_not_rejected_for_size(monkeypatch):
    limit = 4096
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", limit)
    r = post(b"%PDF-" + b"x" * (limit - 5))  # justo el límite: pasa la comprobación de tamaño
    assert r.status_code == 422  # ...y falla después por estar dañado, no por tamaño


def test_page_limit_is_enforced_by_the_server(monkeypatch):
    monkeypatch.setattr(main, "MAX_PAGES", 3)
    r = post(fx.contract_pdf(5))
    assert r.status_code == 413 and "5 páginas" in r.json()["detail"] and "3" in r.json()["detail"]
    assert post(fx.contract_pdf(3)).status_code == 200


def test_declared_oversized_body_is_cut_by_the_middleware_without_reaching_the_endpoint(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main, "FORM_OVERHEAD_BYTES", 0)
    called = []
    monkeypatch.setattr(main, "convert_pdf", lambda *a, **k: called.append(1))
    r = client.post(
        "/convert", content=b"x" * 5000, headers={"content-type": "multipart/form-data; boundary=zzz"}
    )
    assert r.status_code == 413 and r.json()["detail"] == security.TOO_LARGE_MESSAGE
    assert not called


def asgi_post(path: str, first: bytes, chunk: bytes, content_type: str, *, max_chunks: int = 10_000):
    """Envía un cuerpo SIN Content-Length (como una subida por trozos) directamente a la app ASGI."""
    served = {"chunks": 0}
    sent: list[dict] = []

    async def receive():
        served["chunks"] += 1
        if served["chunks"] > max_chunks:  # pragma: no cover - evita un bucle infinito si el límite no actúa
            return {"type": "http.request", "body": b"", "more_body": False}
        return {"type": "http.request", "body": first if served["chunks"] == 1 else chunk, "more_body": True}

    async def send(message):
        sent.append(message)

    scope = {
        "type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": "POST", "scheme": "http",
        "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
        "headers": [(b"content-type", content_type.encode())], "client": ("198.51.100.1", 5000),
        "server": ("test", 80),
    }
    asyncio.run(main.app(scope, receive, send))
    return served["chunks"], sent


def test_chunked_upload_without_content_length_is_cut_while_it_arrives(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main, "FORM_OVERHEAD_BYTES", 0)
    # Cabecera multipart válida seguida de datos "infinitos" (el analizador no puede quejarse antes que el límite).
    head = (
        b'--zzz\r\nContent-Disposition: form-data; name="file"; filename="a.pdf"\r\n'
        b"Content-Type: application/pdf\r\n\r\n%PDF-1.4\n"
    )
    chunks, sent = asgi_post(
        "/convert", head, b"x" * 400, "multipart/form-data; boundary=zzz", max_chunks=1000
    )
    starts = [m for m in sent if m["type"] == "http.response.start"]
    assert len(starts) == 1 and starts[0]["status"] == 413  # una única respuesta, y es la del límite
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    assert security.TOO_LARGE_MESSAGE.encode() in body
    assert chunks <= 6  # se dejó de leer enseguida, no se consumieron los 1000 trozos


# ---------------------------------------------------------------------------
# /download: máximo 5 MB de texto
# ---------------------------------------------------------------------------
def test_download_limit_is_5mb_by_default(monkeypatch):
    assert load_main_copy(monkeypatch).MAX_DOWNLOAD_BYTES == 5 * 1024 * 1024


def test_download_rejects_text_over_the_limit_and_accepts_the_limit_exactly(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 100)
    over = client.post("/download", json={"markdown": "x" * 101})
    assert over.status_code == 413 and "límite" in over.json()["detail"]
    ok = client.post("/download", json={"markdown": "x" * 100})
    assert ok.status_code == 200 and ok.content == b"x" * 100


def test_download_limit_counts_bytes_not_characters(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 100)
    assert client.post("/download", json={"markdown": "ñ" * 50}).status_code == 200  # 100 bytes
    assert client.post("/download", json={"markdown": "ñ" * 51}).status_code == 413  # 102 bytes


def test_download_zip_path_keeps_the_same_content_and_also_obeys_the_limit(monkeypatch):
    body = post(fx.logo_pdf(5, with_figure=True), image_mode="extract").json()
    text = body["markdown"] + "\nñ"
    r = client.post("/download", json={"markdown": text, "filename": "doc.md", "job_id": body["job_id"]})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    import zipfile

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert zf.read("doc.md").decode("utf-8") == text
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 10)
    big = client.post("/download", json={"markdown": text, "filename": "doc.md", "job_id": body["job_id"]})
    assert big.status_code == 413


def test_huge_download_body_is_cut_by_the_middleware(monkeypatch):
    monkeypatch.setattr(main, "MAX_DOWNLOAD_BYTES", 100)
    r = client.post("/download", json={"markdown": "x" * 300_000})
    assert r.status_code == 413


# ---------------------------------------------------------------------------
# Producción: documentación de la API y extracción de imágenes
# ---------------------------------------------------------------------------
DOC_PATHS = ("/docs", "/redoc", "/openapi.json")


def test_docs_are_available_in_development_by_default():
    assert [client.get(p).status_code for p in DOC_PATHS] == [200, 200, 200]


def test_docs_redoc_and_openapi_are_disabled_in_production(monkeypatch):
    c = copy_client(load_main_copy(monkeypatch, MARKPDF_ENV="production"))
    assert [c.get(p).status_code for p in DOC_PATHS] == [404, 404, 404]
    assert c.get("/").status_code == 200 and c.get("/capabilities").status_code == 200  # la app sigue funcionando


def test_image_extraction_is_off_in_production_in_capabilities_and_in_the_server(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_ENV="production")
    c = copy_client(m)
    assert c.get("/capabilities").json()["image_modes"]["extract"] is False
    for alias in ("extract", "references"):  # también por su nombre antiguo
        r = post(fx.logo_pdf(5, with_figure=True), c=c, image_mode=alias)
        assert r.status_code == 400 and "no está disponible" in r.json()["detail"]
    assert post(fx.clean_pdf(), c=c, image_mode="markers").status_code == 200  # el resto sigue igual
    assert m.JOBS == {}  # no se guardó ninguna imagen


def test_image_extraction_can_be_re_enabled_explicitly(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_ENV="production", MARKPDF_ENABLE_IMAGE_EXTRACT="true")
    c = copy_client(m)
    assert c.get("/capabilities").json()["image_modes"]["extract"] is True
    assert post(fx.logo_pdf(5, with_figure=True), c=c, image_mode="extract").status_code == 200


def test_the_extract_code_is_still_there_in_development():
    r = post(fx.logo_pdf(5, with_figure=True), image_mode="extract")
    assert r.status_code == 200 and r.json()["stats"]["images_detail"]["extracted"] == 2


# ---------------------------------------------------------------------------
# Cabeceras de seguridad
# ---------------------------------------------------------------------------
EXPECTED_HEADERS = ("content-security-policy", "x-content-type-options", "x-frame-options", "referrer-policy", "permissions-policy")


@pytest.mark.parametrize("path", ["/", "/capabilities", "/static/logo-header.png"])
def test_security_headers_are_on_every_kind_of_response(path):
    r = client.get(path)
    assert r.status_code == 200
    for name in EXPECTED_HEADERS:
        assert name in r.headers, name
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["x-frame-options"] == "DENY"
    csp = r.headers["content-security-policy"]
    for directive in ("object-src 'none'", "frame-ancestors 'none'", "base-uri 'self'", "default-src 'self'"):
        assert directive in csp


def test_security_headers_are_also_on_error_responses():
    for r in (post(b"no es un pdf"), client.get("/no-existe"), client.post("/download", json={})):
        assert r.status_code >= 400
        assert r.headers["x-content-type-options"] == "nosniff" and "content-security-policy" in r.headers


def test_security_headers_are_also_on_413_from_the_middleware(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1000)
    monkeypatch.setattr(main, "FORM_OVERHEAD_BYTES", 0)
    r = client.post("/convert", content=b"x" * 5000, headers={"content-type": "multipart/form-data; boundary=zzz"})
    assert r.status_code == 413 and "x-frame-options" in r.headers and "content-security-policy" in r.headers


def test_security_headers_are_also_on_429(monkeypatch):
    monkeypatch.setattr(main, "RATE_LIMITER", RateLimiter(1, 3600))
    assert post(fx.clean_pdf()).status_code == 200
    r = post(fx.clean_pdf())
    assert r.status_code == 429 and "x-frame-options" in r.headers and "content-security-policy" in r.headers


def test_security_headers_are_also_on_503():
    assert main.CONVERT_SLOTS.acquire(blocking=False)
    try:
        r = post(fx.clean_pdf())
    finally:
        main.CONVERT_SLOTS.release()
    assert r.status_code == 503 and "x-frame-options" in r.headers and "content-security-policy" in r.headers


def test_hsts_only_in_production(monkeypatch):
    assert "strict-transport-security" not in client.get("/").headers
    c = copy_client(load_main_copy(monkeypatch, MARKPDF_ENV="production"))
    assert c.get("/").headers["strict-transport-security"].startswith("max-age=")


def test_post_responses_are_not_cached_but_static_files_can_be():
    assert "no-store" in post(fx.clean_pdf()).headers["cache-control"]
    assert "no-store" in client.post("/download", json={"markdown": "x"}).headers["cache-control"]
    assert "no-store" not in client.get("/static/logo-header.png").headers.get("cache-control", "")


def test_csp_allows_every_external_script_the_ui_loads():
    """Si alguien añade un CDN a index.html sin actualizar la política, la interfaz dejaría de funcionar."""
    script_src = next(d for d in security.CONTENT_SECURITY_POLICY.split("; ") if d.startswith("script-src"))
    external = re.findall(r'<script[^>]+src="(https://[^/"]+)', HTML)
    assert external, "se esperaban scripts externos en index.html"
    for origin in external:
        assert origin in script_src, f"{origin} no está permitido por la CSP"
    # Los scripts y estilos en línea de la interfaz necesitan 'unsafe-inline'.
    assert "<script>" in HTML and "'unsafe-inline'" in script_src
    assert "<style>" in HTML and "'unsafe-inline'" in security.CONTENT_SECURITY_POLICY


def test_existing_headers_are_not_overwritten_by_the_middleware():
    sent: list[dict] = []

    async def inner(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"x-frame-options", b"SAMEORIGIN")]})
        await send({"type": "http.response.body", "body": b""})

    async def collect(message):
        sent.append(message)

    app = security.SecurityHeadersMiddleware(inner, security.build_security_headers(False))
    asyncio.run(app({"type": "http", "method": "GET", "path": "/"}, None, collect))
    headers = [h for h in sent[0]["headers"] if h[0] == b"x-frame-options"]
    assert headers == [(b"x-frame-options", b"SAMEORIGIN")]


# ---------------------------------------------------------------------------
# Interfaz: la opción de extraer imágenes queda oculta, pero su código sigue
# ---------------------------------------------------------------------------
def test_extract_option_is_hidden_by_default_and_only_revealed_by_capabilities():
    option = re.search(r'<option value="extract"[^>]*>', HTML).group(0)
    assert " hidden" in option and " disabled" in option
    assert "caps.image_modes.extract === true" in HTML
    assert "extractOpt.hidden = false" in HTML and "extractOpt.disabled = false" in HTML


def test_extract_code_is_not_removed_from_the_ui():
    assert "extract: 'Las imágenes se guardan como PNG" in HTML  # texto de ayuda
    assert "Descargar .zip (.md + imágenes)" in HTML  # flujo de descarga del ZIP
    assert 'value="extract"' in HTML


def test_ui_fallback_limit_matches_the_new_default():
    assert '<span id="maxMb">10</span>' in HTML and "maxBytes: 10 * 1024 * 1024" in HTML
