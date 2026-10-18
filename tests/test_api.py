# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""Pruebas del API HTTP: validación, errores seguros, limpieza de temporales, opciones cableadas."""

import io
import math
import os
import re
import tempfile
import time
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import extraction
import main
import options as options_module
from tests import pdf_fixtures as fx

client = TestClient(main.app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def no_ocr_engine(monkeypatch):
    monkeypatch.setattr(extraction, "_ocr_available", lambda: False)


@pytest.fixture
def tmp_dirs(monkeypatch):
    """Registra los directorios temporales creados por la conversión para comprobar que se borran."""
    created: list[Path] = []
    real = tempfile.mkdtemp

    def spy(*args, **kwargs):
        path = real(*args, **kwargs)
        created.append(Path(path))
        return path

    monkeypatch.setattr(main.tempfile, "mkdtemp", spy)
    return created


def post(pdf: bytes, filename="doc.pdf", **data):
    return client.post("/convert", files={"file": (filename, pdf, "application/pdf")}, data=data)


# ---------------------------------------------------------------------------
# Contrato de /convert
# ---------------------------------------------------------------------------
def test_convert_returns_the_documented_response_shape():
    r = post(fx.contract_pdf(5))
    assert r.status_code == 200
    body = r.json()
    for key in ("markdown", "mode", "profile", "options", "metrics", "stats", "report", "warnings", "job_id", "filename"):
        assert key in body
    assert body["mode"] == "clean" and body["profile"] == "Optimizado para IA"  # perfil por defecto
    assert body["filename"] == "doc.md"
    assert body["options"]["image_mode"] == "markers"
    assert body["stats"]["pages"] == 5


def test_profiles_set_defaults_and_options_override_them():
    pdf = fx.contract_pdf(5)
    faithful = post(pdf, mode="faithful").json()
    assert faithful["options"]["remove_headers_footers"] is False and faithful["profile"] == "Fiel al original"
    compact = post(pdf, mode="compact").json()
    assert compact["profile"] == "Compacto"
    override = post(pdf, mode="faithful", remove_headers_footers="true").json()
    assert override["options"]["remove_headers_footers"] is True
    assert "CONTRATO DE TRABAJO - ACME S.L." not in override["markdown"]
    assert faithful["markdown"].count("CONTRATO DE TRABAJO - ACME S.L.") == 5


def test_every_form_option_reaches_the_engine_and_changes_the_result():
    """Ninguna opción de la interfaz es decorativa: cada una produce un cambio verificable."""
    base_dup = post(fx.duplicates_pdf()).json()["markdown"]
    assert post(fx.duplicates_pdf(), remove_duplicates="false").json()["markdown"] != base_dup

    contract = fx.contract_pdf(5)
    assert post(contract, remove_headers_footers="false").json()["markdown"] != post(contract).json()["markdown"]
    assert post(contract, remove_page_numbers="false").json()["markdown"] != post(contract).json()["markdown"]

    hyph = fx.hyphenation_pdf()
    assert post(hyph, join_paragraphs="false").json()["markdown"] != post(hyph).json()["markdown"]

    form = fx.contract_form_pdf()
    assert post(form, omit_empty_fields="true").json()["markdown"] != post(form).json()["markdown"]

    logo = fx.logo_pdf(5, with_figure=True)
    assert post(logo, image_mode="ignore").json()["markdown"] != post(logo).json()["markdown"]
    assert post(logo, image_mode="ignore", image_markers="true").json()["markdown"] == post(logo).json()["markdown"]

    table = fx.table_empty_cells_pdf()
    assert post(table, table_format="linear").json()["markdown"] != post(table, table_format="literal").json()["markdown"]
    assert post(table, table_format="linear").json()["markdown"] != post(table).json()["markdown"]

    dups = fx.duplicates_pdf()
    sentence = "El arrendatario no podra subarrendar"
    keep_all = post(dups, remove_duplicates="false").json()["markdown"]
    blocks = post(dups, remove_duplicates="false", remove_repeated_blocks="true").json()
    assert keep_all.count(sentence) == 2 and blocks["markdown"].count(sentence) == 1
    assert blocks["stats"]["repeated_blocks_removed"] == 1

    # `omit_redundant_headings` solo actúa sobre un encabezado idéntico al inmediatamente anterior (el efecto
    # sobre el texto está probado en test_cleaning.py); aquí se comprueba que llega al motor.
    headings = post(fx.clean_pdf(), omit_redundant_headings="true").json()
    assert headings["options"]["omit_redundant_headings"] is True


def test_option_names_in_the_form_match_the_documented_options():
    form_fields = set(main._FORM_OPTION_NAMES)
    assert form_fields == set(options_module.BOOL_OPTIONS) | {"table_format"}


def test_legacy_identifiers_are_still_accepted():
    r = post(fx.logo_pdf(2, with_figure=True), image_mode="summarize_charts", mode="optimized")
    assert r.status_code == 200
    assert r.json()["options"]["image_mode"] == "markers" and r.json()["mode"] == "clean"


@pytest.mark.parametrize(
    "data",
    [
        {"mode": "turbo"},
        {"image_mode": "magia"},
        {"table_format": "html"},
        {"remove_duplicates": "quizas"},
    ],
)
def test_invalid_options_are_rejected_with_a_clear_message(data):
    r = post(fx.clean_pdf(), **data)
    assert r.status_code == 400
    assert isinstance(r.json()["detail"], str) and len(r.json()["detail"]) > 10


def test_describe_mode_is_rejected_not_faked():
    r = post(fx.clean_pdf(), image_mode="describe")
    assert r.status_code == 400
    assert "visión" in r.json()["detail"].lower()


def test_capabilities_report_what_is_really_available():
    caps = client.get("/capabilities").json()
    assert caps["vision"] is False and caps["image_modes"]["describe"] is False
    assert caps["image_modes"]["markers"] is True and caps["image_modes"]["extract"] is True
    assert caps["modes"] == ["faithful", "clean", "compact"]
    assert set(caps["table_formats"]) == {"auto", "markdown", "linear", "literal"}
    assert isinstance(caps["ocr"], bool) and caps["limits"]["max_upload_mb"] > 0


# ---------------------------------------------------------------------------
# Validación del archivo y errores seguros
# ---------------------------------------------------------------------------
def assert_safe_message(detail: str):
    assert isinstance(detail, str)
    lowered = detail.lower()
    for leaked in ("traceback", "markpdf_", "input.pdf", "\\users\\", "/tmp/", "site-packages", ".py", "exception"):
        assert leaked not in lowered, detail


def test_non_pdf_is_rejected():
    r = post(b"hola, esto es texto", filename="x.pdf")
    assert r.status_code == 400 and "no es un PDF" in r.json()["detail"]


def test_empty_file_is_rejected():
    r = post(b"", filename="x.pdf")
    assert r.status_code == 400 and "vacío" in r.json()["detail"]


def test_corrupt_pdf_gives_a_friendly_error_without_internals():
    r = post(fx.corrupt_pdf())
    assert r.status_code == 422
    assert "dañado" in r.json()["detail"]
    assert_safe_message(r.json()["detail"])


def test_encrypted_pdf_gives_a_friendly_error():
    r = post(fx.encrypted_pdf())
    assert r.status_code == 422 and "contraseña" in r.json()["detail"]
    assert_safe_message(r.json()["detail"])


def test_oversized_upload_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 1024)
    r = post(fx.contract_pdf(5))
    assert r.status_code == 413 and "límite" in r.json()["detail"]


def test_too_many_pages_is_rejected(monkeypatch):
    monkeypatch.setattr(main, "MAX_PAGES", 2)
    r = post(fx.contract_pdf(5))
    assert r.status_code == 413 and "5 páginas" in r.json()["detail"]


def test_unexpected_failures_return_a_generic_message_with_an_error_code(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secreto C:\\Users\\alguien\\input.pdf línea 42")

    monkeypatch.setattr(main, "extract_pages", boom)
    r = post(fx.clean_pdf())
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "No se pudo procesar el PDF" in detail and re.search(r"código [0-9a-f]{8}", detail)
    assert "secreto" not in detail
    assert_safe_message(detail)


def test_scanned_pdf_is_a_successful_response_with_warning_not_an_error():
    r = post(fx.scanned_pdf())
    assert r.status_code == 200
    body = r.json()
    assert body["markdown"] == "" and any("escaneada" in w for w in body["warnings"])


# ---------------------------------------------------------------------------
# Archivos temporales
# ---------------------------------------------------------------------------
def test_temp_dir_is_removed_after_a_successful_conversion(tmp_dirs):
    assert post(fx.contract_pdf(3)).status_code == 200
    assert tmp_dirs and not any(d.exists() for d in tmp_dirs)


@pytest.mark.parametrize("pdf", [fx.corrupt_pdf(), fx.encrypted_pdf()])
def test_temp_dir_is_removed_after_a_validation_error(tmp_dirs, pdf):
    assert post(pdf).status_code == 422
    assert tmp_dirs and not any(d.exists() for d in tmp_dirs)


def test_temp_dir_is_removed_after_an_unexpected_failure(tmp_dirs, monkeypatch):
    monkeypatch.setattr(main, "extract_pages", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("x")))
    assert post(fx.clean_pdf()).status_code == 500
    assert tmp_dirs and not any(d.exists() for d in tmp_dirs)


def test_temp_dir_is_removed_when_extract_mode_produces_no_images(tmp_dirs):
    assert post(fx.clean_pdf(), image_mode="extract").status_code == 200
    assert tmp_dirs and not any(d.exists() for d in tmp_dirs)


def test_extracted_images_are_kept_only_until_expiry_and_then_deleted(tmp_dirs):
    body = post(fx.logo_pdf(5, with_figure=True), image_mode="extract").json()
    job_id = body["job_id"]
    job_dir = Path(main.JOBS[job_id]["dir"])
    assert job_dir.exists() and not (job_dir / "input.pdf").exists()  # el PDF original no se conserva
    # El ZIP se puede descargar mientras el trabajo sigue vivo...
    r = client.post("/download", json={"markdown": body["markdown"], "filename": "doc.md", "job_id": job_id})
    assert r.status_code == 200 and r.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert any(n.startswith("images/") for n in zf.namelist())
    # ... y se borra al caducar.
    main.JOBS[job_id]["ts"] = time.time() - main.JOB_TTL_SECONDS - 5
    main._gc_jobs()
    assert job_id not in main.JOBS and not job_dir.exists()


def test_job_store_is_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MAX_JOBS", 2)
    monkeypatch.setattr(main, "JOBS", {})
    for i in range(4):
        d = tmp_path / f"job{i}"
        d.mkdir()
        main.JOBS[f"j{i}"] = {"dir": d, "ts": time.time() + i}
    main._gc_jobs()
    assert set(main.JOBS) == {"j2", "j3"}
    assert [p.name for p in sorted(tmp_path.iterdir())] == ["job2", "job3"]


def test_no_markpdf_directories_leak_in_the_system_temp_folder():
    before = {p.name for p in Path(tempfile.gettempdir()).glob("markpdf_*")}
    post(fx.contract_pdf(2))
    post(fx.corrupt_pdf())
    post(fx.scanned_pdf())
    after = {p.name for p in Path(tempfile.gettempdir()).glob("markpdf_*")}
    assert after <= before


# ---------------------------------------------------------------------------
# /download: lo que se descarga es exactamente lo que se envía (la vista previa editable)
# ---------------------------------------------------------------------------
def test_download_returns_exactly_the_edited_markdown():
    text = "# Título editado\n\nContenido con ñ y acentos: áéíóú\n"
    r = client.post("/download", json={"markdown": text, "filename": "Mi informe (final).pdf"})
    assert r.status_code == 200
    assert r.content.decode("utf-8") == text
    assert r.headers["content-type"].startswith("text/markdown")
    assert "Mi informe _final_.md" in r.headers["content-disposition"].replace("%20", " ") or "filename" in r.headers["content-disposition"]


def test_download_filename_cannot_escape_the_directory():
    r = client.post("/download", json={"markdown": "x", "filename": "../../etc/pass wd.md"})
    disposition = r.headers["content-disposition"]
    assert ".." not in disposition and "/" not in disposition.split("filename=")[1].split(";")[0]


def test_download_ignores_unknown_job_ids():
    r = client.post("/download", json={"markdown": "hola", "job_id": "no-existe"})
    assert r.status_code == 200 and r.content == b"hola"


def test_response_markdown_equals_what_a_client_would_copy_or_download():
    body = post(fx.contract_pdf(3)).json()
    r = client.post("/download", json={"markdown": body["markdown"], "filename": body["filename"]})
    assert r.content.decode("utf-8") == body["markdown"]


# ---------------------------------------------------------------------------
# Métricas expuestas
# ---------------------------------------------------------------------------
def test_metrics_are_labelled_and_internally_consistent_via_the_api():
    body = post(fx.contract_pdf(5)).json()
    m = body["metrics"]
    assert m["tokens_exact"] is False
    assert m["tokens"] == math.ceil(m["chars"] / 4)
    assert m["comparison"]["reference"] and m["comparison"]["formula"]
    assert m["tokens_before"] > m["tokens"]
    stats = body["stats"]
    assert stats["images_detail"]["detected"] >= stats["images"]


def test_the_ui_is_served_and_static_assets_mount():
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
