# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Límite de hilos de ONNX Runtime (MARKPDF_ORT_THREADS): configuración, reversibilidad y que alcanza de verdad a las
sesiones de pymupdf-layout. Las comprobaciones con el motor real corren en subprocesos, porque las sesiones se
crean al importar `pymupdf.layout` y cada prueba necesita un proceso limpio con su propio entorno.
"""

import hashlib
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ort = pytest.importorskip("onnxruntime")

import ort_threads  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# Convierte un PDF de prueba y vuelca: sesiones ONNX vivas con sus opciones, y el hash del Markdown de cada perfil.
_PROBE = textwrap.dedent(
    """
    import gc, hashlib, json, sys
    sys.path.insert(0, ".")
    import main
    import onnxruntime as ort
    from tests import pdf_fixtures as fx

    hashes = {}
    for name, pdf in (("contract", fx.contract_pdf(3)), ("table", fx.table_pdf()), ("two_col", fx.two_column_pdf())):
        for mode in ("faithful", "clean", "compact"):
            md = main.convert_pdf(pdf, mode=mode)["markdown"]
            hashes[name + "/" + mode] = hashlib.sha256(md.encode()).hexdigest()
    sessions = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, ort.InferenceSession):
                o = obj.get_session_options()
                sessions.append({
                    "intra": o.intra_op_num_threads,
                    "inter": o.inter_op_num_threads,
                    "arena": o.enable_cpu_mem_arena,
                })
        except Exception:
            pass
    import ort_threads
    print(json.dumps({
        "patched": ort_threads.is_active(),
        "class_is_original": ort.InferenceSession.__module__.startswith("onnxruntime"),
        "sessions": sessions,
        "hashes": hashes,
    }))
    """
)


def _probe(threads=None) -> dict:
    env = {k: v for k, v in os.environ.items() if k != ort_threads.ENV_VAR}
    if threads is not None:
        env[ort_threads.ENV_VAR] = str(threads)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def default_run() -> dict:
    return _probe()


@pytest.fixture(scope="module")
def limited_run() -> dict:
    return _probe(2)


# ---------------------------------------------------------------------------
# Lectura de la variable
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw", ["", "  ", "0", "-1", "abc", "1.5", "2x"])
def test_invalid_or_missing_value_is_ignored(monkeypatch, raw):
    monkeypatch.setenv(ort_threads.ENV_VAR, raw)
    assert ort_threads.threads_from_env() is None


def test_valid_value(monkeypatch):
    monkeypatch.setenv(ort_threads.ENV_VAR, " 2 ")
    assert ort_threads.threads_from_env() == 2


def test_unset_variable_does_nothing(monkeypatch):
    monkeypatch.delenv(ort_threads.ENV_VAR, raising=False)
    before = ort.InferenceSession
    assert ort_threads.apply_from_env() is None
    assert ort.InferenceSession is before and not ort_threads.is_active()


# ---------------------------------------------------------------------------
# Parche y reversibilidad (en el proceso de pruebas, con un modelo ONNX diminuto)
# ---------------------------------------------------------------------------
def _tiny_model() -> str:
    """Un modelo ONNX pequeño que ya viene con pymupdf4llm (no hace falta el paquete `onnx`)."""
    pymupdf4llm = pytest.importorskip("pymupdf4llm")
    path = Path(pymupdf4llm.__file__).parent / "ocr" / "ocr_decision_model.onnx"
    if not path.exists():
        pytest.skip("modelo ONNX de ejemplo no disponible")
    return str(path)

@pytest.fixture
def patched():
    original = ort.InferenceSession
    assert ort_threads.apply(2)
    try:
        yield original
    finally:
        ort_threads.revert()
    assert ort.InferenceSession is original  # reversible: vuelve exactamente a la clase original


def test_patch_sets_threads_and_keeps_original_options(patched):
    model = _tiny_model()
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False  # opción propia de quien crea la sesión: debe conservarse
    session = ort.InferenceSession(model, sess_options=opts, providers=["CPUExecutionProvider"])
    got = session.get_session_options()
    assert (got.intra_op_num_threads, got.inter_op_num_threads) == (2, 1)
    assert got.enable_cpu_mem_arena is False
    # sin sess_options también se limita
    plain = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    assert plain.get_session_options().intra_op_num_threads == 2
    assert isinstance(plain, patched)  # sigue siendo una InferenceSession de verdad


def test_patch_respects_explicit_thread_settings(patched):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(_tiny_model(), opts, providers=["CPUExecutionProvider"])
    assert session.get_session_options().intra_op_num_threads == 1


def test_apply_twice_does_not_stack(patched):
    assert ort_threads.apply(1)
    session = ort.InferenceSession(_tiny_model(), providers=["CPUExecutionProvider"])
    assert session.get_session_options().intra_op_num_threads == 1
    assert ort.InferenceSession.__mro__[1] is patched  # una sola capa sobre la original


# ---------------------------------------------------------------------------
# Motor real (subprocesos): alcanza a las sesiones de pymupdf-layout y no cambia el Markdown
# ---------------------------------------------------------------------------
def test_default_run_is_untouched(default_run):
    assert default_run["patched"] is False
    assert default_run["class_is_original"] is True
    layout = [s for s in default_run["sessions"] if s["arena"] is False]  # firma de make_session de pymupdf-layout
    assert layout, "no se encontraron sesiones de pymupdf-layout"
    assert all(s["intra"] == 0 for s in layout)  # 0 = ONNX Runtime decide (comportamiento actual)


def test_limit_reaches_every_pymupdf_layout_session(limited_run):
    assert limited_run["patched"] is True
    layout = [s for s in limited_run["sessions"] if s["arena"] is False]
    assert layout, "no se encontraron sesiones de pymupdf-layout"
    assert all(s["intra"] == 2 and s["inter"] == 1 for s in layout), layout


def test_markdown_identical_with_and_without_limit(default_run, limited_run):
    assert default_run["hashes"] == limited_run["hashes"]
    assert len(default_run["hashes"]) == 9  # 3 PDFs x 3 perfiles
