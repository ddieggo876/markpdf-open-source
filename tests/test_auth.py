# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Google Login (OpenID Connect con Authlib), sesión y acceso a POST /convert.

No se habla con Google: un «Google falso» sustituye SOLO las tres llamadas de red de Authlib (metadatos, claves
públicas y canje del código) y firma ID tokens reales con una clave RSA. Todo lo demás es el código real de Authlib:
comprobación de `state`, `nonce`, firma, emisor, audiencia y caducidad.
"""

from __future__ import annotations

import base64
import json
import time
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import itsdangerous
import pytest
from authlib.integrations.starlette_client.apps import StarletteOAuth2App
from authlib.oidc.core.util import create_half_hash
from fastapi.testclient import TestClient
from joserfc import jwt
from joserfc.jwk import RSAKey

from db import MemoryStore, QuotaStoreError
from quotas import CODE_HEADER, QuotaService
from tests.helpers import ROOT, load_main_copy

CLIENT_ID = "test-client.apps.googleusercontent.com"
SESSION_SECRET = "k" * 40
PUBLIC_URL = "https://markpdf.example.com"
ISSUER = "https://accounts.google.com"
DELETE = object()  # «quita esta claim del ID token»

SETTINGS = {
    "MARKPDF_AUTH_ENABLED": "true",
    "MARKPDF_SESSION_SECRET": SESSION_SECRET,
    "GOOGLE_CLIENT_ID": CLIENT_ID,
    "GOOGLE_CLIENT_SECRET": "client-secret",
    "MARKPDF_PUBLIC_URL": PUBLIC_URL,
}


class FakeGoogle:
    def __init__(self, monkeypatch):
        params = {"kid": "k1", "use": "sig", "alg": "RS256"}
        self.key = RSAKey.generate_key(2048, parameters=params)
        self.other_key = RSAKey.generate_key(2048, parameters=params)  # mismo kid, otra clave: firma falsa
        self.sign_with = self.key
        self.nonce = None  # el nonce que Authlib envió a Google en el último /auth/login
        self.claims: dict = {}  # sobrescrituras de claims (DELETE para quitarlas)
        self.omit_id_token = False
        self.token_error = None
        self.fetch_calls: list[dict] = []
        self.metadata = {
            "issuer": ISSUER,
            "authorization_endpoint": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_endpoint": "https://oauth2.googleapis.com/token",
            "jwks_uri": "https://www.googleapis.com/oauth2/v3/certs",
            "jwks": {"keys": [self.key.as_dict(private=False)]},  # con "jwks" Authlib no hace ninguna petición
            "id_token_signing_alg_values_supported": ["RS256"],
        }
        monkeypatch.setattr(StarletteOAuth2App, "load_server_metadata", self._load_metadata)
        monkeypatch.setattr(StarletteOAuth2App, "fetch_access_token", self._fetch_access_token)

    async def _load_metadata(self):
        return self.metadata

    async def _fetch_access_token(self, **params):
        self.fetch_calls.append(params)
        if self.token_error:
            raise self.token_error
        access_token = "access-token-123"
        now = int(time.time())
        claims = {
            "iss": ISSUER,
            "aud": CLIENT_ID,
            "sub": "google-sub-123",
            "email": "ana@example.com",
            "email_verified": True,
            "name": "Ana García",
            "iat": now,
            "exp": now + 3600,
            "nonce": self.nonce,
            "at_hash": create_half_hash(access_token, "RS256").decode(),
        }
        for name, value in self.claims.items():
            if value is DELETE:
                claims.pop(name, None)
            else:
                claims[name] = value
        token = {"access_token": access_token, "token_type": "Bearer", "expires_in": 3600}
        if not self.omit_id_token:
            token["id_token"] = jwt.encode({"alg": "RS256", "kid": "k1"}, claims, self.sign_with)
        return token


@pytest.fixture
def google(monkeypatch):
    return FakeGoogle(monkeypatch)


@pytest.fixture
def env(monkeypatch, google):
    m = load_main_copy(monkeypatch, **SETTINGS)
    monkeypatch.setattr(m, "convert_pdf", lambda data, options=None: {"markdown": "# ok"})
    with TestClient(m.app, raise_server_exceptions=False) as client:  # `with` ejecuta el arranque (crea QUOTAS)
        yield SimpleNamespace(m=m, client=client, google=google)


# -- utilidades ------------------------------------------------------------------------------------
PDF = ("doc.pdf", b"%PDF-1.4\n%fake\n", "application/pdf")


def convert(client):
    return client.post("/convert", files={"file": PDF})


def start_login(env):
    r = env.client.get("/auth/login", follow_redirects=False)
    assert r.status_code == 302
    query = {k: v[0] for k, v in parse_qs(urlparse(r.headers["location"]).query).items()}
    env.google.nonce = query["nonce"]
    return query


def finish_login(env, state, **extra):
    return env.client.get("/auth/callback", params={"code": "auth-code", "state": state, **extra}, follow_redirects=False)


def login(env):
    query = start_login(env)
    return query, finish_login(env, query["state"])


def me(env):
    return env.client.get("/auth/me").json()


def assert_login_failed(env, response, problem="error"):
    assert response.status_code == 303 and response.headers["location"] == f"/?login={problem}"
    assert me(env)["authenticated"] is False
    assert convert(env.client).status_code == 401
    assert env.m.QUOTAS.store._users == {}  # no se creó ningún usuario


# ===========================================================================
# Inicio del flujo
# ===========================================================================
def test_login_redirects_to_google_with_state_nonce_and_pkce(env):
    query = start_login(env)
    assert query["client_id"] == CLIENT_ID
    assert query["response_type"] == "code"
    assert query["redirect_uri"] == f"{PUBLIC_URL}/auth/callback"
    assert set(query["scope"].split()) == {"openid", "email", "profile"}
    assert len(query["state"]) >= 16 and len(query["nonce"]) >= 16
    assert query["code_challenge_method"] == "S256" and query["code_challenge"]
    again = start_login(env)
    assert (again["state"], again["nonce"]) != (query["state"], query["nonce"])  # nunca se reutilizan


# ===========================================================================
# Login correcto, sesión y cuota
# ===========================================================================
def test_successful_login_creates_user_session_and_quota(env):
    assert me(env) == {"enabled": True, "authenticated": False, "daily_limit": 3}
    query, response = login(env)
    assert response.status_code == 303 and response.headers["location"] == "/"

    session = me(env)
    assert session["authenticated"] is True
    assert session["email"] == "ana@example.com" and session["name"] == "Ana García"
    assert session["quota"]["limit"] == 3 and session["quota"]["remaining"] == 3
    assert env.m.QUOTAS.store.user_is_active(1)

    # El canje del código llevó el verificador PKCE y la misma redirect_uri
    call = env.google.fetch_calls[0]
    assert call["code"] == "auth-code" and call["code_verifier"]
    assert call["redirect_uri"] == f"{PUBLIC_URL}/auth/callback"

    assert convert(env.client).status_code == 200
    assert me(env)["quota"]["remaining"] == 2


def test_second_login_updates_the_same_user(env):
    login(env)
    env.google.claims = {"email": "nuevo@example.com", "name": "Ana B"}
    login(env)
    assert list(env.m.QUOTAS.store._users) == [1]
    assert env.m.QUOTAS.store._users[1]["email"] == "nuevo@example.com"
    assert me(env)["email"] == "nuevo@example.com"


@pytest.mark.parametrize("issuer", ["https://accounts.google.com", "accounts.google.com"])
def test_both_google_issuer_forms_are_accepted(env, issuer):
    env.google.claims = {"iss": issuer}
    _, response = login(env)
    assert response.headers["location"] == "/" and me(env)["authenticated"] is True


def test_quota_runs_out_after_three_and_logout_removes_access(env):
    login(env)
    assert [convert(env.client).status_code for _ in range(3)] == [200, 200, 200]
    denied = convert(env.client)
    assert denied.status_code == 429 and denied.headers[CODE_HEADER] == "daily_quota_exhausted"
    assert me(env)["quota"]["remaining"] == 0

    out = env.client.post("/auth/logout")
    assert out.status_code == 200 and out.json() == {"ok": True}
    assert me(env)["authenticated"] is False
    assert convert(env.client).status_code == 401  # sin sesión ya no se puede, ni siquiera con cuota


def test_logout_without_session_is_harmless(env):
    assert env.client.post("/auth/logout").status_code == 200


def test_login_again_after_logout_keeps_the_same_daily_usage(env):
    login(env)
    convert(env.client)
    env.client.post("/auth/logout")
    login(env)
    assert me(env)["quota"]["used"] == 1  # cerrar sesión no reinicia la cuota


def test_ip_limit_is_off_with_login_enabled(env):
    env.m.RATE_LIMITER.max_hits = 1
    env.m.RATE_LIMITER.reset()
    login(env)
    assert [convert(env.client).status_code for _ in range(3)] == [200, 200, 200]


def test_disabling_a_user_blocks_them_even_with_a_valid_session(env):
    login(env)
    env.m.QUOTAS.store.set_user_active(1, False)
    r = convert(env.client)
    assert r.status_code == 403 and r.headers[CODE_HEADER] == "account_disabled"


def test_auth_endpoints_are_not_cached(env):
    assert env.client.get("/auth/me").headers["cache-control"] == "no-store"
    assert env.client.post("/auth/logout").headers["cache-control"] == "no-store"


# ===========================================================================
# Acceso directo a POST /convert
# ===========================================================================
def test_direct_convert_without_session_is_401(env):
    r = convert(env.client)
    assert r.status_code == 401 and r.headers[CODE_HEADER] == "login_required"
    assert "Inicia sesión" in r.json()["detail"]


def forged_cookie(secret, data):
    payload = base64.b64encode(json.dumps(data).encode("utf-8"))
    return itsdangerous.TimestampSigner(secret).sign(payload).decode()


@pytest.mark.parametrize(
    "cookie",
    [
        "basura",
        base64.b64encode(json.dumps({"uid": 1}).encode()).decode(),  # sin firma
        forged_cookie("otro-secreto-" + "x" * 30, {"uid": 1}),  # firmada con otro secreto
        "",
    ],
    ids=["basura", "sin-firma", "otro-secreto", "vacia"],
)
def test_forged_session_cookies_are_rejected(env, cookie):
    env.m.QUOTAS.store.upsert_user("google-sub-123", "ana@example.com", "Ana")  # el uid 1 existe
    env.client.cookies.set("mp_session", cookie)
    assert convert(env.client).status_code == 401
    assert me(env)["authenticated"] is False


def test_tampered_session_cookie_is_rejected(env):
    login(env)
    value = env.client.cookies.get("mp_session")
    env.client.cookies.set("mp_session", value[:-3] + ("AAA" if not value.endswith("AAA") else "BBB"))
    assert me(env)["authenticated"] is False
    assert convert(env.client).status_code == 401


def test_session_with_a_non_integer_user_id_is_not_a_session(env):
    for bad in ("1", True, None, [1], 1.5):
        env.client.cookies.set("mp_session", forged_cookie(SESSION_SECRET, {"uid": bad}))
        assert convert(env.client).status_code == 401, bad


# ===========================================================================
# El login se rechaza si el ID token o el flujo no son válidos
# ===========================================================================
@pytest.mark.parametrize(
    "claims",
    [
        {"aud": "otra-aplicacion.apps.googleusercontent.com"},  # audiencia ajena
        {"aud": DELETE},
        {"iss": "https://evil.example.com"},  # emisor falso
        {"iss": DELETE},
        {"nonce": "nonce-de-otro-flujo"},  # reutilización / inyección de token
        {"nonce": DELETE},
        {"exp": int(time.time()) - 3600},  # caducado
        {"exp": DELETE},
        {"sub": DELETE},
        {"email": DELETE},
        {"at_hash": "hash-incorrecto"},
    ],
    ids=["aud-ajena", "sin-aud", "iss-falso", "sin-iss", "nonce-otro", "sin-nonce", "caducado", "sin-exp",
         "sin-sub", "sin-email", "at_hash-malo"],
)
def test_invalid_id_token_claims_do_not_log_in(env, claims):
    env.google.claims = claims
    _, response = login(env)
    assert_login_failed(env, response)


def test_id_token_signed_with_another_key_is_rejected(env):
    env.google.sign_with = env.google.other_key
    _, response = login(env)
    assert_login_failed(env, response)


def test_response_without_id_token_is_rejected(env):
    env.google.omit_id_token = True
    _, response = login(env)
    assert_login_failed(env, response)


def test_token_endpoint_failure_is_rejected(env):
    env.google.token_error = RuntimeError("Google no responde")
    _, response = login(env)
    assert_login_failed(env, response)


def test_wrong_state_is_rejected_without_exchanging_the_code(env):
    query = start_login(env)
    response = finish_login(env, query["state"] + "x")
    assert_login_failed(env, response)
    assert env.google.fetch_calls == []  # ni siquiera se canjeó el código


def test_callback_without_a_prior_login_is_rejected(env):
    response = env.client.get("/auth/callback", params={"code": "c", "state": "inventado"}, follow_redirects=False)
    assert_login_failed(env, response)
    assert env.google.fetch_calls == []


def test_callback_without_state_is_rejected(env):
    start_login(env)
    response = env.client.get("/auth/callback", params={"code": "c"}, follow_redirects=False)
    assert_login_failed(env, response)


def test_state_cannot_be_replayed(env):
    query, first = login(env)
    assert first.headers["location"] == "/"
    replay = finish_login(env, query["state"])
    assert replay.headers["location"] == "/?login=error"
    assert len(env.google.fetch_calls) == 1  # el segundo intento no llegó a canjear nada


def test_state_from_another_browser_is_rejected(env):
    query = start_login(env)  # el `state` vive en la sesión del navegador que inició el login
    attacker = TestClient(env.m.app, raise_server_exceptions=False)
    response = attacker.get("/auth/callback", params={"code": "c", "state": query["state"]}, follow_redirects=False)
    assert response.headers["location"] == "/?login=error"
    assert attacker.get("/auth/me").json()["authenticated"] is False


def test_user_cancelling_at_google_is_handled(env):
    query = start_login(env)
    response = finish_login(env, query["state"], error="access_denied")
    assert_login_failed(env, response)
    assert env.google.fetch_calls == []


@pytest.mark.parametrize("verified", [False, "true", None, DELETE])
def test_unverified_email_is_rejected(env, verified):
    env.google.claims = {"email_verified": verified}
    _, response = login(env)
    assert_login_failed(env, response, "unverified")


def test_database_down_during_login_does_not_create_a_session(env, monkeypatch):
    def down(*a, **k):
        raise QuotaStoreError("base de datos caída")

    monkeypatch.setattr(env.m.QUOTAS.store, "upsert_user", down)
    _, response = login(env)
    assert response.headers["location"] == "/?login=unavailable"
    assert me(env)["authenticated"] is False


def test_me_survives_a_database_failure(env, monkeypatch):
    login(env)

    def down(*a, **k):
        raise QuotaStoreError("base de datos caída")

    monkeypatch.setattr(env.m.QUOTAS.store, "usage_user", down)
    body = me(env)
    assert body["authenticated"] is True and body["quota"] is None


def test_login_failure_logs_do_not_contain_user_data(env, caplog):
    env.google.claims = {"aud": "otra"}
    login(env)
    assert "ana@example.com" not in caplog.text and "google-sub-123" not in caplog.text


# ===========================================================================
# Cookie de sesión
# ===========================================================================
def test_session_cookie_flags_in_development(env):
    _, response = login(env)
    cookie = response.headers["set-cookie"].lower()
    assert "mp_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert "secure" not in cookie  # en local se usa http


def test_session_cookie_is_secure_httponly_and_lax_in_production(monkeypatch, google):
    m = load_main_copy(monkeypatch, MARKPDF_ENV="production", **SETTINGS)
    monkeypatch.setattr(m, "convert_pdf", lambda data, options=None: {"markdown": "# ok"})
    monkeypatch.setattr(m, "QUOTAS", QuotaService(MemoryStore()))  # en producción el arranque exige PostgreSQL
    client = TestClient(m.app, base_url=PUBLIC_URL, raise_server_exceptions=False)
    env = SimpleNamespace(m=m, client=client, google=google)
    r = client.get("/auth/login", follow_redirects=False)
    assert "secure" in r.headers["set-cookie"].lower()  # incluso la cookie del flujo OAuth
    _, response = login(env)
    cookie = response.headers["set-cookie"].lower()
    assert "secure" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert convert(client).status_code == 200


# ===========================================================================
# Configuración al arrancar y comportamiento con el flag desactivado
# ===========================================================================
@pytest.mark.parametrize("missing", ["MARKPDF_SESSION_SECRET", "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET", "MARKPDF_PUBLIC_URL"])
def test_missing_setting_stops_the_startup(monkeypatch, missing):
    settings = {k: v for k, v in SETTINGS.items() if k != missing}
    with pytest.raises(RuntimeError, match=missing):
        load_main_copy(monkeypatch, **settings)


def test_short_session_secret_stops_the_startup(monkeypatch):
    with pytest.raises(RuntimeError, match="MARKPDF_SESSION_SECRET"):
        load_main_copy(monkeypatch, **{**SETTINGS, "MARKPDF_SESSION_SECRET": "corto"})


def test_production_requires_an_https_public_url(monkeypatch):
    with pytest.raises(RuntimeError, match="https://"):
        load_main_copy(monkeypatch, MARKPDF_ENV="production", **{**SETTINGS, "MARKPDF_PUBLIC_URL": "http://markpdf.example.com"})


def test_production_requires_a_database(monkeypatch):
    m = load_main_copy(monkeypatch, MARKPDF_ENV="production", **SETTINGS)
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with TestClient(m.app):
            pass


def test_trailing_slash_in_public_url_is_ignored(monkeypatch, google):
    m = load_main_copy(monkeypatch, **{**SETTINGS, "MARKPDF_PUBLIC_URL": PUBLIC_URL + "/"})
    with TestClient(m.app) as client:
        r = client.get("/auth/login", follow_redirects=False)
    assert parse_qs(urlparse(r.headers["location"]).query)["redirect_uri"] == [f"{PUBLIC_URL}/auth/callback"]


def test_disabled_flag_has_no_login_routes_and_no_session_middleware(monkeypatch, google):
    m = load_main_copy(monkeypatch)
    paths = {getattr(r, "path", "") for r in m.app.routes}
    assert not {"/auth/login", "/auth/callback", "/auth/logout"} & paths
    assert all(mw.cls.__name__ != "SessionMiddleware" for mw in m.app.user_middleware)


# ===========================================================================
# Interfaz: solo los elementos pedidos y la guarda de sesión
# ===========================================================================
def test_ui_has_the_login_elements_and_never_uploads_before_checking_the_session():
    html = (ROOT / "index.html").read_text(encoding="utf-8")
    for text in ("Continuar con Google", "Cerrar sesión", "/auth/login", "/auth/me", "/auth/logout", "quotaCounter"):
        assert text in html
    convert_body = html.split("async function convert(", 1)[1].split("xhr.open(", 1)[0]
    assert "await authReady" in convert_body
    assert convert_body.index("await authReady") < convert_body.index("auth.authenticated")
    assert "Te quedan" in html and "conversiones hoy" in html
