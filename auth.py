# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Login con Google (OpenID Connect con Authlib) y sesión por cookie firmada.

Solo se activa con MARKPDF_AUTH_ENABLED=true (ver main.py). Rutas:

    GET  /auth/login      redirige a Google (Authlib genera `state`, `nonce` y PKCE y los guarda en la sesión)
    GET  /auth/callback   Google vuelve aquí; Authlib comprueba `state`, intercambia el código y valida el ID token
                          (firma con las claves públicas de Google, emisor, audiencia, caducidad y `nonce`)
    GET  /auth/me         sesión actual y conversiones restantes (la usa la interfaz)
    POST /auth/logout     borra la sesión

La sesión guarda solo {uid, email, name} en una cookie firmada con MARKPDF_SESSION_SECRET (HttpOnly, SameSite=Lax y,
en producción, Secure). Los datos persistentes del usuario (google_sub, email, nombre) viven en PostgreSQL.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from fastapi import APIRouter, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, RedirectResponse

from db import QuotaStoreError
from quotas import QuotaService

log = logging.getLogger("markpdf.auth")

GOOGLE_METADATA_URL = "https://accounts.google.com/.well-known/openid-configuration"
# Google firma sus ID tokens con uno de estos dos emisores (así lo documenta).
GOOGLE_ISSUERS = ["https://accounts.google.com", "accounts.google.com"]
SESSION_COOKIE = "mp_session"
SESSION_MAX_AGE = 7 * 24 * 3600  # 7 días
MIN_SECRET_LENGTH = 32
NO_STORE = {"Cache-Control": "no-store"}


def validate_settings(
    *, session_secret: str, client_id: str, client_secret: str, public_url: str, production: bool
) -> None:
    """Falla al arrancar, diciendo qué variable falta, en lugar de fallar con el primer usuario."""
    problems = []
    if len(session_secret) < MIN_SECRET_LENGTH:
        problems.append(f"MARKPDF_SESSION_SECRET (mínimo {MIN_SECRET_LENGTH} caracteres)")
    if not client_id:
        problems.append("GOOGLE_CLIENT_ID")
    if not client_secret:
        problems.append("GOOGLE_CLIENT_SECRET")
    if not public_url:
        problems.append("MARKPDF_PUBLIC_URL (por ejemplo https://tu-app.onrender.com)")
    if problems:
        raise RuntimeError("Con MARKPDF_AUTH_ENABLED=true faltan o no son válidas estas variables: " + ", ".join(problems))
    if production and not public_url.startswith("https://"):
        raise RuntimeError("En producción MARKPDF_PUBLIC_URL debe empezar por https://")
    if not public_url.startswith(("https://", "http://")):
        raise RuntimeError("MARKPDF_PUBLIC_URL debe empezar por https:// (o http:// en desarrollo local)")


def build_oauth(client_id: str, client_secret: str):
    """Cliente OpenID Connect de Google. PKCE (S256) además de state y nonce."""
    from authlib.integrations.starlette_client import OAuth  # noqa: PLC0415  (solo se necesita con login)

    oauth = OAuth()
    oauth.register(
        name="google",
        client_id=client_id,
        client_secret=client_secret,
        server_metadata_url=GOOGLE_METADATA_URL,
        client_kwargs={"scope": "openid email profile", "code_challenge_method": "S256"},
    )
    return oauth


def session_user_id(request: Request) -> Optional[int]:
    """Id de usuario de la sesión, o None. Exige SessionMiddleware (solo presente con login activado)."""
    uid = request.session.get("uid")
    return uid if isinstance(uid, int) and not isinstance(uid, bool) else None


def build_router(oauth, public_url: str, get_quotas: Callable[[], Optional[QuotaService]]) -> APIRouter:
    router = APIRouter()
    redirect_uri = f"{public_url}/auth/callback"

    def back(problem: Optional[str] = None) -> RedirectResponse:
        target = f"/?login={problem}" if problem else "/"
        return RedirectResponse(target, status_code=303, headers=NO_STORE)

    @router.get("/auth/login")
    async def login(request: Request):
        return await oauth.google.authorize_redirect(request, redirect_uri)

    @router.get("/auth/callback")
    async def callback(request: Request):
        try:
            token = await oauth.google.authorize_access_token(
                request, claims_options={"iss": {"essential": True, "values": GOOGLE_ISSUERS}}
            )
        except Exception as exc:  # noqa: BLE001 - state, nonce, firma, emisor, audiencia, caducidad, red...
            log.warning("Login con Google rechazado (%s)", type(exc).__name__)  # sin detalles ni datos del usuario
            return back("error")

        claims = token.get("userinfo")  # Authlib solo lo rellena si el ID token pasó TODAS las validaciones
        if not claims or not claims.get("sub") or not claims.get("email"):
            log.warning("Login con Google rechazado (ID token sin datos de usuario)")
            return back("error")
        if claims.get("email_verified") is not True:
            return back("unverified")

        service = get_quotas()
        if service is None:
            return back("unavailable")
        name = (claims.get("name") or "")[:100] or None
        try:
            uid = await run_in_threadpool(service.store.upsert_user, claims["sub"], claims["email"], name)
        except QuotaStoreError as exc:
            log.error("No se pudo guardar el usuario: %s", exc)
            return back("unavailable")

        request.session.clear()  # descarta el estado del flujo OAuth y cualquier sesión anterior
        request.session.update({"uid": uid, "email": claims["email"], "name": name or ""})
        return back()

    @router.get("/auth/me")
    def me(request: Request) -> JSONResponse:
        service = get_quotas()
        body: dict = {"enabled": True, "authenticated": False, "daily_limit": service.daily_limit if service else None}
        uid = session_user_id(request)
        if uid is not None and service is not None:
            body.update(authenticated=True, email=request.session.get("email", ""), name=request.session.get("name", ""))
            try:
                body["quota"] = service.user_status(uid)
            except QuotaStoreError as exc:
                log.error("No se pudo leer la cuota: %s", exc)
                body["quota"] = None  # la interfaz simplemente no muestra el contador
        return JSONResponse(body, headers=NO_STORE)

    @router.post("/auth/logout")
    def logout(request: Request) -> JSONResponse:
        request.session.clear()
        return JSONResponse({"ok": True}, headers=NO_STORE)

    return router
