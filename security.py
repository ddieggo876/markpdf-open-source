# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Seguridad básica de MarkPDF (sin servicios externos ni base de datos; todo en memoria del proceso).

  - `RateLimiter`: límite de usos por clave (IP) en una ventana de tiempo deslizante.
  - `client_ip`: IP del cliente SIN fiarse de cabeceras que pueda inventar el usuario.
  - `BodyLimitMiddleware`: corta las peticiones demasiado grandes mientras llegan, antes de guardarlas entera.
  - `SecurityHeadersMiddleware` / `build_security_headers`: cabeceras HTTP de seguridad.

No toca el motor de conversión.
"""

from __future__ import annotations

import ipaddress
import json
import math
import threading
import time
from collections import deque
from typing import Any, Awaitable, Callable, Mapping, Optional

# ---------------------------------------------------------------------------
# Límite de uso por clave (IP)
# ---------------------------------------------------------------------------


class RateLimiter:
    """
    Ventana deslizante en memoria: como mucho `max_hits` usos por clave cada `window_seconds`.
    `max_hits <= 0` desactiva el límite. Es por proceso: con varios procesos cada uno cuenta aparte.
    """

    def __init__(
        self,
        max_hits: int,
        window_seconds: float,
        *,
        max_tracked: int = 10_000,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.max_hits = max_hits
        self.window = window_seconds
        self.max_tracked = max_tracked  # tope de claves en memoria (evita crecer sin límite)
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.max_hits > 0

    def _wait(self, key: str, now: float) -> int:
        """Segundos que faltan para poder volver a usarlo (0 = permitido). Requiere el cerrojo."""
        hits = self._hits.get(key)
        if hits is None:
            return 0
        while hits and now - hits[0] >= self.window:
            hits.popleft()
        if not hits:
            del self._hits[key]
            return 0
        if len(hits) < self.max_hits:
            return 0
        return max(1, math.ceil(hits[0] + self.window - now))

    def retry_after(self, key: str) -> int:
        """Consulta sin registrar un uso. 0 = permitido; si no, segundos de espera."""
        if not self.enabled:
            return 0
        with self._lock:
            return self._wait(key, self._clock())

    def hit(self, key: str) -> int:
        """Registra un uso de forma atómica. Devuelve 0 si se permitió o los segundos de espera si no."""
        if not self.enabled:
            return 0
        with self._lock:
            now = self._clock()
            wait = self._wait(key, now)
            if wait:
                return wait
            if key not in self._hits and len(self._hits) >= self.max_tracked:
                self._evict(now)
            self._hits.setdefault(key, deque()).append(now)
            return 0

    def _evict(self, now: float) -> None:
        for key in list(self._hits):
            self._wait(key, now)  # descarta usos caducados y claves vacías
        if len(self._hits) >= self.max_tracked:
            oldest = min(self._hits, key=lambda k: self._hits[k][-1])
            del self._hits[oldest]

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


# ---------------------------------------------------------------------------
# IP del cliente
# ---------------------------------------------------------------------------
def normalize_ip(raw: Optional[str]) -> str:
    """IPv4 tal cual; IPv4 mapeada en IPv6 como IPv4; el resto de IPv6 agrupado por /64 (un usuario controla un /64)."""
    text = (raw or "").strip()
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return text or "unknown"
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped is not None:
            return str(ip.ipv4_mapped)
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip)


def client_ip(peer: Optional[str], forwarded_for: Optional[str], trusted_hops: int) -> str:
    """
    IP del cliente para el límite de uso.

    - `trusted_hops = 0` (por defecto): se IGNORA `X-Forwarded-For` (cualquiera puede enviarlo) y se usa la IP
      del socket. Es lo correcto si la app está expuesta directamente a internet.
    - `trusted_hops = N > 0`: hay N proxies de confianza delante (p. ej. el balanceador de la plataforma). Cada
      proxy AÑADE al final de `X-Forwarded-For` la IP desde la que le llegó la petición, así que la única entrada
      fiable es la N-ésima contando desde la DERECHA. Lo que el usuario escriba a la izquierda no cuenta.
    Si la cabecera no tiene esa entrada o no es una IP válida, se usa la IP del socket.
    """
    if trusted_hops > 0 and forwarded_for:
        parts = [p.strip() for p in forwarded_for.split(",") if p.strip()]
        if len(parts) >= trusted_hops:
            candidate = parts[-trusted_hops]
            try:
                ipaddress.ip_address(candidate)
            except ValueError:
                pass
            else:
                return normalize_ip(candidate)
    return normalize_ip(peer)


# ---------------------------------------------------------------------------
# Límite de tamaño del cuerpo de la petición
# ---------------------------------------------------------------------------
Scope = dict
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]

TOO_LARGE_MESSAGE = "La petición es demasiado grande."


async def _send_json(send: Send, status: int, detail: str, *, close: bool = False) -> None:
    body = json.dumps({"detail": detail}, ensure_ascii=False).encode("utf-8")
    headers = [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]
    if close:
        headers.append((b"connection", b"close"))
    await send({"type": "http.response.start", "status": status, "headers": headers})
    await send({"type": "http.response.body", "body": body})


class BodyLimitMiddleware:
    """
    Rechaza con 413 las peticiones POST cuyo cuerpo supere el máximo de su ruta, SIN esperar a recibirlas enteras:
      1. Si `Content-Length` ya declara demasiado, responde sin leer nada.
      2. Si no lo declara (o miente), cuenta los bytes según llegan y corta al pasarse.
    `limits` devuelve {ruta: bytes} y se consulta en cada petición (así los límites se pueden configurar).
    Es un freno de seguridad grueso: la validación exacta y los mensajes finos los da cada endpoint.
    """

    def __init__(self, app: Any, limits: Callable[[], Mapping[str, int]]) -> None:
        self.app = app
        self.limits = limits

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return
        cap = self.limits().get(scope.get("path", ""))
        if cap is None:
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    break  # se contará igualmente según llegue
                if declared > cap:
                    await _send_json(send, 413, TOO_LARGE_MESSAGE, close=True)
                    return
                break

        state = {"seen": 0, "rejected": False}

        async def limited_receive() -> dict:
            message = await receive()
            if message["type"] == "http.request":
                state["seen"] += len(message.get("body", b""))
                if state["seen"] > cap:
                    if not state["rejected"]:
                        state["rejected"] = True
                        await _send_json(send, 413, TOO_LARGE_MESSAGE, close=True)
                    return {"type": "http.disconnect"}  # la app deja de leer
            return message

        async def guarded_send(message: dict) -> None:
            if state["rejected"]:
                return  # ya respondimos 413; lo que la app intente enviar después se descarta
            await send(message)

        try:
            await self.app(scope, limited_receive, guarded_send)
        except Exception:
            if not state["rejected"]:
                raise


# ---------------------------------------------------------------------------
# Cabeceras de seguridad
# ---------------------------------------------------------------------------
# La interfaz es un único index.html con scripts y estilos en línea, y carga Tailwind, marked y DOMPurify desde
# dos CDN: por eso la política permite 'unsafe-inline' y esos dos orígenes. Es más estricta en todo lo demás.
CONTENT_SECURITY_POLICY = "; ".join(
    [
        "default-src 'self'",
        "script-src 'self' 'unsafe-inline' https://cdn.tailwindcss.com https://cdn.jsdelivr.net",
        "style-src 'self' 'unsafe-inline'",
        "img-src 'self' data: blob:",
        "font-src 'self' data:",
        "connect-src 'self'",
        "object-src 'none'",
        "base-uri 'self'",
        "form-action 'self'",
        "frame-ancestors 'none'",
    ]
)


def build_security_headers(production: bool) -> list[tuple[bytes, bytes]]:
    headers = {
        "Content-Security-Policy": CONTENT_SECURITY_POLICY,
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "Referrer-Policy": "no-referrer",
        "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    }
    if production:
        headers["Strict-Transport-Security"] = "max-age=31536000"  # solo tiene efecto sobre HTTPS
    return [(k.lower().encode("latin-1"), v.encode("latin-1")) for k, v in headers.items()]


class SecurityHeadersMiddleware:
    """Añade las cabeceras de seguridad a todas las respuestas HTTP (sin pisar las que ya existan).
    Las respuestas a POST (el texto del documento) llevan además `Cache-Control: no-store`."""

    def __init__(self, app: Any, headers: list[tuple[bytes, bytes]]) -> None:
        self.app = app
        self.headers = headers

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        extra = list(self.headers)
        if scope.get("method") == "POST":
            extra.append((b"cache-control", b"no-store"))

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                current = list(message.get("headers", []))
                present = {name.lower() for name, _ in current}
                current.extend((k, v) for k, v in extra if k not in present)
                message = {**message, "headers": current}
            await send(message)

        await self.app(scope, receive, send_with_headers)
