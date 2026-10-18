-- MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
-- Copyright (C) 2026 MarkPDF contributors
-- SPDX-License-Identifier: AGPL-3.0-or-later
--
-- Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
-- Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
-- Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

-- MarkPDF · migración 001 · usuarios y cuotas.
-- La aplica db.PostgresStore.init() una sola vez (queda anotada en schema_migrations).
-- Todos los "días" son días naturales en UTC: el contador se reinicia a las 00:00 UTC.

-- Usuarios registrados con Google. Solo se guarda lo mínimo.
CREATE TABLE users (
    id            BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    google_sub    TEXT        NOT NULL UNIQUE,   -- identificador estable de Google (no cambia aunque cambie el email)
    email         TEXT        NOT NULL,
    name          TEXT,
    is_active     BOOLEAN     NOT NULL DEFAULT TRUE,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_login_at TIMESTAMPTZ
);

-- Conversiones completadas por usuario y día. La clave primaria (user_id, day) es lo que permite
-- reservar cuota con un único INSERT ... ON CONFLICT DO UPDATE atómico.
CREATE TABLE daily_usage (
    user_id BIGINT  NOT NULL REFERENCES users (id) ON DELETE CASCADE,
    day     DATE    NOT NULL,
    used    INTEGER NOT NULL CHECK (used >= 0),
    PRIMARY KEY (user_id, day)
);
