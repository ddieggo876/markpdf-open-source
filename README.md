# MarkPDF

MarkPDF convierte PDF en Markdown estructurado y eficiente en tokens, pensado como contexto para modelos de lenguaje
(LLM). Es una aplicación web (FastAPI) con tres perfiles de conversión (`faithful`, `clean`, `compact`), tratamiento
de tablas, formularios e imágenes, un informe de lo que se elimina y métricas de tokens. La conversión es
determinista: no llama a ninguna IA ni servicio externo.

Opcionalmente incluye Google Login y cuotas diarias por usuario (PostgreSQL). Están desactivados por defecto.

## Requisitos

- Python 3.12 (Starlette 1.8 exige Python >= 3.11; las versiones fijadas se probaron con 3.12).
- `pip` y un entorno virtual.
- Opcional: PostgreSQL y una credencial OAuth de Google, solo si activas el login.
- Opcional: Tesseract (con sus datos de idioma) o `rapidocr`, solo para hacer OCR de páginas escaneadas.

## Instalación y ejecución local

```bash
git clone https://github.com/ddieggo876/markpdf-open-source.git markpdf
cd markpdf
python -m venv venv
source venv/bin/activate        # Windows (PowerShell): venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

Abre <http://127.0.0.1:8000>. Para ejecutar los tests:

```bash
pip install -r requirements-dev.txt
pytest
```

Los tests contra PostgreSQL real se omiten salvo que definas `TEST_DATABASE_URL` con una base **desechable** (sus tablas
se vacían en cada test).

## Variables de entorno

Todas son opcionales; sin ellas MarkPDF funciona en local sin login. En `.env.example` tienes la lista completa con
valores ficticios. La aplicación **no** lee archivos `.env` por sí sola: exporta las variables en tu shell, o cárgalas
con tu gestor de procesos o de despliegue.

| Variable | Para qué sirve | Por defecto |
|---|---|---|
| `MARKPDF_ENV` | `production` oculta `/docs`, añade HSTS y desactiva la extracción de imágenes. | vacío |
| `MARKPDF_MAX_UPLOAD_MB` | Tamaño máximo de cada PDF. | `10` |
| `MARKPDF_MAX_PAGES` | Páginas máximas por PDF. | `100` |
| `MARKPDF_MAX_DOWNLOAD_MB` | Texto máximo aceptado por `/download`. | `5` |
| `MARKPDF_MAX_CONCURRENT` | Conversiones simultáneas. | `1` |
| `MARKPDF_RATE_LIMIT_PER_HOUR` | Conversiones por IP y hora, solo sin login (`0` = sin límite). | `10` |
| `MARKPDF_TRUSTED_PROXY_HOPS` | Proxies de confianza delante de la app (para leer `X-Forwarded-For`). | `0` |
| `MARKPDF_ENABLE_IMAGE_EXTRACT` | Permite extraer imágenes a un `.zip`. | `true` en local, `false` en producción |
| `MARKPDF_AUTH_ENABLED` | Activa Google Login y cuotas (exige las cinco variables siguientes). | `false` |
| `DATABASE_URL` | Conexión a PostgreSQL (usuarios y cuotas). | vacío |
| `MARKPDF_SESSION_SECRET` | Firma de la cookie de sesión (mínimo 32 caracteres, aleatoria). | vacío |
| `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` | Credenciales OAuth de Google. | vacío |
| `MARKPDF_PUBLIC_URL` | URL pública de la app; Google debe tener `<URL>/auth/callback` como redirección autorizada. | vacío |
| `MARKPDF_DAILY_FREE_CONVERSIONS` | Conversiones gratuitas por usuario y día (UTC). | `3` |
| `MARKPDF_PERF_LOG` | `false` desactiva el log de tiempos y memoria de `/convert`. | activado |
| `MARKPDF_ORT_THREADS` | Limita los hilos de ONNX Runtime (útil en contenedores con poca CPU). | sin límite |

Genera un secreto de sesión con, por ejemplo: `python -c "import secrets; print(secrets.token_urlsafe(48))"`.

## Licencias de terceros

MarkPDF usa **PyMuPDF**, **PyMuPDF4LLM** y **PyMuPDF Layout** (Artifex Software), que se instalan como dependencias y
**no** se incluyen en este repositorio. Cada uno está sujeto a su propia licencia (doble licencia: GNU AGPL-3.0 o
licencia comercial de Artifex). Consulta sus condiciones antes de usarlos o distribuirlos. El resto de dependencias
(FastAPI, ONNX Runtime, NumPy, Authlib, etc.) tienen las suyas, listadas en `requirements.txt`.

La interfaz web carga desde CDN Tailwind CSS, marked y DOMPurify; no se distribuyen con este código y mantienen sus
propias licencias.

## Licencia

El código de MarkPDF se distribuye bajo la **GNU Affero General Public License, versión 3 o posterior
(AGPL-3.0-or-later)**. Consulta el archivo [LICENSE](LICENSE). Si modificas MarkPDF y lo ofreces como servicio en red,
la sección 13 de la AGPL te obliga a ofrecer el código fuente modificado a sus usuarios.

Copyright (C) 2026 MarkPDF contributors.
