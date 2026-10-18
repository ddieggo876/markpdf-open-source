# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""
Limpieza determinista de Markdown extraído de PDFs (sin IA, sin dependencias externas).

Principios (por orden de prioridad):
  1. Fidelidad al documento original: no se cambian cifras, fechas, nombres, importes ni negaciones.
  2. Orden de lectura (nunca se reordenan bloques; solo se unen líneas con evidencia).
  3. Estructura semántica (encabezados, listas, tablas y bloques de código se respetan).
  4. Reducción de ruido, solo cuando se identifica con confianza.
  5. Reducción de tokens.

Un único motor con políticas configurables (`options.ConvertOptions`). Los perfiles solo fijan valores
por defecto:
  - "faithful": caracteres de control, espacios repetidos y rellenos. Conserva los saltos de línea del PDF.
  - "clean"   : además quita cabeceras/pies/numeración detectados, une líneas partidas con evidencia y
                elimina bloques duplicados consecutivos exactos.
  - "compact" : "clean" + reducción de ruido de formato Markdown (relleno de tablas, negritas redundantes,
                líneas en blanco entre elementos de lista).
Las opciones que DESCARTAN contenido (`omit_empty_fields`, `omit_redundant_headings`,
`remove_repeated_blocks`) están desactivadas en todos los perfiles y solo actúan si se piden.

Campos en blanco: una secuencia larga de puntos o guiones bajos en un formulario o contrato es un CAMPO
SIN RELLENAR, que es información ("fecha ............" ≠ "fecha"). Se normaliza a `___` (o a
`[campo en blanco]` si ocupa una línea entera) en lugar de borrarse. Los índices ("Capítulo ..... 12") y las
rayas decorativas ("-----") sí se simplifican.

Punto de entrada principal: ``clean_markdown(md, mode)``; para documentos paginados
``clean_pages(pages, mode, margin_lines)`` (permite unir párrafos partidos entre páginas).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from metrics import count_words, estimate_tokens  # noqa: F401  (se reexportan por compatibilidad)
from options import DEFAULT_MODE, MODES, ConvertOptions

BLANK_MARK = "___"
BLANK_LINE = "[campo en blanco]"
MARKER_LINE_RE = re.compile(
    r"^\[(?:Imagen|Dibujo vectorial|Elemento visual|Página sin texto|Tabla:|campo en blanco)[^\n]*\]$"
)
MARKER_PREFIX = "[Imagen"  # compatibilidad: los marcadores de MarkPDF nunca se fusionan


# ---------------------------------------------------------------------------
# 1) Caracteres: control, invisibles, espacios
# ---------------------------------------------------------------------------
_CONTROL_RE = re.compile(r"[\x00-\x08\x0e-\x1f\x7f-\x9f]")
_FORM_FEED_RE = re.compile(r"[\x0b\x0c]")
_INVISIBLE_RE = re.compile("[\u200b\u2060\ufeff]")  # espacio de ancho cero, word joiner, BOM
_SPACES_RE = re.compile("[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]")
_SOFT_HYPHEN = "\u00ad"
_FULL_MARK = "\u200b"  # lo añade lines.py a las filas que llegan al borde derecho de su columna
_SOFT_HYPHEN_EOL_RE = re.compile(f"{_SOFT_HYPHEN}[ \\t\u200b]*\\n[ \\t]*(?=\\S)")


def normalize_chars(text: str) -> str:
    """Elimina caracteres de control/invisibles y unifica saltos de línea y espacios Unicode."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _FORM_FEED_RE.sub("\n", text)
    text = _CONTROL_RE.sub("", text)
    text = _INVISIBLE_RE.sub("", text)
    text = _SPACES_RE.sub(" ", text)
    return text


# ---------------------------------------------------------------------------
# 2) Secuencias y líneas de relleno (puntos guía, rayas para rellenar formularios)
# ---------------------------------------------------------------------------
# Umbrales deliberadamente altos: "...", "…" y "...." son puntuación legítima.
_DOTS = r"\.(?:[ \t]?\.){5,}"  # >= 6 puntos (admite un espacio entre ellos)
_LONG_DOTS = r"\.(?:[ \t]?\.){9,}"  # >= 10 puntos (típico de índices)
_ELLIPSIS = "…(?:[ \t]?…){2,}"  # >= 3 caracteres "…" seguidos
_MIDDOTS = "·(?:[ \t]?·){5,}"
_UNDERSCORES = r"(?:\\?_)(?:[ \t]?(?:\\?_)){3,}"  # >= 4 guiones bajos (a veces escapados)
_LEADER = "|".join([_DOTS, _ELLIPSIS, _MIDDOTS, _UNDERSCORES])

LEADER_RE = re.compile(_LEADER)
# El relleno que precede a una puntuación absorbe su espacio previo: "fecha ......, y" -> "fecha ___, y".
# (?![._…·]) obliga a consumir la secuencia completa: sin él, el motor de expresiones
# regulares retrocede y deja el último punto como "puntuación" siguiente.
_LEADER_BEFORE_PUNCT_RE = re.compile(rf"[ \t]*(?:{_LEADER})(?![._…·\\])[ \t]*(?=[,;:.)\]»”’\"'])")
_LEADER_AT_START_RE = re.compile(rf"^[ \t*>]*(?:{_LEADER})")
_TOC_RE = re.compile(
    rf"^(?P<head>[^\n]*?\S)[ \t]*(?:{_LONG_DOTS})[ \t]*(?P<num>\d{{1,4}}|[ivxlcdm]{{1,7}}|[IVXLCDM]{{1,7}})$"
)
_DASH_FILL_RE = re.compile(r"-(?:[ \t]?-){9,}|=(?:[ \t]?=){9,}")  # >= 10 guiones o signos "="
# Negrita/cursiva que se quedó vacía porque el relleno que contenía se normalizó: **......** -> ___.
_EMPHASIS_AROUND_FILLER_RE = re.compile(rf"(\*\*|__)[ \t]*(?:{_LEADER})[ \t]*\1")


def _filler_weight(core: str) -> int:
    return sum(3 if ch == "…" else 1 for ch in core)


def _filler_kind(line: str) -> Optional[str]:
    """'blank' (puntos/guiones bajos: campo sin rellenar) o 'rule' (rayas decorativas) si la línea es solo relleno."""
    if line.lstrip().startswith("|"):
        return None
    core = re.sub(r"[\s*\\`>#]", "", line)
    if not core:
        return None
    if re.fullmatch(r"[._…·]+", core):
        return "blank" if _filler_weight(core) >= 6 else None
    if re.fullmatch(r"[-=]+", core):
        return "rule" if len(core) >= 10 else None
    return None


def _collapse_spaces(line: str) -> str:
    """Colapsa espacios repetidos conservando la sangría inicial; quita espacios finales."""
    match = re.match(r"^([ \t]*)(.*?)[ \t]*$", line, re.DOTALL)
    indent, body = match.group(1), match.group(2)
    return indent + re.sub(r"[ \t]+", " ", body)


@dataclass
class _Line:
    text: str
    leader_start: bool = False  # la línea original empezaba con relleno (continúa la frase anterior)
    blank_only: bool = False  # la línea original era solo relleno (campo sin rellenar)
    full: bool = False  # la línea llegaba al borde derecho de su columna (evidencia geométrica de ajuste)


def _clean_line(line: str) -> tuple[Optional[str], bool, int, bool]:
    """
    Limpia artefactos de una línea (fuera de bloques de código).
    Devuelve (texto o None si se descarta, empezaba_con_relleno, nº de rellenos tratados, es_solo_relleno).
    """
    if not line.strip():
        return "", False, 0, False
    kind = _filler_kind(line)
    if kind == "rule":
        return None, False, 1, False
    if kind == "blank":
        return BLANK_MARK, False, 1, True

    is_table = line.lstrip().startswith("|")
    removed = len(LEADER_RE.findall(line))
    if not is_table:
        removed += len(_DASH_FILL_RE.findall(line))
    if removed == 0:
        return _collapse_spaces(line), False, 0, False

    leader_start = bool(_LEADER_AT_START_RE.match(line))
    stripped = line.strip()
    toc = None if is_table else _TOC_RE.match(stripped)
    if toc and len(LEADER_RE.findall(stripped)) == 1 and len(toc.group("head")) >= 6:
        # Entrada de índice: se conserva un separador mínimo entre título y página.
        return _collapse_spaces(f"{toc.group('head')} ... {toc.group('num')}"), False, removed, False

    out = _EMPHASIS_AROUND_FILLER_RE.sub(f" {BLANK_MARK} ", line)
    out = _LEADER_BEFORE_PUNCT_RE.sub(f" {BLANK_MARK}", out)
    out = LEADER_RE.sub(BLANK_MARK, out)
    if not is_table:
        out = _DASH_FILL_RE.sub(" ", out)
    out = _collapse_spaces(out)
    if not out.strip() or out.strip() in {"-", "*", "+", "•"}:
        return None, False, removed, False  # solo quedaba un marcador de lista vacío
    return out, leader_start, removed, False


# ---------------------------------------------------------------------------
# 3) Cabeceras y pies de página (el PDF aporta las líneas detectadas por página)
# ---------------------------------------------------------------------------
def norm_margin_line(text: str) -> str:
    """Forma comparable de una línea: sin marcas Markdown y con espacios normalizados."""
    return re.sub(r"\s+", " ", re.sub(r"[*_`#>\\|\u200b]", "", text)).strip()


def remove_margin_lines_detailed(page_md: str, literals: Iterable[str], edge: int = 4) -> tuple[str, list[str]]:
    """
    Elimina de los extremos de la página (primeras/últimas `edge` líneas no vacías) las
    líneas que coinciden literalmente con texto de cabecera/pie detectado en ESA página.
    Devuelve el texto y las líneas eliminadas.
    """
    wanted = {norm_margin_line(x) for x in literals if x and x.strip()}
    wanted.discard("")
    if not wanted:
        return page_md, []
    lines = page_md.split("\n")
    nonblank = [i for i, ln in enumerate(lines) if ln.strip()]
    candidates = set(nonblank[:edge]) | set(nonblank[-edge:])
    drop = sorted(i for i in candidates if norm_margin_line(lines[i]) in wanted)
    if not drop:
        return page_md, []
    removed = [lines[i].strip() for i in drop]
    return "\n".join(ln for i, ln in enumerate(lines) if i not in set(drop)), removed


def remove_margin_lines(page_md: str, literals: Iterable[str], edge: int = 4) -> tuple[str, int]:
    text, removed = remove_margin_lines_detailed(page_md, literals, edge)
    return text, len(removed)


# ---------------------------------------------------------------------------
# 4) Bloques y reconstrucción conservadora de párrafos
# ---------------------------------------------------------------------------
_LIST_RE = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)]|[A-Za-z][.)]|\(\d{1,3}\)|\([A-Za-z]\))\s+\S")
_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
_TERMINAL_RE = re.compile(r"[.!?…:;]['\"”’»)\]]*$")
_HYPHEN_END_RE = re.compile(r"[^\W_]-$")  # letra o dígito + guion al final de la línea
_CHECKBOX_RE = re.compile(r"^\s*\[[ xX?]\](?:\s|$)")
_LABEL_START_RE = re.compile(r"^\s*[^\W\d_][\w .()/'’-]{0,40}:(?:\s|$)")  # "Nombre: ..." (entrada de formulario)
_FENCE_RE = re.compile(r"^\s*(```|~~~)")


@dataclass
class _Block:
    kind: str  # text | list | table | heading | fence | quote | html | rule | marker
    lines: list[str] = field(default_factory=list)
    leader_start: bool = False
    page: int = 0
    full: list[bool] = field(default_factory=list)  # evidencia geométrica por línea (paralela a `lines`)

    def text(self) -> str:
        return "\n".join(self.lines)


def _classify(first: str) -> str:
    s = first.strip()
    if _FENCE_RE.match(s):
        return "fence"
    if MARKER_LINE_RE.match(s):
        return "marker"
    if s.startswith("|"):
        return "table"
    if _HEADING_RE.match(first):
        return "heading"
    if s.startswith(">"):
        return "quote"
    if s.startswith("<"):
        return "html"
    if re.fullmatch(r"(?:-{3,}|\*{3,}|_{3,})", s):
        return "rule"
    if _LIST_RE.match(first):
        return "list"
    return "text"


def _list_type(first_line: str) -> str:
    return "bullet" if re.match(r"^\s*[-*+•]\s", first_line) else "ordered"


def _is_structural(line: str) -> bool:
    s = line.strip()
    return bool(
        _LIST_RE.match(line)
        or _HEADING_RE.match(line)
        or _CHECKBOX_RE.match(line)
        or MARKER_LINE_RE.match(s)
        or s.startswith(("|", ">", "<", "```", "~~~"))
    )


def _ends_sentence(text: str) -> bool:
    return bool(_TERMINAL_RE.search(text.rstrip()))


def _join_pair(a: str, b: str) -> str:
    """Une dos líneas ya decididas como continuación (gestiona palabras cortadas con guion)."""
    a = a.rstrip()
    b = b.lstrip()
    if a.endswith(_SOFT_HYPHEN):
        return a[:-1] + b
    if _HYPHEN_END_RE.search(a) and b[:1].isalnum():
        fragment = re.search(r"([^\W\d_]+)-$", a)
        if fragment and fragment.group(1).islower() and b[:1].isalpha() and b[:1].islower():
            return a[:-1] + b  # "con-" + "trato" -> "contrato"
        return a + b  # "Decreto-" + "ley" -> "Decreto-ley"; "COVID-" + "19" -> "COVID-19"
    return a + " " + b


def _should_join(a: str, b: str, long_line: bool, full: bool = False) -> bool:
    """
    Evidencia de que `b` continúa la frase de `a` (no se une si hay dudas). `full`: la línea `a` llegaba
    al borde derecho de su columna en el PDF (ajuste automático de párrafo), salvo que `b` sea otra
    entrada de formulario ("Etiqueta: ...") o `a` termine en dos puntos.
    """
    if not a.strip() or not b.strip() or _is_structural(b):
        return False
    a_s = a.rstrip()
    if a_s.endswith(_SOFT_HYPHEN):
        return True
    if _HYPHEN_END_RE.search(a_s) and b.lstrip()[:1].isalnum():
        return True
    if full and not a_s.endswith(":") and not _LABEL_START_RE.match(b):
        return True
    if _ends_sentence(a_s):
        return False
    if b.lstrip()[:1].islower():
        return True
    # Línea llena (ajuste automático de párrafo) seguida de mayúscula/dígito. Una entrada de formulario
    # ("Nombre: ...") no se une aunque sea larga: lo normal es que la siguiente línea sea otro campo.
    return long_line and not _LABEL_START_RE.match(a_s)


def _merge_text_lines(lines: list[str], fulls: Optional[list[bool]] = None) -> list[str]:
    fulls = fulls if fulls and len(fulls) == len(lines) else [False] * len(lines)
    lengths = [len(x.strip()) for x in lines[:-1]]
    max_len = max(lengths) if lengths and len(lines) >= 3 else 0
    out: list[str] = []
    out_full: list[bool] = []
    for line, full in zip(lines, fulls):
        if out:
            a = out[-1]
            long_line = max_len >= 50 and len(a.strip()) >= 0.85 * max_len
            if _should_join(a, line, long_line, out_full[-1]):
                out[-1] = _join_pair(a, line)
                out_full[-1] = full
                continue
        out.append(line)
        out_full.append(full)
    return out


def _merge_list_lines(lines: list[str]) -> list[str]:
    """Las líneas de continuación de un elemento de lista (sin marcador) se unen a su elemento."""
    out: list[str] = []
    for line in lines:
        if out and not _is_structural(line):
            out[-1] = _join_pair(out[-1], line)
        else:
            out.append(line)
    return out


def _split_blocks(lines: list[_Line]) -> list[_Block]:
    blocks: list[_Block] = []
    cur: list[_Line] = []
    in_fence = False

    def flush() -> None:
        nonlocal cur
        if cur:
            blocks.append(
                _Block(_classify(cur[0].text), [x.text for x in cur], cur[0].leader_start, full=[x.full for x in cur])
            )
            cur = []

    for ln in lines:
        s = ln.text.strip()
        if in_fence:
            cur.append(ln)
            if _FENCE_RE.match(s):
                in_fence = False
                flush()
            continue
        if _FENCE_RE.match(s):
            flush()
            cur = [ln]
            in_fence = True
            continue
        if not s:
            flush()
            continue
        cur.append(ln)
    flush()
    return blocks


# ---------------------------------------------------------------------------
# 5) Campos vacíos (solo con omit_empty_fields), duplicados y bloques repetidos
# ---------------------------------------------------------------------------
_LABEL_BLANK_RE = re.compile(r"^\s*(?:[-*+•]\s+)?[^\n|]{1,80}?:\s*___\s*$")
_INLINE_BLANK_RE = re.compile(rf"[ \t]*{BLANK_MARK}(?=[\s,;:.)\]]|$)")
_SIGNATURE_RE = re.compile(
    r"firma|firmad|\bsign(?:ed|ature)|certificad|sello|\bcsv\b|huella|verificaci|timbr|\bfecha\b|\bdate\b",
    re.IGNORECASE,
)


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[*_`#>]", "", text)).strip().casefold()


def _omit_empty_in_line(line: str) -> tuple[Optional[str], int]:
    """Quita campos en blanco de una línea. Devuelve (texto o None si la línea desaparece, nº omitidos)."""
    s = line.strip()
    if s == BLANK_LINE or s == BLANK_MARK:
        return None, 1
    if s.startswith("|"):
        cells = re.split(r"(?<!\\)\|", s.strip("|"))
        n = sum(1 for c in cells if c.strip() == BLANK_MARK)
        if n:
            return "|" + "|".join("" if c.strip() == BLANK_MARK else c for c in cells) + "|", n
        return line, 0
    if _LABEL_BLANK_RE.match(line):
        return None, 1
    n = len(_INLINE_BLANK_RE.findall(line))
    if n:
        return _collapse_spaces(_INLINE_BLANK_RE.sub("", line)), n
    return line, 0


# ---------------------------------------------------------------------------
# 6) Compactación de formato (sin tocar el contenido)
# ---------------------------------------------------------------------------
_CELL_SPLIT_RE = re.compile(r"(?<!\\)\|")
_SEP_CELL_RE = re.compile(r"^:?-+:?$")
_EMPTY_BOLD_RE = re.compile(r"(?<!\*)\*{4}(?!\*)")  # "****": negrita vacía
_BOLD_ADJACENT_RE = re.compile(r"\*\*([^*\n]+?)\*\*[ \t]+\*\*([^*\n]+?)\*\*")
_HEADING_BOLD_RE = re.compile(r"^(#{1,6})[ \t]+\*\*([^*\n]+?)\*\*[ \t]*$")


def _compact_table_line(line: str, drop_empty: bool) -> Optional[str]:
    cells = [c.strip() for c in _CELL_SPLIT_RE.split(line.strip().strip("|"))]
    cells = [re.sub(r"^(?:<br\s*/?>\s*)+|(?:\s*<br\s*/?>)+$", "", c).strip() for c in cells]
    if cells and all(_SEP_CELL_RE.match(c) for c in cells):
        fixed = []
        for c in cells:
            left, right = c.startswith(":"), c.endswith(":")
            fixed.append((":" if left else "") + "---" + (":" if right else ""))
        return "|" + "|".join(fixed) + "|"
    if drop_empty and cells and not any(cells):
        return None  # fila totalmente vacía (solo si se pidió omitir campos vacíos)
    return "| " + " | ".join(cells) + " |"


def _compact_block(block: _Block, drop_empty: bool) -> list[str]:
    lines = block.lines
    if block.kind == "fence":
        return lines
    if block.kind == "table":
        out = [_compact_table_line(x, drop_empty) if x.strip().startswith("|") else x for x in lines]
        return [x for x in out if x is not None]
    result = []
    for line in lines:
        line = _EMPTY_BOLD_RE.sub("", line)
        prev = None
        while prev != line:  # **a** **b** -> **a b**
            prev = line
            line = _BOLD_ADJACENT_RE.sub(r"**\1 \2**", line)
        line = _HEADING_BOLD_RE.sub(r"\1 \2", line)
        line = re.sub(r"^([ \t]*)•[ \t]+", r"\1- ", line)
        result.append(_collapse_spaces(line) if line.strip() else line)
    return result


# ---------------------------------------------------------------------------
# API pública
# ---------------------------------------------------------------------------
@dataclass
class CleanResult:
    text: str
    margin_lines_removed: int = 0
    fillers_removed: int = 0  # rellenos normalizados (a `___`) o descartados (rayas)
    duplicates_removed: int = 0
    repeated_blocks_removed: int = 0
    redundant_headings_removed: int = 0
    empty_fields_omitted: int = 0
    removed: list[dict[str, Any]] = field(default_factory=list)  # informe de lo eliminado


def _resolve_options(mode: str, options: Optional[ConvertOptions]) -> ConvertOptions:
    if mode not in MODES:
        raise ValueError(f"Modo desconocido: {mode!r}. Opciones: {MODES}")
    if options is None:
        return ConvertOptions.for_mode(mode)
    if options.mode != mode:
        raise ValueError("`mode` y `options.mode` no coinciden")
    return options


def _process_page(
    page_md: str,
    page_no: int,
    opts: ConvertOptions,
    margin_literals: Optional[Iterable[str]],
    stats: CleanResult,
) -> list[_Block]:
    text = page_md.replace("\r\n", "\n").replace("\r", "\n")
    text = _SOFT_HYPHEN_EOL_RE.sub("", text)  # guion blando a final de línea: palabra cortada con certeza
    # Un guion blando fuera del final de línea no aporta nada.
    text = text.replace(_SOFT_HYPHEN, "")

    if margin_literals and (opts.remove_headers_footers or opts.remove_page_numbers):
        text, removed = remove_margin_lines_detailed(text, margin_literals)
        stats.margin_lines_removed += len(removed)
        stats.removed += [{"kind": "margin", "text": t.replace(_FULL_MARK, ""), "page": page_no + 1} for t in removed]

    cleaned: list[_Line] = []
    in_fence = False
    for raw in text.split("\n"):
        full = raw.rstrip().endswith(_FULL_MARK)
        raw = normalize_chars(raw)  # también retira la marca de "fila llena"
        if in_fence or _FENCE_RE.match(raw.strip()):
            if _FENCE_RE.match(raw.strip()):
                in_fence = not in_fence
            cleaned.append(_Line(raw.rstrip()))  # el código se conserva tal cual
            continue
        out, leader_start, removed, blank_only = _clean_line(raw)
        stats.fillers_removed += removed
        if out is not None:
            cleaned.append(_Line(out, leader_start, blank_only, full))

    # Una línea que es solo relleno entre dos líneas de texto es un hueco dentro de la frase.
    lines: list[_Line] = []
    for i, ln in enumerate(cleaned):
        if ln.blank_only:
            prev = lines[-1].text if lines else ""
            nxt = cleaned[i + 1].text if i + 1 < len(cleaned) else ""
            if prev.strip() and nxt.strip() and not _is_structural(prev) and not prev.rstrip().endswith("```"):
                lines[-1].text = prev.rstrip() + " " + BLANK_MARK
                continue
            ln.text = BLANK_LINE
        lines.append(ln)

    if opts.omit_empty_fields:
        kept: list[_Line] = []
        for ln in lines:
            new, n = _omit_empty_in_line(ln.text) if not _FENCE_RE.match(ln.text.strip()) else (ln.text, 0)
            stats.empty_fields_omitted += n
            if n and new is None:
                stats.removed.append({"kind": "empty_field", "text": ln.text.strip(), "page": page_no + 1})
            if new is not None:
                kept.append(_Line(new, ln.leader_start, ln.blank_only, ln.full))
        lines = kept

    blocks = _split_blocks(lines)
    for block in blocks:
        block.page = page_no
        if opts.join_paragraphs:
            if block.kind == "text":
                block.lines = _merge_text_lines(block.lines, block.full)
            elif block.kind == "list":
                block.lines = _merge_list_lines(block.lines)
    return blocks


def _is_protected(text: str) -> bool:
    """Bloques que nunca se tratan como ruido repetido (firmas, fechas de firma, sellos...)."""
    return bool(_SIGNATURE_RE.search(text))


def _drop_repetitions(blocks: list[_Block], opts: ConvertOptions, stats: CleanResult) -> list[_Block]:
    out: list[_Block] = []
    seen_long: set[str] = set()
    for block in blocks:
        prev = out[-1] if out else None
        key = _norm_key(block.text())
        reason: Optional[str] = None
        if opts.remove_duplicates and prev is not None and block.kind in ("text", "quote") and prev.kind == block.kind:
            if len(key) >= 20 and key == _norm_key(prev.text()) and not _is_protected(key):
                reason = "duplicate_block"
                stats.duplicates_removed += 1
        if reason is None and opts.omit_redundant_headings and block.kind == "heading":
            if prev is not None and prev.kind == "heading" and key == _norm_key(prev.text()):
                reason = "redundant_heading"
                stats.redundant_headings_removed += 1
        if reason is None and opts.remove_repeated_blocks and block.kind == "text" and len(key) >= 60:
            if key in seen_long and not _is_protected(key):
                reason = "repeated_block"
                stats.repeated_blocks_removed += 1
        if reason:
            stats.removed.append({"kind": reason, "text": block.text()[:160], "page": block.page + 1})
            continue
        if block.kind == "text" and len(key) >= 60:
            seen_long.add(key)
        out.append(block)
    return out


def clean_pages(
    pages: list[str],
    mode: str = DEFAULT_MODE,
    margin_lines: Optional[list[Iterable[str]]] = None,
    options: Optional[ConvertOptions] = None,
) -> CleanResult:
    """Limpia el Markdown de cada página y lo une en un único documento."""
    opts = _resolve_options(mode, options)
    stats = CleanResult("")
    all_blocks: list[_Block] = []
    page_start_idx: set[int] = set()

    for i, page_md in enumerate(pages):
        literals = margin_lines[i] if margin_lines and i < len(margin_lines) else None
        blocks = _process_page(page_md, i, opts, literals, stats)
        if blocks:
            page_start_idx.add(len(all_blocks))
            all_blocks.extend(blocks)

    merged: list[_Block] = []
    for idx, block in enumerate(all_blocks):
        prev = merged[-1] if merged else None
        if opts.join_paragraphs and prev is not None and prev.kind == "text" and block.kind == "text":
            prev_text = prev.lines[-1]
            first = block.lines[0]
            continuation = False
            if block.leader_start and not _ends_sentence(prev_text):
                continuation = True  # el relleno indica que la línea seguía a la anterior
            elif idx in page_start_idx and not _ends_sentence(prev_text) and first.lstrip()[:1].islower():
                continuation = True  # párrafo partido por un salto de página
            elif idx in page_start_idx and _HYPHEN_END_RE.search(prev_text.rstrip()) and first.lstrip()[:1].isalnum():
                continuation = True  # palabra partida con guion entre páginas
            if continuation:
                prev.lines[-1] = _join_pair(prev_text, first)
                prev.lines.extend(block.lines[1:])
                continue
        merged.append(block)

    merged = _drop_repetitions(merged, opts, stats)

    if opts.compact_format:
        for block in merged:
            block.lines = _compact_block(block, opts.omit_empty_fields)
        merged = [b for b in merged if any(x.strip() for x in b.lines)]

    parts: list[str] = []
    prev_block: Optional[_Block] = None
    for block in merged:
        text = block.text()
        if (
            opts.compact_format
            and prev_block is not None
            and prev_block.kind == "list"
            and block.kind == "list"
            and _list_type(prev_block.lines[0]) == _list_type(block.lines[0])
        ):
            parts[-1] = parts[-1] + "\n" + text  # sin línea en blanco entre elementos de la misma lista
        else:
            parts.append(text)
        prev_block = block

    body = "\n\n".join(parts).replace(_SOFT_HYPHEN, "").strip("\n")
    stats.text = body + "\n" if body else ""
    return stats


def clean_markdown(md: str, mode: str = DEFAULT_MODE, options: Optional[ConvertOptions] = None) -> str:
    """Limpia un documento Markdown completo (sin información de páginas)."""
    return clean_pages([md], mode, options=options).text
