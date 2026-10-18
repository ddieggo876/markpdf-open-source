# MarkPDF - convierte PDF en Markdown para usar como contexto de LLM.
# Copyright (C) 2026 MarkPDF contributors
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Este programa es software libre: puedes redistribuirlo y/o modificarlo bajo los terminos de la GNU Affero General
# Public License publicada por la Free Software Foundation, version 3 o (a tu eleccion) cualquier version posterior.
# Se distribuye SIN NINGUNA GARANTIA; consulta el archivo LICENSE para mas detalles.

"""PDFs sintéticos pequeños para las pruebas (se generan en memoria con PyMuPDF)."""

from __future__ import annotations

import pymupdf

# Nota: se evita la letra "í" porque la fuente base-14 usada por insert_text() no la
# mapea a Unicode (artefacto del generador de fixtures, no de MarkPDF).

A4 = (595, 842)


def _new_doc() -> "pymupdf.Document":
    return pymupdf.open()


def contract_pdf(n_pages: int = 5) -> bytes:
    """Contrato con cabecera, pie y numeración repetidos y una línea de puntos de relleno."""
    doc = _new_doc()
    for i in range(1, n_pages + 1):
        p = doc.new_page(width=A4[0], height=A4[1])
        p.insert_text((72, 40), "CONTRATO DE TRABAJO - ACME S.L.", fontsize=9)
        p.insert_text((72, 120), f"CLAUSULA {i}", fontsize=16)
        p.insert_text((72, 160), f"El trabajador prestara servicios desde el 20/04/2026 en la sede numero {i} y", fontsize=11)
        p.insert_text((72, 176), "percibira un salario de 1.500,00 EUR brutos mensuales ............................", fontsize=11)
        p.insert_text((72, 192), "segun lo pactado entre las partes.", fontsize=11)
        p.insert_text((260, 810), f"Pagina {i} de {n_pages}", fontsize=9)
    return doc.tobytes()


def two_column_pdf() -> bytes:
    """Dos columnas de texto continuo: debe leerse toda la columna A y después toda la B."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((50, 60), "Titulo a todo el ancho del documento", fontsize=16)
    col_a = " ".join(f"alfa{i}" for i in range(1, 121))
    col_b = " ".join(f"beta{i}" for i in range(1, 121))
    p.insert_textbox(pymupdf.Rect(50, 90, 280, 500), col_a, fontsize=10)
    p.insert_textbox(pymupdf.Rect(320, 90, 550, 500), col_b, fontsize=10)
    return doc.tobytes()


def clean_pdf() -> bytes:
    """Documento corto ya limpio (sin ruido que eliminar)."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 100), "Informe de situacion", fontsize=18)
    p.insert_text((72, 140), "Este documento resume el estado del proyecto en una sola pagina.", fontsize=11)
    p.insert_text((72, 160), "No contiene cabeceras repetidas ni lineas de relleno.", fontsize=11)
    return doc.tobytes()


def empty_pdf() -> bytes:
    doc = _new_doc()
    doc.new_page(width=A4[0], height=A4[1])
    return doc.tobytes()


def scanned_pdf() -> bytes:
    """Página que solo contiene una imagen (sin capa de texto), como un escaneo."""
    src = _new_doc()
    sp = src.new_page(width=A4[0], height=A4[1])
    sp.insert_text((72, 100), "Pagina escaneada con texto dentro de la imagen", fontsize=16)
    pix = sp.get_pixmap(dpi=72)
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_image(p.rect, pixmap=pix)
    return doc.tobytes()


def image_pdf() -> bytes:
    """Texto + una imagen grande con pie de figura."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((50, 60), "Informe con imagen", fontsize=16)
    p.insert_text((50, 90), "Texto antes de la imagen para dar contexto al documento de prueba.", fontsize=10)
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    pix.set_rect(pix.irect, (30, 120, 220))
    pix.set_rect(pymupdf.IRect(20, 20, 120, 100), (240, 200, 20))
    p.insert_image(pymupdf.Rect(100, 120, 400, 320), pixmap=pix)
    p.insert_text((100, 340), "Figura 1: Ventas por trimestre", fontsize=9)
    p.insert_text((50, 380), "Texto despues de la imagen.", fontsize=10)
    return doc.tobytes()


def table_pdf() -> bytes:
    """Tabla con bordes (celdas con líneas) que pymupdf4llm convierte en tabla Markdown."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 80), "Tabla salarial", fontsize=14)
    rows = [("Concepto", "Importe"), ("Salario base", "1200"), ("Plus transporte", "150"), ("Antiguedad", "75")]
    x0, y0, cw, rh = 72, 100, 160, 24
    for r, (a, b) in enumerate(rows):
        for c, text in enumerate((a, b)):
            rect = pymupdf.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            p.insert_text((rect.x0 + 6, rect.y0 + 16), text, fontsize=11)
    return doc.tobytes()


# ---------------------------------------------------------------------------
# Fixtures añadidos en la auditoría (formularios, tablas, numeración, duplicados, imágenes...)
# ---------------------------------------------------------------------------
def _draw_table(p: "pymupdf.Page", x0: float, y0: float, rows, widths, rh: float = 24, fontsize: float = 10) -> None:
    for r, row in enumerate(rows):
        x = x0
        for c, text in enumerate(row):
            rect = pymupdf.Rect(x, y0 + r * rh, x + widths[c], y0 + (r + 1) * rh)
            p.draw_rect(rect, color=(0, 0, 0), width=0.8)
            if text:
                p.insert_text((rect.x0 + 4, rect.y0 + 15), text, fontsize=fontsize)
            x += widths[c]


def table_empty_cells_pdf() -> bytes:
    """Tabla con celdas vacías en distintas posiciones y texto pegado debajo (no debe ser una celda)."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 60), "Tabla con celdas vacias", fontsize=14)
    _draw_table(
        p, 72, 80,
        [("Concepto", "Importe", "Notas"), ("Salario base", "1200", ""), ("", "150", "Plus transporte"), ("Antiguedad", "", "")],
        [150, 100, 150],
    )
    p.insert_text((72, 210), "Texto justo despues de la tabla que no debe ser celda.", fontsize=10)
    return doc.tobytes()


def two_tables_pdf() -> bytes:
    """Dos tablas independientes separadas por un párrafo: no deben fusionarse."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 60), "Primera tabla", fontsize=12)
    _draw_table(p, 72, 70, [("Producto", "Precio"), ("Cafe", "2"), ("Te", "3")], [150, 100])
    p.insert_text((72, 190), "Entre las dos tablas hay un parrafo de texto normal que no pertenece a ninguna.", fontsize=10)
    p.insert_text((72, 230), "Segunda tabla", fontsize=12)
    _draw_table(p, 72, 240, [("Ciudad", "Habitantes"), ("Madrid", "3300000"), ("Soria", "40000")], [150, 100])
    return doc.tobytes()


def contract_form_pdf() -> bytes:
    """Contrato con campos cumplimentados, campos en blanco (puntos), casillas y texto legal fijo."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    lines = [
        (60, "DATOS DEL ARRENDATARIO", 13),
        (90, "Nombre: Maria Lopez Garcia", 10),
        (106, "DNI: 12345678Z", 10),
        (122, "Telefono (opcional): ................................", 10),
        (138, "Email:", 10),
        (154, "Direccion: Calle Mayor 5, 3 B", 10),
        (190, "[X] Acepto las condiciones   [ ] No acepto   [ ] Autorizo comunicaciones", 10),
        (226, "El arrendatario NO podra subarrendar el inmueble sin consentimiento escrito.", 10),
        (250, "Renta mensual: 850,00 EUR (art. 17.2 de la Ley 29/1994).", 10),
    ]
    for y, text, size in lines:
        p.insert_text((72, y), text, fontsize=size)
    return doc.tobytes()


def widget_form_pdf() -> bytes:
    """Formulario AcroForm real: texto cumplimentado, texto vacío, casilla marcada y casilla sin marcar."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    for y, label in [(80, "Nombre completo:"), (110, "Telefono (opcional):"), (140, "Acepto condiciones"), (170, "No acepto")]:
        p.insert_text((72, y), label, fontsize=11)

    def add(kind, name, rect, value=None):
        w = pymupdf.Widget()
        w.field_type, w.field_name, w.rect = kind, name, pymupdf.Rect(*rect)
        if value is not None:
            w.field_value = value
        p.add_widget(w)

    add(pymupdf.PDF_WIDGET_TYPE_TEXT, "nombre", (200, 65, 400, 85), "Juan Perez")
    add(pymupdf.PDF_WIDGET_TYPE_TEXT, "telefono", (200, 95, 400, 115))
    add(pymupdf.PDF_WIDGET_TYPE_CHECKBOX, "acepto", (200, 128, 214, 142), True)
    add(pymupdf.PDF_WIDGET_TYPE_CHECKBOX, "noacepto", (200, 158, 214, 172), False)
    return doc.tobytes()


def numbered_pdf(n_pages: int = 5, *, constant_number: bool = False) -> bytes:
    """Páginas con un número suelto al pie (1..n) o, si `constant_number`, el mismo valor (p. ej. un año)."""
    doc = _new_doc()
    for i in range(1, n_pages + 1):
        p = doc.new_page(width=A4[0], height=A4[1])
        p.insert_text((72, 120), f"Contenido de la pagina numero {i} con el articulo {i + 10}.", fontsize=11)
        p.insert_text((290, 810), "2026" if constant_number else str(i), fontsize=9)
    return doc.tobytes()


def varying_header_pdf(n_pages: int = 5) -> bytes:
    """Cabecera cuyo número NO sigue la numeración de página (capítulo 3, 3, 7, 7, 9): no es un encabezado repetido."""
    chapters = [3, 3, 7, 7, 9, 9, 9, 9]
    doc = _new_doc()
    for i in range(n_pages):
        p = doc.new_page(width=A4[0], height=A4[1])
        p.insert_text((72, 40), f"Capitulo {chapters[i]} del reglamento", fontsize=9)
        p.insert_text((72, 120), f"Texto del cuerpo de la pagina {i + 1}.", fontsize=11)
    return doc.tobytes()


def repeated_in_body_pdf(n_pages: int = 4) -> bytes:
    """El mismo texto aparece en el margen superior Y como contenido del cuerpo: no se debe eliminar."""
    doc = _new_doc()
    for i in range(n_pages):
        p = doc.new_page(width=A4[0], height=A4[1])
        p.insert_text((72, 40), "ANEXO DE CONDICIONES", fontsize=9)
        p.insert_text((72, 300), "ANEXO DE CONDICIONES", fontsize=14)
        p.insert_text((72, 330), f"Condicion particular numero {i + 1}.", fontsize=11)
    return doc.tobytes()


def duplicates_pdf() -> bytes:
    """Párrafo repetido de forma consecutiva, texto superpuesto en dos capas y un bloque de firma."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    sentence = "El arrendatario no podra subarrendar el inmueble sin consentimiento escrito."
    p.insert_text((72, 100), sentence, fontsize=11)
    p.insert_text((72, 140), sentence, fontsize=11)
    p.insert_text((72, 200), "IMPORTE TOTAL: 1.500,00 EUR", fontsize=12)
    p.insert_text((72, 200), "IMPORTE TOTAL: 1.500,00 EUR", fontsize=12)  # misma posición: capa superpuesta
    p.insert_text((72, 260), "Firmado digitalmente por: GARCIA LOPEZ MARIA - 12345678Z", fontsize=9)
    p.insert_text((72, 272), "Fecha: 2026.04.20 10:33:12 +02'00'", fontsize=9)
    return doc.tobytes()


def hyphenation_pdf() -> bytes:
    """Palabras partidas con guion a final de línea y compuestos con guion legítimos."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    text = (
        "El presente contrato regula la contra-\ntacion de servicios entre las partes y se\n"
        "rige por la normativa vigente en materia de\nCOVID-\n19 y por el Real Decreto-\nley 1/2020 de 7 de enero."
    )
    y = 80
    for ln in text.split("\n"):
        p.insert_text((72, y), ln, fontsize=11)
        y += 14
    return doc.tobytes()


def list_pdf() -> bytes:
    """Lista numerada con subelementos y lista con viñetas anidada (sangría por posición)."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    rows = [(72, "1. Primero"), (90, "a) subitem uno"), (90, "b) subitem dos"), (72, "2. Segundo"), (72, "- punto A"), (90, "- punto B")]
    for i, (x, text) in enumerate(rows):
        p.insert_text((x, 100 + i * 16), text, fontsize=10)
    return doc.tobytes()


def logo_pdf(n_pages: int = 5, *, with_figure: bool = False) -> bytes:
    """Una imagen repetida en todas las páginas (cabecera) y, opcionalmente, una figura única en la página 2."""
    logo = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 120, 60), False)
    logo.set_rect(logo.irect, (200, 30, 30))
    figure = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 300, 200), False)
    figure.set_rect(figure.irect, (30, 120, 220))
    tiny = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 10, 10), False)
    tiny.set_rect(tiny.irect, (0, 0, 0))
    doc = _new_doc()
    for i in range(1, n_pages + 1):
        p = doc.new_page(width=A4[0], height=A4[1])
        p.insert_image(pymupdf.Rect(72, 36, 322, 186), pixmap=logo)  # 250x150 pt (> 6 % de la página)
        p.insert_image(pymupdf.Rect(500, 700, 510, 710), pixmap=tiny)  # decorativa por tamaño
        p.insert_text((72, 240), f"Texto de la pagina {i}.", fontsize=11)
        if with_figure and i == 2:
            p.insert_image(pymupdf.Rect(100, 300, 400, 500), pixmap=figure)
            p.insert_text((100, 520), "Figura 2: Evolucion mensual", fontsize=9)
    return doc.tobytes()


def symbols_pdf() -> bytes:
    """Casillas dibujadas con la fuente ZapfDingbats (glifo de casilla que el motor no interpreta bien)."""
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 100), "Texto previo a las casillas", fontsize=11)
    p.insert_text((72, 130), "Opcion marcada", fontsize=11)
    p.insert_text((200, 130), "4", fontsize=11, fontname="zadb")  # ✔ en ZapfDingbats
    p.insert_text((72, 160), "Otra opcion", fontsize=11)
    p.insert_text((200, 160), "o", fontsize=11, fontname="zadb")  # ❏ en ZapfDingbats
    return doc.tobytes()


def wrapped_pages_pdf() -> bytes:
    """Párrafos justificados por ajuste automático; el segundo continúa en la página siguiente."""
    doc = _new_doc()
    para1 = (
        "Las partes acuerdan que el presente contrato se regira por lo dispuesto en la legislacion vigente, "
        "sin perjuicio de los pactos particulares que figuran en las clausulas siguientes y que las partes "
        "declaran conocer y aceptar en todos sus terminos."
    )
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_textbox(pymupdf.Rect(72, 80, 523, 200), para1, fontsize=11)
    p.insert_text((72, 760), "Las obligaciones de las partes se mantendran vigentes durante toda la relacion y", fontsize=11)
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 80), "hasta su completa extincion segun lo previsto.", fontsize=11)
    return doc.tobytes()


def corrupt_pdf() -> bytes:
    return b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 99 0 R >>\nendobj\nesto no es un pdf valido\n"


def encrypted_pdf() -> bytes:
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 100), "Documento confidencial", fontsize=12)
    return doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, user_pw="secreto", owner_pw="propietario")

def mixed_scanned_pdf() -> bytes:
    """Página 1 con texto seleccionable y página 2 que es solo una imagen (escaneo)."""
    src = _new_doc()
    sp = src.new_page(width=A4[0], height=A4[1])
    sp.insert_text((72, 100), "Pagina escaneada con texto dentro de la imagen", fontsize=16)
    pix = sp.get_pixmap(dpi=72)
    doc = _new_doc()
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_text((72, 100), "Pagina con texto seleccionable y normal.", fontsize=12)
    p = doc.new_page(width=A4[0], height=A4[1])
    p.insert_image(p.rect, pixmap=pix)
    return doc.tobytes()