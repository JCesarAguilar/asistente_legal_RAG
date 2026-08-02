from pdf2image import convert_from_path
import pytesseract


def extraer_pagina_como_imagen(ruta_pdf, numero_pagina: int, dpi: int = 300):
    """Convierte una pagina especifica del PDF en imagen."""
    imagenes = convert_from_path(
        ruta_pdf, dpi=dpi, 
        first_page=numero_pagina, 
        last_page=numero_pagina
    )

    return imagenes[0]  # Retorna la primera (y unica) imagen de la lista


def ocr_tesseract(imagen) -> str:
    """Ejecuta OCR local con Tesseract"""
    return pytesseract.image_to_string(imagen, lang='spa', config='--psm 6') 


def reprocesar_pagina(ruta_pdf: str, numero_pagina: int) -> str:
    """Orquesta: extrae imagen + aplica OCR, devuelve texto"""
    imagen = extraer_pagina_como_imagen(ruta_pdf, numero_pagina)
    texto_extraido = ocr_tesseract(imagen)
    return texto_extraido