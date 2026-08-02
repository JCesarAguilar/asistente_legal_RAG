import re

def calidad_ocr_sospechosa(texto: str, umbral: float = 0.03) -> bool:
    """Detecta si un texto parece tener OCR de mala calidad"""
    if not texto.strip():
        return True  
    caracteres_raros = len(re.findall(r'[^a-zA-Z0-9\s.,;:!?()\-\'\"áéíóúÁÉÍÓÚñÑ]', texto))
    proporcion = caracteres_raros / max(len(texto), 1)  # Evitar división por cero
    return proporcion > umbral