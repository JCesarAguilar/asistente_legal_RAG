from dotenv import load_dotenv
load_dotenv()

from config import DIRECTORIO_DOCUMENTOS, PINECONE_INDEX_NAME
from services import (
    cargar_documentos,
    calidad_ocr_sospechosa,
    reprocesar_pagina,
    dividir_en_chunks,
    construir_vectorstore,
    enriquecer_con_materia,
    obtener_embeddings
)


def reprocesar_documentos_con_ocr(documentos: list) -> list:
    for doc in documentos:
        if calidad_ocr_sospechosa(doc.page_content):
            ruta_pdf = doc.metadata.get("source")
            pagina = doc.metadata.get("page", 0)
            print(f"Reprocesando OCR: {ruta_pdf} - página {pagina + 1}")
            try:
                texto_corregido = reprocesar_pagina(ruta_pdf, pagina)
                if texto_corregido.strip():
                    doc.page_content = texto_corregido
                    doc.metadata["ocr_reprocesado"] = True
            except Exception as e:
                print(f"Error al reprocesar página {pagina + 1}: {e}")
                doc.metadata["ocr_reprocesado"] = False
    return documentos


def main():
    print("1. Cargando documentos...")
    documentos = cargar_documentos(DIRECTORIO_DOCUMENTOS)
    print(f"{len(documentos)} páginas cargadas.")

    documentos = enriquecer_con_materia(documentos)

    print("2. Verificando calidad OCR...")
    documentos = reprocesar_documentos_con_ocr(documentos)

    print("3. Dividiendo documentos en chunks...")
    chunks = dividir_en_chunks(documentos)
    print(f"{len(chunks)} chunks generados.")

    print("4. Construyendo vectorstore en Pinecone...")
    embeddings = obtener_embeddings()
    construir_vectorstore(chunks, embeddings, PINECONE_INDEX_NAME)
    print(f"Vectorstore construido en Pinecone, índice: {PINECONE_INDEX_NAME}")


if __name__ == "__main__":
    main()