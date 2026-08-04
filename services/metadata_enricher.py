from langchain_core.documents import Document

MATERIAS = {
    "administrativo": "Derecho Administrativo",
    "civil": "Derecho Civil",
    "comercial": "Derecho Comercial",
    "constitucional": "Derecho Constitucional",
    "consumidor": "Derecho del Consumidor",
    "familia": "Derecho de Familia",
    "laboral": "Derecho Laboral",
    "penal": "Derecho Penal",
    "societario": "Derecho Societario",
    "tributario": "Derecho Tributario",
}


def enriquecer_con_materia(documentos: list[Document]) -> list[Document]:
    """
    Agrega la materia legal a cada documento, según la subcarpeta de donde proviene.
    Solo etiqueta los datos — no filtra ni decide nada sobre las búsquedas.
    """
    for doc in documentos:
        ruta = doc.metadata.get("source", "")
        for carpeta, nombre_materia in MATERIAS.items():
            if f"/{carpeta}/" in ruta:
                doc.metadata["materia"] = nombre_materia
                break
        else:
            doc.metadata["materia"] = "General"

    return documentos