from langchain_core.documents import Document

  
def format_docs(docs: list[Document]) -> str:
    """
    Formatea una lista de documentos recuperados en un string legible,
    incluyendo la fuente y página de cada fragmento para trazabilidad.
    """
    fragmentos = []

    for i, doc in enumerate(docs, 1):
        header = f"[Fragmento {i}]"

        if doc.metadata:
            materia = doc.metadata.get("materia")
            if materia:
                header += f" (Materia: {materia})"   # 👈 nueva línea

            fuente_raw = doc.metadata.get("source", "")
            if fuente_raw:
                fuente = fuente_raw.split("\\")[-1] if "\\" in fuente_raw else fuente_raw.split("/")[-1]
                header += f" (Fuente: {fuente})"

            pagina = doc.metadata.get("page")
            if pagina is not None:
                header += f" (Página: {pagina + 1})"

        contenido = doc.page_content.strip()
        fragmentos.append(f"{header}\n{contenido}")

    return "\n\n".join(fragmentos)