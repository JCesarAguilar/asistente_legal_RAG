from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document


def cargar_documentos(directorio: str) -> list[Document]:
    """Carga todos los PDFs de un directorio, uno por página"""
    loader = PyPDFDirectoryLoader(directorio, glob="**/*.pdf")
    pages = loader.load()
    return pages