from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def construir_vectorstore(
        chunks: list[Document],
        embeddings: Embeddings,
        persist_directory: str 
) -> Chroma:
    """Construye un vector store Chroma..."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )