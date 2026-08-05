import time
from typing import cast
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def construir_vectorstore(
    chunks: list[Document],
    embeddings: Embeddings,
    index_name: str,
    lote_size: int = 20,
    pausa_segundos: int = 3
) -> PineconeVectorStore:
    """Construye un vector store en Pinecone, procesando en lotes."""
    vectorstore = None
    total = len(chunks)

    for i in range(0, total, lote_size):
        lote = chunks[i:i + lote_size]
        numero_lote = i // lote_size + 1
        total_lotes = (total + lote_size - 1) // lote_size

        print(f"  Procesando lote {numero_lote}/{total_lotes} ({len(lote)} chunks)...")

        if vectorstore is None:
            vectorstore = PineconeVectorStore.from_documents(
                documents=lote,
                embedding=embeddings,
                index_name=index_name
            )
        else:
            vectorstore.add_documents(lote)

        time.sleep(pausa_segundos)

    return cast(PineconeVectorStore, vectorstore)