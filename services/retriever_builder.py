from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from langchain_classic.retrievers import MultiQueryRetriever


def crear_retriever_base(
        vectorstore: VectorStore,
        search_type: str = "mmr",
        k: int = 2,
        fetch_k: int = 10,
        lambda_mult: float = 0.5,
) -> BaseRetriever:
    """Crea un retriever simple a partir de un vectorstore ya existente."""
    return vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )

def crear_multi_query_retriever(
        base_retriever: BaseRetriever,
        llm: BaseChatModel,
        prompt=None
) -> MultiQueryRetriever:
    """Envuelve un retriever base con reformulacion automatica de consultas (MultiQueryRetriever)."""
    if prompt:
        return MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=llm,
            prompt=prompt
        )
    return MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )