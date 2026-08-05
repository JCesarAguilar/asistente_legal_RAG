from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore

from config import *
from prompts.prompts import *
from services.document_formatter import format_docs
from services.retriever_builder import crear_retriever_base, crear_multi_query_retriever, crear_similarity_retriever, crear_ensemble_retriever
from services.llm_provider import obtener_embeddings, obtener_llm_queries, obtener_llm_generativo

@st.cache_resource
def initialize_rag_system():
    """
    Inicializa todo el sistema RAG (vectorstore, modelos, retriever, cadena)
    Se cachea con @st.cache_resource para no reconstruirlo en cada intercaccion del usuario.
    """
    embeddings = obtener_embeddings()

    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )

    # Modelos
    llm_queries = obtener_llm_queries()
    llm_generative = obtener_llm_generativo()

    # Retriever MMR (Maximal Margin Relevance)
    base_retriever = crear_retriever_base(
        vectorstore=vectorstore,
        search_type=SEARCH_TYPE,
        k=SEARCH_K,
        fetch_k=MMR_FETCH_K,
        lambda_mult=MMR_DIVERSITY_LAMBDA
    )

    # Prompt personalizado para el MultiQueryRetriever
    multi_query_prompt = PromptTemplate.from_template(MULTI_QUERY_PROMPT)

    # MultiQueryRetriever con prompt personalizado
    mmr_multi_retriever = crear_multi_query_retriever(
        base_retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    # Retriever de similitud simple
    similarity_retriever = crear_similarity_retriever(
        vectorstore=vectorstore,
        k=SEARCH_K
    )

    # Combinar ambos retrievers en un EnsembleRetriever
    ensemble_retriever = crear_ensemble_retriever(
        retrievers=[mmr_multi_retriever, similarity_retriever],
        weights=ENSEMBLE_WEIGHTS  # Ajusta los pesos según tus necesidades
    )

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = (
        {
            "context": ensemble_retriever | format_docs,
            "question": RunnablePassthrough()
        } 
        | prompt
        | llm_generative 
        | StrOutputParser()
    )

    return rag_chain, ensemble_retriever


def query_rag(question):
    try:
        rag_chain, retriever = initialize_rag_system()

        # Obtener respuesta
        response = rag_chain.invoke(question)

        # Obtener documentos para mostrarlos
        docs = retriever.invoke(question)

        # Formatera los documentos para mostrarlos
        docs_info = []
        for i, doc in enumerate(docs[:SEARCH_K], 1):
            docs_info.append({
                "fragmento": i,
                "contenido": doc.page_content[:1000] + "..." if len(doc.page_content) > 1000 else doc.page_content,
                "fuente": doc.metadata.get("source", "No especificada"),
                "pagina": doc.metadata.get("page", "No especificada")
            })

        return response, docs_info    

    except Exception as e:
        error_msg = f"Error al procesar la pregunta: {str(e)}"
        return error_msg, []


def get_retriever_info():
    """Obtiene informacion sobre la configuracion del retriever"""
    return {
        "tipo": f"{SEARCH_TYPE.upper()} + MultiQuery + Similarity (Ensemble)",
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "pesos_ensemble": ENSEMBLE_WEIGHTS
    }           