import os
import streamlit as st
from pydantic import SecretStr
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from config import *
from prompts.prompts import *
from services.document_formatter import format_docs
from services.retriever_builder import crear_retriever_base, crear_multi_query_retriever

@st.cache_resource
def initialize_rag_system():
    """
    Inicializa todo el sistema RAG (vectorstore, modelos, retriever, cadena)
    Se cachea con @st.cache_resource para no reconstruirlo en cada intercaccion del usuario.
    """

    # Vector Store
    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )

    vectorstore = Chroma(
    embedding_function=embeddings, 
    persist_directory=CHROMA_DB_PATH
    )

    # Modelos
    llm_queries = ChatGoogleGenerativeAI(
        model=QUERY_MODEL,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )
    llm_generative = ChatGoogleGenerativeAI(
        model=GENERATIVE_MODEL,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )

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

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)

    rag_chain = (
        {
            "context": mmr_multi_retriever | format_docs,
            "question": RunnablePassthrough()
        } 
        | prompt
        | llm_generative 
        | StrOutputParser()
    )

    return rag_chain, mmr_multi_retriever


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
        "tipo": f"{SEARCH_TYPE.upper()}",
        "documentos": SEARCH_K,
        "diversidad": MMR_DIVERSITY_LAMBDA,
        "candidatos": MMR_FETCH_K,
        "umbral": None
    }           