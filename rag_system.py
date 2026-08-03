import os
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pydantic import SecretStr

from prompts.prompts import *
from services.retriever_builder import crear_retriever_base, crear_multi_query_retriever
from config import EMBEDDING_MODEL, CHROMA_DB_PATH, QUERY_MODEL, GENERATIVE_MODEL, SEARCH_TYPE, SEARCH_K, MMR_FETCH_K, MMR_DIVERSITY_LAMBDA

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
    multi_query_retriever = crear_multi_query_retriever(
        base_retriever=base_retriever,
        llm=llm_queries,
        prompt=multi_query_prompt
    )

    prompt = PromptTemplate.from_template(RAG_TEMPLATE)