import os
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    LLM_PROVIDER,
    EMBEDDING_MODEL_GEMINI, QUERY_MODEL_GEMINI, GENERATIVE_MODEL_GEMINI,
    EMBEDDING_MODEL_OLLAMA, QUERY_MODEL_OLLAMA, GENERATIVE_MODEL_OLLAMA,
)


def obtener_embeddings() -> Embeddings:
    """Devuelve el modelo de embeddings según LLM_PROVIDER configurado."""
    if LLM_PROVIDER == "ollama":
        return OllamaEmbeddings(model=EMBEDDING_MODEL_OLLAMA)

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL_GEMINI,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )


def obtener_llm_queries() -> BaseChatModel:
    """Devuelve el LLM usado para reformular consultas (MultiQueryRetriever)."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=QUERY_MODEL_OLLAMA, temperature=0)

    return ChatGoogleGenerativeAI(
        model=QUERY_MODEL_GEMINI,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )


def obtener_llm_generativo() -> BaseChatModel:
    """Devuelve el LLM usado para generar la respuesta final al usuario."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=GENERATIVE_MODEL_OLLAMA, temperature=0)

    return ChatGoogleGenerativeAI(
        model=GENERATIVE_MODEL_GEMINI,
        api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
    )