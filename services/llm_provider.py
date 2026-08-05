import os
from pydantic import SecretStr
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import (
    EMBEDDING_MODEL_OPENAI, QUERY_MODEL_OPENAI, GENERATIVE_MODEL_OPENAI,
)


def obtener_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL_OPENAI,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"])
    )


def obtener_llm_queries() -> ChatOpenAI:
    return ChatOpenAI(
        model=QUERY_MODEL_OPENAI,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        temperature=0
    )


def obtener_llm_generativo() -> ChatOpenAI:
    return ChatOpenAI(
        model=GENERATIVE_MODEL_OPENAI,
        api_key=SecretStr(os.environ["OPENAI_API_KEY"]),
        temperature=0
    )