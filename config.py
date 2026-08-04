import os
from pathlib import Path

# Rutas del proyecto (ancladas al propio archivo, no ha rutas absolutas)
PROJECT_ROOT = Path(__file__).parent
DIRECTORIO_DOCUMENTOS = os.getenv("DIRECTORIO_DOCUMENTOS", str(PROJECT_ROOT / "documentos"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))

# Proveedor de modelos: gemini o ollama
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")


# Modelos Gemini
EMBEDDING_MODEL_GEMINI = "models/gemini-embedding-001"
QUERY_MODEL_GEMINI = "gemini-3.5-flash-lite"       
GENERATIVE_MODEL_GEMINI = "gemini-3.5-flash"

# Modelos Ollama
EMBEDDING_MODEL_OLLAMA = "nomic-embed-text"
QUERY_MODEL_OLLAMA = "llama3.2"
GENERATIVE_MODEL_OLLAMA = "llama3.2"


# Configuracion del retriever
SEARCH_TYPE = "mmr"  # Tipo de búsqueda: "mmr" (Maximal Margin Relevance) o "similarity" (similaridad)
SEARCH_K = 5  # Número de documentos finales a devolver después de aplicar MMR
MMR_DIVERSITY_LAMBDA = 0.5  # Parámetro de diversidad para MMR (0.0 a 1.0, donde 0.0 es menos diverso y 1.0 es más diverso)
MMR_FETCH_K = 20  # Número de documentos a recuperar antes de aplicar MMR
ENSEMBLE_WEIGHTS = [0.7, 0.3] 
