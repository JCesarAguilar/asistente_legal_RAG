import os
from pathlib import Path

# Rutas del proyecto (ancladas al propio archivo, no ha rutas absolutas)
PROJECT_ROOT = Path(__file__).parent
DIRECTORIO_DOCUMENTOS = os.getenv("DIRECTORIO_DOCUMENTOS", str(PROJECT_ROOT / "documentos"))

# Proveedor de vectorstore: 
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "asistente-legal-rag")

# Modelos Openai
EMBEDDING_MODEL_OPENAI = "text-embedding-3-small"
QUERY_MODEL_OPENAI = "gpt-4o-mini"
GENERATIVE_MODEL_OPENAI = "gpt-4o"

# Configuracion del retriever
SEARCH_TYPE = "mmr"  # Tipo de búsqueda: "mmr" (Maximal Margin Relevance) o "similarity" (similaridad)
SEARCH_K = 5  # Número de documentos finales a devolver después de aplicar MMR
MMR_DIVERSITY_LAMBDA = 0.5  # Parámetro de diversidad para MMR (0.0 a 1.0, donde 0.0 es menos diverso y 1.0 es más diverso)
MMR_FETCH_K = 20  # Número de documentos a recuperar antes de aplicar MMR
ENSEMBLE_WEIGHTS = [0.7, 0.3] 
