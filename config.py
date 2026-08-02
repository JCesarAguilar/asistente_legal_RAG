import os
from pathlib import Path

# Rutas del proyecto (ancladas al propio archivo, no ha rutas absolutas)
PROJECT_ROOT = Path(__file__).parent
DIRECTORIO_DOCUMENTOS = os.getenv("DIRECTORIO_DOCUMENTOS", str(PROJECT_ROOT / "documentos"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))

# Modelos
EMBEDDING_MODEL = "models/gemini-embedding-001"
QUERY_MODEL = "gemini-3.5-flash-lite"       # no tan potente 
GENERATIVE_MODEL = "gemini-3.5-flash"       # modelo mas potente para generar respuestas

# Configuracion del retriever
SEARCH_TYPE = "mmr"  # Tipo de búsqueda: "mmr" (Maximal Margin Relevance) o "similarity" (similaridad)
MMR_DIVERSITY_LAMBDA = 0.7  # Parámetro de diversidad para MMR (0.0 a 1.0, donde 0.0 es menos diverso y 1.0 es más diverso)
MMR_FETCH_K = 20  # Número de documentos a recuperar antes de aplicar MMR
SEARCH_K = 2  # Número de documentos finales a devolver después de aplicar MMR

# Configuracion alternativa para retriever hibrido
ENABLE_HYBRID_SEARCH = True  # Habilitar búsqueda híbrida (MMR + Similaridad)
SIMILARITY_THRESHOLD = 0.75  # Umbral de similitud para filtrar documentos similares (0.0 a 1.0, donde 1.0 es idéntico)