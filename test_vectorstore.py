from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from pydantic import SecretStr
import os

from config import EMBEDDING_MODEL, CHROMA_DB_PATH


embeddings = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    api_key=SecretStr(os.environ["GOOGLE_API_KEY"])
)

vectorstore = Chroma(
    embedding_function=embeddings, 
    persist_directory=CHROMA_DB_PATH
)


consulta = "¿Cuál es el DNI del arrendador Sergio Hernández?"
resultados = vectorstore.similarity_search(consulta, k=3)


print(f"Consulta: {consulta}\n")
print(f"Se encontraron {len(resultados)} resultados:\n")

for i, doc in enumerate(resultados, start=1):
    print(f"Resultado {i}:")
    print(f"Contenido: {doc.page_content}")
    print(f"Metadatos: {doc.metadata}\n")
    print()