from langchain_ollama import ChatOllama

llm = ChatOllama(model="llama3.2", temperature=0)
respuesta = llm.invoke("Hola, ¿cómo estás? Responde en una sola oración.")
print(respuesta.content)