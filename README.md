# ⚖️ Asistente Legal RAG — Perú

Sistema de consulta normativa inteligente para abogados peruanos, construido con Retrieval Augmented Generation (RAG). Permite consultar en lenguaje natural la Constitución y los principales códigos y leyes del Perú, obteniendo respuestas precisas con citas exactas de artículo y código.

🔗 **Demo en vivo:** [agrega aquí tu link de Streamlit Cloud]

![Asistente Legal RAG Demo](agrega-aqui-tu-screenshot.png)

---

## 📋 Descripción

Un abogado que necesita ubicar información específica en cientos de páginas de códigos y expedientes pierde tiempo valioso buscando manualmente. Este sistema automatiza esa búsqueda usando comprensión semántica: entiende el significado de la pregunta, no solo coincidencias de palabras clave, y devuelve respuestas fundamentadas citando el artículo y código exactos.

El sistema **no sustituye el criterio de un abogado** — es una herramienta de apoyo para localizar y sintetizar información normativa rápidamente. Cada respuesta incluye una advertencia para verificar la vigencia de la norma en fuentes oficiales.

## ✨ Funcionalidades

- 📄 **Ingesta de documentos legales** — carga códigos, leyes y expedientes en PDF, incluyendo subcarpetas organizadas por materia
- 🔍 **OCR inteligente selectivo** — detecta automáticamente páginas escaneadas con texto de mala calidad y las reprocesa con Tesseract en alta resolución, sin tocar las páginas que ya tienen buen texto
- 🏷️ **Clasificación automática por materia** — cada fragmento se etiqueta según su área legal (Civil, Penal, Laboral, Constitucional, etc.) según la estructura de carpetas
- 🧠 **Retriever híbrido avanzado** — combina búsqueda MMR (Maximal Marginal Relevance), reformulación automática de consultas (MultiQuery) y búsqueda por similitud simple en un `EnsembleRetriever` ponderado
- 🌐 **Vector store en la nube (Pinecone)** — la base de conocimiento vive en infraestructura remota, accesible desde cualquier despliegue, no depende de una máquina local
- 🛡️ **Diseñado para no alucinar** — el sistema responde únicamente con base en los fragmentos recuperados; si no encuentra la información, lo indica explícitamente en vez de inventar contenido
- 🎨 **Interfaz interactiva** construida con Streamlit, mostrando la materia legal y fuente de cada fragmento citado

## 🏗️ Arquitectura

El proyecto sigue el principio de separación de responsabilidades (SOLID), con cada módulo enfocado en una única función:

```
asistente_legal_RAG/
├── app.py                        # Punto de entrada de la aplicación
├── config.py                     # Configuración centralizada
├── vector_stores.py              # Orquestador de ingesta (carga → OCR → chunks → vectorstore)
├── rag_system.py                 # Orquestador de consulta (retriever → generación)
├── services/
│   ├── document_loader.py        # Carga de PDFs (incluye subcarpetas)
│   ├── ocr_quality.py            # Detección de OCR de baja calidad
│   ├── ocr_processor.py          # Re-procesamiento de OCR con Tesseract
│   ├── text_splitter_service.py  # División de documentos en chunks
│   ├── metadata_enricher.py      # Etiquetado automático por materia legal
│   ├── vectorstore_builder.py    # Construcción del índice vectorial (Pinecone)
│   ├── retriever_builder.py      # Construcción de retrievers (MMR, MultiQuery, Ensemble)
│   ├── document_formatter.py     # Formateo de resultados para el prompt del LLM
│   └── llm_provider.py           # Proveedor de modelos (embeddings y LLMs de OpenAI)
├── prompts/
│   └── prompts.py                # Templates de prompts del sistema
├── ui/
│   └── streamlit_ui.py           # Interfaz de usuario
└── requirements.txt
```

Esta separación permite, por ejemplo, cambiar de proveedor de LLM o de vector store sin modificar la lógica de negocio del resto del sistema.

## 🛠️ Tecnologías

- **Python 3.11**
- **LangChain** — orquestación del pipeline RAG
- **OpenAI API** (`langchain-openai`) — embeddings (`text-embedding-3-small`) y modelos generativos (`gpt-4o` / `gpt-4o-mini`)
- **Pinecone** — base de datos vectorial en la nube
- **Streamlit** — interfaz de usuario
- **Tesseract OCR** (`pytesseract`, `pdf2image`) — reconocimiento óptico de caracteres para documentos escaneados
- **pypdf** — extracción de texto desde PDF

## 🚀 Instalación local

**1. Clona el repositorio**

```bash
git clone https://github.com/tu-usuario/asistente-legal-rag.git
cd asistente-legal-rag
```

**2. Crea y activa un entorno virtual**

```bash
python3.11 -m venv venv
source venv/bin/activate   # En Windows: venv\Scripts\activate
```

**3. Instala las dependencias**

```bash
pip install -r requirements.txt
```

**4. Instala Tesseract OCR** (dependencia del sistema, no de Python)

```bash
# macOS
brew install tesseract tesseract-lang poppler

# Ubuntu/Debian
sudo apt install tesseract-ocr tesseract-ocr-spa poppler-utils
```

**5. Configura tus variables de entorno**

Crea un archivo `.env` en la raíz del proyecto:

```
OPENAI_API_KEY=tu-clave-de-openai
PINECONE_API_KEY=tu-clave-de-pinecone
PINECONE_INDEX_NAME=asistente-legal-rag
```

**6. Organiza tus documentos**

Coloca tus PDFs organizados por materia legal dentro de `documentos/`:

```
documentos/
├── civil/
├── penal/
├── laboral/
├── constitucional/
└── procesal/
```

**7. Construye la base de conocimiento**

```bash
python vector_stores.py
```

Este paso carga los documentos, corrige el OCR de baja calidad, los divide en fragmentos y genera sus embeddings en Pinecone. Solo es necesario correrlo una vez, o cada vez que se agreguen documentos nuevos.

**8. Ejecuta la aplicación**

```bash
streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`

## 📖 Uso

1. Escribe tu consulta legal en lenguaje natural
2. El sistema recupera los fragmentos normativos más relevantes de la base de conocimiento
3. Genera una respuesta citando el código y artículo exacto de cada norma referenciada
4. Muestra los documentos fuente utilizados, incluyendo su materia legal, archivo de origen y página

**Ejemplo de consulta:**

> ¿Qué pasa legalmente si un inquilino no paga la renta durante varios meses?

**Respuesta esperada:** cita los artículos aplicables del Código Civil (resolución de contrato por falta de pago) y, cuando corresponda, del Código Procesal Civil (procedimiento de desalojo), organizados por materia.

## 🧪 Validación del sistema

Durante el desarrollo se realizaron pruebas específicas para verificar la confiabilidad del sistema:

| Prueba                       | Objetivo                                                                    | Resultado                                                |
| ---------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------- |
| Consulta de definición legal | Precisión en la cita del artículo correcto                                  | ✅ Cita exacta y verificable                             |
| Artículo inexistente         | Verificar que el sistema no alucina                                         | ✅ Responde honestamente que no encuentra la información |
| Consulta multi-materia       | Verificar que el retriever cruza distintas áreas legales cuando corresponde | ✅ Combina Civil y Procesal Civil coherentemente         |

Durante el desarrollo se comparó el rendimiento entre un modelo de lenguaje local (Ollama, 3B parámetros) y un modelo en la nube (OpenAI GPT-4o-mini). El modelo local, si bien más económico, presentó alucinaciones al no seguir estrictamente las instrucciones del prompt — generando artículos y contenido normativo inexistente. Esto llevó a la decisión de usar un LLM más capaz para la generación de respuestas en un dominio de alta responsabilidad como el legal.

## 🗺️ Posibles mejoras futuras

- [ ] Filtrado de búsqueda por materia legal seleccionada por el usuario
- [ ] Detección automática de la materia de la consulta
- [ ] Modo de análisis y recomendación (no solo recuperación de información)
- [ ] Soporte para más códigos y leyes especializadas
- [ ] Extracción estructurada de entidades clave (partes, montos, fechas) de documentos cargados

## ⚠️ Disclaimer

Este sistema es una herramienta de apoyo a la búsqueda de información legal y **no constituye asesoría legal**. Las respuestas generadas deben ser verificadas por un profesional del derecho antes de ser aplicadas a cualquier caso real. Se recomienda confirmar la vigencia de toda norma citada en el Sistema Peruano de Información Jurídica (SPIJ).

## 📄 Licencia

Este proyecto es de uso educativo/portafolio.

---

Desarrollado por [Julio César Aguilar](https://github.com/JCesarAguilar)
