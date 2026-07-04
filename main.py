import os
import sys
import json
import numpy as np
import pandas as pd
import pymupdf
from pathlib import Path
from langchain_core.documents import Document
from dotenv import load_dotenv
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
import time
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import re
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from azure.core.exceptions import AzureError, ResourceNotFoundError
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    HnswParameters,
    VectorSearchProfile,
    BM25SimilarityAlgorithm,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from requests.exceptions import RequestException
from ssl import SSLError
import asyncio
import logging
from datalake import DataLakeAdapter, DataLakeConfig

#-------------------------------------- CARGA DE CLAVES -------------------------------------------
load_dotenv()

HUGGINF_FACE_API = os.getenv("HUGGINF_FACE_API")
AZURE_SEARCH_CLIENT = os.getenv("AZURE_SEARCH_CLIENT")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ------------- Declaración de los modelos (unificamos en un sitio para posibles cambios)
'''
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
VECTOR_SIZE_EMBEDDING_MODEL = 768
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "phi3:mini"
'''
EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=1024)
VECTOR_SIZE_EMBEDDING_MODEL = 1024 #reducido para no usar tanto espacio (máx. 3072)
CROSS_ENCODER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEN_MODEL = "gpt-5.4-mini"

#===================================================================================================

#-------------------------------------- LEEMOS DOCUMENTOS ------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

CORPUS_FOLDER = CURRENT_DIR / "data" / "corpus"
CORPUS_DATALAKE = "/corpus"

async def download_corpus_from_datalake():
    datalake = DataLakeAdapter(
        DataLakeConfig(connection_string=os.getenv("CONNECTION_STRING"))
    )
    datalake._logger = logging.getLogger("datalake")

    try:
        CORPUS_FOLDER.mkdir(parents=True, exist_ok=True)

        blobs = await datalake.list_blobs_by_prefix(CORPUS_DATALAKE)
        pdf_blobs = [blob for blob in blobs if blob["name"].lower().endswith(".pdf")]

        print(f"Descargando {len(pdf_blobs)} PDFs desde Data Lake...")

        for blob in pdf_blobs:
            blob_name = blob["name"].lstrip("/")
            local_path = CORPUS_FOLDER / Path(blob_name).name

            storage_path = datalake._normalize_storage_path(blob_name)
            blob_client = datalake._get_blob_client(storage_path)

            download_stream = await blob_client.download_blob()
            local_path.write_bytes(await download_stream.readall())

            print(f"PDF descargado: {local_path.name}")

    finally:
        await datalake.close()


asyncio.run(download_corpus_from_datalake())

def clean_extracted_text(text: str) -> str:
    """Del texto extraído lo limpia para que sea más fácil de entender por el LLM y no tener errores StringIO en RAGAS

    Args:
        text (str): texto sucio

    Returns:
        str: texto limpio
    """    
    if not text:
        return ""

    text = re.sub(r'^[^\n]+?(\s*\.\s*){4,}\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-[ \t]*\n[ \t]*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()

all_documents = []
for pdf_path in CORPUS_FOLDER.glob("*.pdf"):
    print(f"Leyendo documento {pdf_path}...")
    reader = pymupdf.open(pdf_path)

    for id_page, page in enumerate(reader):
        raw_text = page.get_text()
        cleaned_text = clean_extracted_text(raw_text)
        doc = Document(
            page_content=cleaned_text,
            metadata={
                "document_name" : pdf_path.name,
                "page_number" : id_page+1
            }
        )
        all_documents.append(doc)
    reader.close()
#===================================================================================================

#--------------------------------- DEFINICIÓN DE FUNCIONES -----------------------------------------
#azure.core.exceptions.ServiceRequestError -> error de pérdida de conexión con Azure
@retry(
    stop=stop_after_attempt(10), 
    wait=wait_exponential(multiplier=1, min=5, max=60), 
    retry=retry_if_exception_type(AzureError)
    )
def safe_retrieval(retrieval_instance, query, params):
    return retrieval_instance.retrieve(query, params)

@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=2, min=5, max=60),
    retry=retry_if_exception_type((AzureError, RequestException, OSError, SSLError)),
    reraise=True
)
def safe_upload_documents(azure_client: SearchClient, documents):
    return azure_client.upload_documents(documents=documents)

def create_custom_index(index_name: str=AZURE_SEARCH_CLIENT):
    """crea un índice nuevo en caso de que no exista, si existe uno previo lo elimina,
    así podemos quitar la espera de limpiar el índice ya que tarda mucho en liberar caché

    Args:
        index_name (str, optional): nombre del índice a crear. Defaults to AZURE_SEARCH_CLIENT.
    """    
    if not AZURE_SEARCH_ENDPOINT or not AZURE_SEARCH_KEY:
        raise ValueError("Error crítico: Faltan las variables de entorno AZURE_SEARCH_ENDPOINT o AZURE_SEARCH_KEY")
    
    index_client = SearchIndexClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    try:
        print("Comprobando si existe algún índice...")
        index_client.delete_index(index_name)
        print("Sí existe, borrando índice...")
        time.sleep(500)
        print("Indice anterior eliminado.")
    except ResourceNotFoundError:
        print("No existe ningún índice previo.")
        pass

    fields = [
        SimpleField(
            name="id", 
            type="Edm.String", 
            key=True, 
            filterable=False, 
            retrievable=True, 
            sortable=False, 
            facetable=False
        ),
        SearchableField(
            name="text", 
            type="Edm.String", 
            analyzer_name="standard.lucene", 
            filterable=False, 
            retrievable=True, 
            sortable=False, 
            facetable=False
        ),
        SearchableField(
            name="document_name", 
            type="Edm.String", 
            analyzer_name="standard.lucene", 
            filterable=True, 
            retrievable=True, 
            sortable=False, 
            facetable=False
        ),
        SearchField(
            name="embedding",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=VECTOR_SIZE_EMBEDDING_MODEL,
            vector_search_profile_name="vector-profile-oll",
            filterable=False,
            retrievable=False
        ),
        SearchableField(
            name="page_number", 
            type="Edm.String", 
            analyzer_name="standard.lucene", 
            filterable=True, 
            retrievable=True, 
            sortable=False, 
            facetable=False
        ),
        SimpleField(
            name="is_parent_child", 
            type="Edm.Boolean", 
            filterable=True, 
            retrievable=True, 
            sortable=False, 
            facetable=True
        ),
        SimpleField(
            name="metadata", 
            type="Edm.String", 
            filterable=False, 
            retrievable=True, 
            sortable=False, 
            facetable=False
        )
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="vector-config-1780302141124",
                kind="hnsw",
                parameters={
                    "metric": "cosine",
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500
                }
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="vector-profile-oll",
                algorithm_configuration_name="vector-config-1780302141124"
            )
        ]
    )

    similarity = BM25SimilarityAlgorithm()

    index_definition = SearchIndex(
        name=index_name,
        fields=fields,
        vector_search=vector_search,
        similarity=similarity
    )   

    index_client.create_index(index_definition) #creación índice en azure

    return SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=index_name,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )


def index_chunks(azure_client: SearchClient, final_chunks: List[Document], embedding_model, batch_size: int=20):
    """Recorre la lista de chunks finales, convierte su objeto Document de Langchain a un diccionario
    configurado en el cliente de azure y lo sube en lotes

    Args:
        azure_client (SearchClient): cliente de azure
        final_chunks (List[Document]): lista con todos los chunks
        embedding_model (_type_): embedding a usar
        batch_size (int, optional): tamaño de lotes. Defaults to 20.
    """    
    payload_batch = []

    for idx, chunk in enumerate(final_chunks):
        metadata_dict = dict(chunk.metadata)
        chunk_role = metadata_dict.get("chunk_role", "chunk")

        payload_doc = {
            "id": str(metadata_dict.get("id", "")),
            "document_name": metadata_dict.get("document_name", ""),
            "page_number": str(metadata_dict.get("page_number", "")),
            "text" : chunk.page_content,
            "is_parent_child": bool(metadata_dict.get("is_parent_child", False)),
            "metadata": json.dumps(metadata_dict)
        }

        if chunk_role != "parent":
            payload_doc["embedding"] = embedding_model.embed_query(chunk.page_content)

        payload_batch.append(payload_doc)

        if len(payload_batch) >= batch_size or idx == len(final_chunks) - 1:
            if len(payload_batch) > 0:
                print(f"Subiendo lote de {len(payload_batch)} chunks a Azure (Progreso: {idx + 1}/{len(final_chunks)})...")
                safe_upload_documents(azure_client, payload_batch)
                payload_batch = []

#llm = ChatOllama(model=GEN_MODEL, temperature=0.1)
llm = ChatOpenAI(model=GEN_MODEL, temperature=0.1, max_completion_tokens=1024)

SYSTEM_PROMPT = """
Eres un asistente de preguntas y respuestas basado exclusivamente en documentos proporcionados como contexto.
Debes responder siempre en español.
Tu tarea es responder a la pregunta del usuario usando únicamente la información contenida en el contexto proporcionado, pero sin mencionar nunca el contexto ni la información aportada en la respuesta.
No utilices conocimiento externo, internet, suposiciones ni información no respaldada por los documentos.
Antes de responder, comprueba si el contexto responde de forma directa a la pregunta completa.
Responde cuando el contexto contenga evidencia suficiente para contestar el núcleo de la pregunta de forma fiel y útil.
Puedes hacer inferencias simples, comparaciones y correcciones de premisas falsas cuando estén directamente apoyadas por fragmentos concretos del contexto.
No completes huecos con conocimiento general ni con fragmentos solo relacionados por tema.
La evidencia debe corresponder a la misma entidad, figura, procedimiento, documento, producto o tipo jurídico por el que se pregunta. No sustituyas una entidad por otra parecida.
No respondas sobre un comité distinto, una excedencia distinta, un procedimiento distinto, un producto distinto, una política distinta o una fase distinta aunque pertenezcan al mismo tema general.
Si la pregunta contiene una premisa falsa y el contexto recuperado permite refutar esa premisa concreta, responde "No" y corrige la premisa con la información respaldada.
Si la premisa solo puede refutarse usando información relacionada pero no exacta, responde exactamente:
"No dispongo de información suficiente en los documentos para responder."
Si la pregunta pide pasos, plazos, fechas, mercados, procedimientos, causas, consecuencias, comparaciones o requisitos, esos elementos deben estar respaldados por el contexto para poder afirmarlos.
Si el contexto permite responder la idea principal y los matices secundarios no cambian la respuesta, responde solo con lo respaldado y no añadas detalles no disponibles.
Si faltan pasos, plazos, responsables, fases o condiciones que forman parte central de la pregunta, responde exactamente:
"No dispongo de información suficiente en los documentos para responder."
Si falta información esencial para responder el núcleo de la pregunta, responde exactamente:
"No dispongo de información suficiente en los documentos para responder."
No expliques qué falta, no digas que algo "no aparece", "no se indica", "no se especifica", "no consta", "no figura" o "no está disponible".
Si la pregunta contiene una afirmación incorrecta, corrígela explícitamente basándote solo en la información del contexto y después responde a la pregunta cuando sea posible.
Si el contexto no contiene evidencia suficiente para responder, responde exactamente:
"No dispongo de información suficiente en los documentos para responder."
No incluyas fórmulas introductorias, metacomentarios ni expresiones de relleno como "Según el contexto", "Según el fragmento", "De acuerdo con el contexto", "En el contexto", "En la información aportada", "La información aportada indica", "con la información disponible", "a partir de los documentos", "los documentos indican", "el fragmento señala" ni expresiones similares.
Empieza directamente con la respuesta sustantiva, sin justificar de dónde sale.
Cada afirmación de la respuesta debe estar apoyada por el contexto recuperado. No uses fragmentos cercanos temáticamente para responder algo distinto a lo preguntado.
Si la pregunta pide una definición, da una definición o descripción directa; no sustituyas la definición por una explicación general de importancia, gestión o contexto.
Si la pregunta pide una conducta, requisito, responsable, fecha o dato concreto, responde únicamente con ese dato concreto y no añadas otros datos cercanos que no respondan a la pregunta.
Si la respuesta no puede darse por falta de información suficiente, escribe únicamente la frase exacta de rechazo y no añadas fuentes.
Después de una respuesta válida, incluye un apartado titulado "Fuentes" donde indiques los documentos concretos en los que se basa la respuesta.
Formato de respuesta:
- Primero, la respuesta directa.
- Después, si procede, un apartado:
Fuentes:
- Nombre del documento o identificador disponible en el contexto
No cites un documento como fuente si no has usado información procedente de ese documento para construir la respuesta, 
usa los metadatos para identificar exitósamente el nombre del documento en específico.
Mantén un tono preciso, profesional y conciso acorde a los documentos de contexto.
"""

def generate_answer(question: str, context: str) -> str:
    '''
    Genera una respuesta basada en la pregunta y el contexto proporcionados.

    Args:
        question: La pregunta del usuario.
        context: El contexto relevante extraído de los documentos.

    Returns:
        La respuesta generada por el modelo.
    '''
    user_message =(
        f"CONTEXTO: {context}\n\n"
        f"PREGUNTA: {question}"
    )

    messages = [
        {"role" : "system", "content" : SYSTEM_PROMPT},
        {"role" : "user", "content" : user_message}
    ]

    response = llm.invoke(messages)
    return response.content.strip()

def generate_context(chunks: List[Document]) -> str:
    """estructura la información para que el LLM tenga la información mejor organizada

    Args:
        chunks (List[Document]): los chunks a organizar

    Returns:
        str: el texto ya estructurado
    """    
    fragments = []
    for i, chunk in enumerate(chunks, 1):
        fragment = (f"[Fragmento {i}] (documento: {chunk.metadata.get('document_name', '')}) "
                    f"\npágina número {chunk.metadata.get('page_number', '')}\n"
                    f"contenido: {chunk.page_content}")
        fragments.append(fragment)
    
    return "\n\n---\n\n".join(fragments)
#===================================================================================================

#------------------------------------ CARGA JSON CONFIG --------------------------------------------

CONFIG_FOLDER = CURRENT_DIR / "config" 
INTERMEDIATE_FILEPATH = CURRENT_DIR / "results" / "intermediate_files" # para comprobar que no ha sido procesado

processed = {f.name for f in INTERMEDIATE_FILEPATH.glob("*.json")}

# coge el primer archivo json que no ha sido procesado
for config_file in CONFIG_FOLDER.glob("*.json"):
    print(f"Procesando archivo: {config_file.name}")
    if config_file.name in processed:
        print("Este archivo ya ha sido procesado.")
        continue

    print(f"Procesando: {config_file.name}")
    with open(config_file, "r", encoding="utf-8") as f:
        data_config = json.load(f)
    print(f"Fichero extraído: {data_config}")

#===================================================================================================

#------------------------------------- CREACIÓN ÍNDICE ---------------------------------------------

    try:
        print("Creando índice...")
        azure_client = create_custom_index(AZURE_SEARCH_CLIENT)
        print("Índice creado.")

#===================================================================================================

#-------------------------------- CHUNKING DE DOCUMENTOS -------------------------------------------

        from src.chunking import CHUNKER_REGISTRY
        chunking_method = data_config.get("chunking").get("strategy")

        print(f"Aplicando la técnica de chunking {chunking_method}...")
        if chunking_method in CHUNKER_REGISTRY:
            chunker_instance = CHUNKER_REGISTRY[chunking_method]()
            final_chunks = chunker_instance.split(all_documents, data_config)
        print(f"Chunking completado con un número total de chunks: {len(final_chunks)}.")
#===================================================================================================

#-------------------------------- SUBIDA CHUNKS A AZURE --------------------------------------------

        chunks_indexed = index_chunks(azure_client=azure_client, final_chunks=final_chunks, embedding_model=EMBEDDING_MODEL)
        print("Chunks subidos correctamente al índice de Azure.")
#===================================================================================================

#------------------ RETRIEVER DE CHUNKS DE LAS QUERIES DE GOLDEN_DATASET Y GENERACIÓN --------------

#---------- RETRIEVE
        print("Recuperando los documentos y generando una respuesta...")

        from src.retriever import RETRIEVER_REGISTRY, reranking

        retrieve_method = data_config.get("retrieval", "{}").get("method", "hybrid")
        retrieval_instance=RETRIEVER_REGISTRY[retrieve_method](azure_client, EMBEDDING_MODEL) if retrieve_method in RETRIEVER_REGISTRY else None

        reranker_model = CrossEncoder(CROSS_ENCODER) if data_config.get("retrieval", {}).get("params", {}).get("reranking", False) else None

#------------ LOOP SOBRE GOLDEN_DATASET

        GOLDEN_FILEPATH = CURRENT_DIR / "data" / "golden_dataset" / "golden_dataset_single-turn.jsonl"

        with open(GOLDEN_FILEPATH, 'r', encoding='utf-8') as f:
            golden_queries = [json.loads(line) for line in f]

        results =[]

        for it, row in enumerate(golden_queries, 1):
            print(f"Actualmente recuperando y generando una respuesta para la pregunta número {it}/{len(golden_queries)}...")

            query = row['user_input']
            retrieval_params = dict(data_config.get("retrieval", {}).get("params", {}))
            retrieval_params["chunking_strategy"] = chunking_method

            #Retrieval
            retrieval_instance = RETRIEVER_REGISTRY[retrieve_method](azure_client, EMBEDDING_MODEL)
            retrieved_docs=safe_retrieval(retrieval_instance, query, retrieval_params)

            #Reranking
            if reranker_model:
                retrieved_docs = reranking(query, retrieved_docs, retrieval_params, reranker_model, azure_client)

            #Generation
            context = generate_context(retrieved_docs)
            answer = generate_answer(query, context)

            #Building the results list
            results.append({
                "user_input": query,
                "response": answer,
                "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
                "retrieved_metadata": [doc.metadata for doc in retrieved_docs],
                "reference_contexts": row["reference_contexts"],
                "reference": row["reference"],
                "metadata": row["metadata"]
            })

        print("Recuperación y generación completadas con éxito!")
        print(f"Un ejemplo: \n{results[3]}")
        print(f"\n\nLa pregunta: \n{results[3]["user_input"]} \nTuvo la respuesta del LLM: \n{results[3]["response"]}")

        with open(INTERMEDIATE_FILEPATH / f"{data_config["experiment_name"]}.json", 'w', encoding='utf-8') as f:
            json.dump({"data_config": data_config, "results": results}, f, ensure_ascii=False, indent=2)

        print(f"Resultados temporales listos para la evaluación guardados en: {INTERMEDIATE_FILEPATH}")

    except:
        print(f"XXX Error con {config_file.name} XXX")

        import traceback
        traceback.print_exc()
#===================================================================================================
