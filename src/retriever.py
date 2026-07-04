'''
Se configurarán 2 técnicas diferentes de hacer búsquedas de índices a azure, además de optar por poner reranking:
* Vectorial
* Híbrida

Los campos de los índices es la siguiente:
* id: índice del chunk -> String - Recuperable
* text: el texto que se encuentra en este chunk -> String - Recuperable - Se puede buscar
* document_name: nombre del documento -> String - Se puede filtrar - Se puede buscar
* embedding: modelo de embedding -> SingleCollection - nº dimensiones del modelo - HNSW - Cosine - Se puede buscar
* page_number: número de la página (semantic chunker no tiene) -> String - Recuperable - Se puede buscar
* is_parent_text: nos dice si se ha hecho el chunking mediante parent-child -> Booleano - Recuperable - Clasificable
* metadata: aquí irán todos los datos que almacenaba el chunk inicial -> String - Recuperable
'''
import os
import json
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_core.documents import Document
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_community.vectorstores import AzureSearch
from sentence_transformers import CrossEncoder
from azure.search.documents.models import VectorizedQuery

#------------------- CARGA DE APIS ----------------------
load_dotenv()
AZURE_SEARCH_CLIENT = os.getenv("AZURE_SEARCH_CLIENT")
#========================================================

def azure_to_document(azure_result: Dict[str, Any], azure_client: SearchClient = None) -> Document:
    """Al hacer retrieve lo que obtenemos de azure es un JSON con todos los metadatos de los campos configurados:
    {
        "id": 1,
        "text": -------,
        "document_name": -------
    }

    Para que RAGAS nos entienda tenemos que pasar los JSON de azure a un document de langchain como:
    {
        page_content: -----------,
        metadata: {
            "metadata1": --------,
            "metadata2": --------
        }
    }

    Args:
        azure_result (Dict[str, Any]): _description_

    Returns:
        Document: _description_
    """    
    raw_metadata = azure_result.get("metadata", "{}")
    metadata_dict = json.loads(raw_metadata) if isinstance(raw_metadata, str) else raw_metadata
    
    page_content = azure_result.get("text", "")
    metadata_dict["search_score"] = azure_result.get("@search.score", 0.0)

    return Document(page_content=page_content, metadata=metadata_dict)


def deduplicate_documents(documents: List[Document]) -> List[Document]:
    seen = set()
    deduplicated = []

    for doc in documents:
        dedup_key = doc.metadata.get("parent_id") or doc.metadata.get("id") or doc.page_content
        if dedup_key in seen:
            continue

        seen.add(dedup_key)
        deduplicated.append(doc)

    return deduplicated

class VectorSearch:
    def __init__(self, azure_client: SearchClient, embedding_model):
        self.client = azure_client
        self.embedding_model = embedding_model

    def retrieve(self, query: str, params: Dict[str, Any]) -> List[Document]:
        top_k = params.get("top_k", 5)
        search_filter = "is_parent_child eq true" if params.get("chunking_strategy") == "hierarchical" else None
        query_embeded = self.embedding_model.embed_query(query)

        vectorizedQuery = VectorizedQuery(
            vector=query_embeded,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )

        results = self.client.search(
            search_text=None,
            vector_queries=[vectorizedQuery],
            top=top_k,
            filter=search_filter,
            select=["id", "text", "document_name", "page_number", "metadata"]
        )

        return deduplicate_documents([azure_to_document(r, self.client) for r in results])
    
class HybridSearch:
    def __init__(self, azure_client: SearchClient, embedding_model):
        self.client = azure_client
        self.embedding_model = embedding_model

    def retrieve(self, query: str, params: Dict[str, Any]) -> List[Document]:
        top_k = params.get("top_k", 5)
        search_filter = "is_parent_child eq true" if params.get("chunking_strategy") == "hierarchical" else None
        query_embeded = self.embedding_model.embed_query(query)

        vectorizedQuery = VectorizedQuery(
            vector=query_embeded,
            k_nearest_neighbors=top_k,
            fields="embedding"
        )

        results = self.client.search(
            search_text=query,
            vector_queries=[vectorizedQuery],
            top=top_k,
            filter=search_filter,
            select=["id", "text", "document_name", "page_number", "metadata"]
        )

        return deduplicate_documents([azure_to_document(r, self.client) for r in results])
    
def reranking(query: str, documents: List[Document], params: Dict[str, Any], reranker_model, azure_client: SearchClient = None) -> List[Document]:
    """Reordena los documentos recuperados por Azure en base a su relevancia real con la query usando un CrossEncoder.

    Args:
        query (str): pregunta del usuario
        documents (List[Document]): documentos recuperados por el retriever
        params (Dict[str, Any]): parámetros del config
        reranker_model: modelo CrossEncoder con método .predict()

    Returns:
        List[Document]: documentos reordenados por relevancia, con rerank_score en metadata
    """
    top_n = params.get("top_n", 5)

    pairs = [(query, doc.page_content) for doc in documents]

    reranking_scores = reranker_model.predict(pairs)

    # Empareja cada documento con su score y ordena de mayor a menor relevancia
    ranked = sorted(zip(documents, reranking_scores), key=lambda x: x[1], reverse=True)
    
    reranked_docs = []
    for doc, score in ranked:
        doc.metadata["rerank_score"] = float(score)
        reranked_docs.append(doc)

    if top_n:
        reranked_docs = reranked_docs[:top_n]

    reranked_docs = deduplicate_documents(reranked_docs)    

    for doc in reranked_docs:
        if not doc.metadata.get("is_parent_child", False):
            continue    
        parent_id = doc.metadata.get("parent_id")
        if not parent_id:
            continue    
        parent_doc = azure_client.get_document(key=parent_id)
        doc.metadata["parent_document_id"] = parent_id
        doc.page_content = parent_doc.get("text", doc.page_content) 
    return reranked_docs


RETRIEVER_REGISTRY = {
    "vector": VectorSearch,
    "hybrid": HybridSearch
}
