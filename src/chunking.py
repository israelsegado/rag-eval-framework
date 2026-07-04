'''Técnicas de chunking: 
        * Tamaño fijo
        * Con solapamiento
        * Basado en oraciones y párrafos
        * De ventana deslizante
        * Recursivo de separadores ✅
        * Semántico ✅
        * Jerárquico (padre e hijo) ✅
        * Basado en estructura y elementos (los documentos no son solo texto plano) <-
        * Consciente del código
        * Estructurado (Unstructured) <- Si se hace habría que hacer otro .py para elegir si separar por PdfReader o UnstructuredLoader
        * Agentic
        * Tardío <-

Para evaluar chunking:
        * Recall@k / Precision@k
        * MRR / nDCG

URL: https://www.glukhov.org/es/rag/retrieval/chunking-strategies-in-rag/#chunking-de-tama%C3%B1o-fijo
'''
import os
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chonkie import SemanticChunker as ChonkieSemanticChunker
from collections import defaultdict
from chonkie.embeddings import AutoEmbeddings, BaseEmbeddings


class EmbeddingsWithTokenizer(BaseEmbeddings):
    def __init__(self, embeddings_model, tokenizer_name: str = "cl100k_base"):
        super().__init__()
        self._embeddings_model = embeddings_model
        self._tokenizer_name = tokenizer_name

    def embed(self, text: str):
        return self._embeddings_model.embed(text)

    def embed_batch(self, texts: list[str]):
        return self._embeddings_model.embed_batch(texts)

    @property
    def dimension(self):
        return self._embeddings_model.dimension

    def get_tokenizer(self):
        return self._tokenizer_name


class RecursiveChunker:
    def split(self, documents: List[Document], params: Dict[str, Any]) -> List[Document]:
        """Hace split de lo documentos mediante RecursiveCharacterTextSplitter en base a los parámetros obtenidos del JSON

        Args:
            documents (List[Document]): lista de documentos para hacer chunk
            params (_type_): es el json, un diccionario con estructura "str": cualquier cosa a obtener con get

        Returns:
            List[Document]: lista con documento hecho chunk
        """
        chunking_params = params["chunking"]["params"]

        chunk_size = chunking_params.get("chunk_size", 500)
        chunk_overlap = chunking_params.get("chunk_overlap", 50)
        separators = chunking_params.get("separators", ["\n\n", "\n", ".", ",", " ", ""])

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=separators
        )

        chunks = splitter.split_documents(documents)

        for i, chunk in enumerate(chunks):
            chunk.metadata["id"]=i
            chunk.metadata["chunk_size"]=chunk_size
            chunk.metadata["chunk_overlap"]=chunk_overlap
            chunk.metadata["chunking_strategy"]="recursive_character_text_splitter"
            chunk.metadata["is_parent_child"]=False
        return chunks

class SemanticChunkerStrategy:
    def split(self, documents:List[Document], params: Dict[str, Any]) -> List[Document]:
        """Hace split de lo documentos mediante SemanticChunker en base a los parámetros obtenidos del JSON.
        En chunking semántico juntamos los documentos en uno solo por archivo ya que necesita ver el documento 
        completo para detectar rupturas semánticas reales.

        Args:
            documents (List[Document]): lista de documentos parahacer chunk
            params (_type_): es el json, un diccionario con estructura "str": cualquier cosa a obtener con get

        Returns:
            List[Document]: lista con documento hecho chunk
        """
        chunking_params = params["chunking"]["params"]
        embeddings_model = chunking_params.get("embedding")

        if not embeddings_model:
            raise ValueError("Para SemanticChunking se requiere un modelo de embedding configurado.")
        
        #chonkie necesita el nombre del modelo, no un HuggingFaceEmbedding(model_name="") por ejemplo
        if isinstance(embeddings_model, str):
            model_name = embeddings_model
        else:
            model_name = getattr(embeddings_model, "model_name", embeddings_model.__class__.__name__) #guardamos solo el nombre del modelo

        threshold_amount = chunking_params.get("threshold", 0.8)
        chunk_size = chunking_params.get("chunk_size", 30) #chunks bajos
        similarity_window = chunking_params.get("similarity_window", 2)
        skip_window = chunking_params.get("skip_window", 0)

        chonkie_embeddings = AutoEmbeddings.get_embeddings(embeddings_model)

        splitter = ChonkieSemanticChunker(
            embedding_model=EmbeddingsWithTokenizer(chonkie_embeddings),
            threshold=threshold_amount,
            chunk_size=chunk_size,
            similarity_window=similarity_window,
            skip_window=skip_window
        )

        # juntamos los documentos para mejor búsqueda semántica
        docs_by_name = defaultdict(list)
        for doc in documents:
            name = doc.metadata.get("document_name", "unknown_document")
            docs_by_name[name].append(doc.page_content)

        chunks = []
        chunk_id = 0

        for doc_name, doc_content in docs_by_name.items():
            full_text = "\n".join(doc_content)
            chonkie_chunks = splitter.chunk(full_text)

            for c in chonkie_chunks:
                
                new_metadata = {
                    "document_name" : doc_name,
                    "id" : chunk_id,
                    "chunk_size" : chunk_size,
                    "chunking_strategy" : "semantic_chonkie",
                    "threshold_amount" : threshold_amount,
                    "similarity_window" : similarity_window,
                    "skip_window" : skip_window,
                    "embedding_model_name" : model_name,
                    "is_parent_child" : False
                }
                doc = Document(
                    page_content = c.text,
                    metadata = new_metadata
                )
                chunks.append(doc)
                chunk_id += 1

        return chunks

class ParentChildChunker:
    def split(self, documents: List[Document], params: Dict[str, Any]) -> List[Document]:
        """Hace split de lo documentos mediante ParentChild en base a los parámetros obtenidos del JSON.
        Para ello vamos a almacenar el texto padre (el grande) dentro de los metadatos del hijo al que se le
        va a hacer el embedding.

        Args:
            documents (List[Document]): lista de documentos parahacer chunk
            params (_type_): es el json, un diccionario con estructura "str": cualquier cosa a obtener con get

        Returns:
            List[Document]: lista con documento hecho chunk de los hijos
        """  
        chunking_params = params["chunking"]["params"]
        
        parent_chunks = chunking_params.get("parent_chunks", 1000)
        parent_overlap = chunking_params.get("parent_overlap", 200)

        child_chunks = chunking_params.get("child_chunks", 200)
        child_overlap = chunking_params.get("child_overlap", 50)

        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunks,
            chunk_overlap=parent_overlap,
        )
        
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunks,
            chunk_overlap=child_overlap
        )

        parent_index = 0
        child_index = 0
        final_chunks = []
        for doc in documents:
            parent_fragments = parent_splitter.split_documents([doc])

            for fragment in parent_fragments:
                parent_text = fragment.page_content
                parent_id = f"parent_{parent_index}"

                parent_metadata = doc.metadata.copy()
                parent_metadata["id"] = parent_id
                parent_metadata["parent_chunk_size"] = parent_chunks
                parent_metadata["parent_overlap"] = parent_overlap
                parent_metadata["chunking_strategy"] = "hierarchical_parent_child"
                parent_metadata["chunk_role"] = "parent"
                parent_metadata["is_parent_child"] = False

                final_chunks.append(Document(
                    page_content=parent_text,
                    metadata=parent_metadata
                ))

                child_fragments = child_splitter.split_documents([fragment])
                
                for child in child_fragments:
                    new_metadata = doc.metadata.copy()
                    new_metadata["id"]=f"child_{child_index}"
                    new_metadata["parent_id"]=parent_id
                    new_metadata["parent_chunk_size"]=parent_chunks
                    new_metadata["parent_overlap"]=parent_overlap
                    new_metadata["child_chunk_size"]=child_chunks
                    new_metadata["child_overlap"]=child_overlap
                    new_metadata["chunking_strategy"]="hierarchical_parent_child"
                    new_metadata["chunk_role"]="child"
                    new_metadata["is_parent_child"]=True

                    child_doc = Document(
                        page_content=child.page_content,
                        metadata=new_metadata
                    )

                    final_chunks.append(child_doc)
                    child_index += 1
                parent_index += 1

        return final_chunks

CHUNKER_REGISTRY = {
    "recursive" :   RecursiveChunker,
    "semantic":     SemanticChunkerStrategy,
    "hierarchical": ParentChildChunker
}
