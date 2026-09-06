from typing import List, Dict, Any, Optional, Union
import chromadb
from chromadb.config import Settings
import pickle
import os
import json
import threading
from chromadb.utils.embedding_functions import (
    SentenceTransformerEmbeddingFunction,
    OpenAIEmbeddingFunction  
)

# Chroma 会先缓存共享实例、再完成启动；冷启动并发构造客户端会提前访问租户。
_CLIENT_INIT_LOCK = threading.Lock()

def simple_tokenize(text):
    from nltk.tokenize import word_tokenize
    return word_tokenize(text)

class ChromaRetriever:
    """Vector database retrieval using ChromaDB"""
    def __init__(
        self, 
        collection_name: str = "memories",
        model_name: str = "all-MiniLM-L6-v2",
        embedder_provider: str = "sentence-transformers", 
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_client=None,
    ):
        """Initialize ChromaDB retriever.
        Args:
            collection_name: Name of the ChromaDB collection
            model_name: Name of the embedding model
            embedder_provider: Provider for embeddings ("sentence-transformers" or "openai")
            api_key: API key for OpenAI (required if embedder_provider is "openai")
            base_url: Base URL for OpenAI API (optional, defaults to official OpenAI endpoint)
                     Useful for using proxies or OpenAI-compatible services
        """
        with _CLIENT_INIT_LOCK:
            self.client = chromadb.Client(Settings(allow_reset=False, anonymized_telemetry=False))
        self.embedder_provider = embedder_provider
        self.model_client = model_client
        self.operation_stage = "amem_build_embedding"

        if model_client is not None:
            self.embedding_function = None  # 显式传向量，避免 Chroma 另开未计费的 SDK 客户端。
        elif embedder_provider == "openai":
            api_key = api_key or os.environ.get("OPENAI_API_KEY")
            base_url = base_url or os.getenv('OPENAI_API_BASE')
            if api_key is None:
                raise ValueError(
                    "API key is required for OpenAI embeddings. "
                    "Please provide it via api_key parameter or OPENAI_API_KEY environment variable."
                )
            openai_params = {
                "api_key": api_key,
                "model_name": model_name,
                "api_base": base_url
            }
            self.embedding_function = OpenAIEmbeddingFunction(**openai_params)
        else:  
            self.embedding_function = SentenceTransformerEmbeddingFunction(model_name=model_name)

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        
    def add_document(self, document: str, metadata: Dict, doc_id: str):
        """Add a document to ChromaDB with enhanced embedding using metadata.
        
        Args:
            document: Text content to add
            metadata: Dictionary of metadata including keywords, tags, context
            doc_id: Unique identifier for the document
        """
        # Build enhanced document content including semantic metadata
        enhanced_document = document
        
        # Add context information
        if 'context' in metadata and metadata['context'] != "General":
            enhanced_document += f" context: {metadata['context']}"
        
        # Add keywords information    
        if 'keywords' in metadata and metadata['keywords']:
            keywords = metadata['keywords'] if isinstance(metadata['keywords'], list) else json.loads(metadata['keywords'])
            if keywords:
                enhanced_document += f" keywords: {', '.join(keywords)}"
        
        # Add tags information
        if 'tags' in metadata and metadata['tags']:
            tags = metadata['tags'] if isinstance(metadata['tags'], list) else json.loads(metadata['tags'])
            if tags:
                enhanced_document += f" tags: {', '.join(tags)}"
        
        # Convert MemoryNote object to serializable format
        processed_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, list):
                processed_metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                processed_metadata[key] = json.dumps(value)
            else:
                processed_metadata[key] = str(value)
        
        # Store enhanced document content for better embedding
        processed_metadata['enhanced_content'] = enhanced_document
                
        # Use enhanced document content for embedding generation
        existing = self.collection.get(ids=[doc_id], include=["documents", "metadatas"])
        if existing["ids"] and existing["documents"][0] == enhanced_document:
            if existing["metadatas"][0] != processed_metadata:
                self.collection.update(ids=[doc_id], metadatas=[processed_metadata])
            return
        embedding = ({"embeddings": self.model_client.embed([enhanced_document], stage=self.operation_stage)}
                     if self.model_client is not None else {})
        self.collection.upsert(
            documents=[enhanced_document],
            metadatas=[processed_metadata],
            ids=[doc_id], **embedding,
        )
        
    def delete_document(self, doc_id: str):
        """Delete a document from ChromaDB.
        
        Args:
            doc_id: ID of document to delete
        """
        self.collection.delete(ids=[doc_id])
        
    def search(self, query: str, k: int = 5):
        """Search for similar documents.
        
        Args:
            query: Query text
            k: Number of results to return
            
        Returns:
            Dict with documents, metadatas, ids, and distances
        """
        inputs = ({"query_embeddings": self.model_client.embed([query], stage=self.operation_stage)}
                  if self.model_client is not None else {"query_texts": [query]})
        results = self.collection.query(
            **inputs,
            n_results=k
        )
        
        # Convert string metadata back to original types
        if 'metadatas' in results and results['metadatas'] and len(results['metadatas']) > 0:
            # First level is a list with one item per query
            for i in range(len(results['metadatas'])):
                # Second level is a list of metadata dicts for each result
                if isinstance(results['metadatas'][i], list):
                    for j in range(len(results['metadatas'][i])):
                        # Process each metadata dict
                        if isinstance(results['metadatas'][i][j], dict):
                            metadata = results['metadatas'][i][j]
                            for key, value in metadata.items():
                                try:
                                    # Try to parse JSON for lists and dicts
                                    if isinstance(value, str) and (value.startswith('[') or value.startswith('{')):
                                        metadata[key] = json.loads(value)
                                    # Convert numeric strings back to numbers
                                    elif isinstance(value, str) and value.replace('.', '', 1).isdigit():
                                        if '.' in value:
                                            metadata[key] = float(value)
                                        else:
                                            metadata[key] = int(value)
                                except (json.JSONDecodeError, ValueError):
                                    # If parsing fails, keep the original string
                                    pass
                        
        return results
