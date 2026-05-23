"""
FAISS vector database adapter for RAG retrieval (Windows-compatible alternative to Chroma)
"""
import faiss
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Manages FAISS vector database for storing and retrieving news embeddings.
    Lightweight, no C++ compilation needed on Windows.
    """

    def __init__(self, persist_dir: str = "./data/faiss", embedding_dim: int = 384):
        """
        Initialize FAISS client with persistent storage.
        
        Args:
            persist_dir: Directory for persistent storage
            embedding_dim: Dimension of embeddings (default 384 for all-MiniLM-L6-v2)
        """
        self.persist_dir = persist_dir
        self.embedding_dim = embedding_dim
        self.index_path = os.path.join(persist_dir, "index.faiss")
        self.metadata_path = os.path.join(persist_dir, "metadata.pkl")
        
        os.makedirs(persist_dir, exist_ok=True)

        # Initialize or load index
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            logger.info(f"Loading existing FAISS index from {self.index_path}")
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, 'rb') as f:
                self.metadata = pickle.load(f)
        else:
            logger.info(f"Creating new FAISS index with dimension {embedding_dim}")
            self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance
            self.metadata = []

        logger.info(f"FAISS initialized. Current size: {self.index.ntotal}")

    def add_chunks(self, chunks: List[Dict]) -> None:
        """
        Insert chunks into the vector store.
        
        Args:
            chunks: List of chunk dicts with 'embedding', 'text', and metadata
        """
        embeddings = []
        new_metadata = []

        for chunk in chunks:
            if 'embedding' not in chunk:
                logger.warning(f"Chunk missing 'embedding' key, skipping")
                continue

            embeddings.append(chunk["embedding"])
            
            # Store metadata
            new_metadata.append({
                "text": chunk.get("text", ""),
                "title": chunk.get("title", ""),
                "ticker": chunk.get("ticker", ""),
                "link": chunk.get("link", ""),
                "published": chunk.get("published", ""),
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("chunk_index", 0),
            })

        if embeddings:
            embeddings_array = np.array(embeddings, dtype='float32')
            self.index.add(embeddings_array)
            self.metadata.extend(new_metadata)
            self._save()
            logger.info(f"Added {len(embeddings)} chunks. Total: {self.index.ntotal}")

    def query(self, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Query the vector store for similar documents.
        
        Args:
            query_embedding: Query embedding (vector)
            top_k: Number of top results to return
            
        Returns:
            List of dicts with 'text', 'metadata', 'distance'
        """
        if self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return []

        query_array = np.array([query_embedding], dtype='float32')
        
        # Search
        distances, indices = self.index.search(query_array, min(top_k, self.index.ntotal))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx >= 0:  # -1 means no result
                result = {
                    "text": self.metadata[idx]["text"],
                    "metadata": {k: v for k, v in self.metadata[idx].items() if k != "text"},
                    "distance": float(dist),  # L2 distance
                }
                results.append(result)

        logger.info(f"Query returned {len(results)} results")
        return results

    def query_text(self, query_text: str, query_embedding: List[float], top_k: int = 5) -> List[Dict]:
        """
        Convenience method to query by text string.
        
        Args:
            query_text: Query text (for logging)
            query_embedding: Query embedding vector
            top_k: Number of results
            
        Returns:
            List of retrieved chunks with metadata
        """
        logger.info(f"Querying for: {query_text[:50]}...")
        return self.query(query_embedding, top_k)

    def count_documents(self) -> int:
        """
        Count documents in the index.
        
        Returns:
            Number of documents
        """
        return self.index.ntotal

    def _save(self) -> None:
        """Save index and metadata to disk."""
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, 'wb') as f:
            pickle.dump(self.metadata, f)
        logger.info(f"Index saved to {self.index_path}")

    def reset(self) -> None:
        """Reset the index."""
        self.index.reset()
        self.metadata = []
        self._save()
        logger.info("Index reset")


def create_vectorstore(persist_dir: str = "./data/faiss", embedding_dim: int = 384) -> FAISSVectorStore:
    """
    Convenience function to create a vector store.
    
    Args:
        persist_dir: Directory for storage
        embedding_dim: Embedding dimension
        
    Returns:
        Initialized FAISSVectorStore instance
    """
    store = FAISSVectorStore(persist_dir, embedding_dim)
    return store
