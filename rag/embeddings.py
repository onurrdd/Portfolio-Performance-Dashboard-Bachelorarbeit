"""
Embedding generation module using sentence-transformers (offline, no API key needed)
"""
from sentence_transformers import SentenceTransformer
from typing import List, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Generates embeddings using sentence-transformers (local, offline).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model.
        
        Args:
            model_name: Name of sentence-transformers model
                       (all-MiniLM-L6-v2: 384-dim, lightweight)
                       (all-mpnet-base-v2: 768-dim, higher quality)
        """
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Model loaded. Embedding dimension: {self.embedding_dim}")

    def embed_text(self, text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """
        Generate embedding(s) for text.
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Single embedding (list) or list of embeddings if input is list
        """
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        if is_single:
            return embeddings[0].tolist()
        else:
            return embeddings.tolist()

    def embed_chunks(self, chunks: List[dict]) -> List[dict]:
        """
        Add embeddings to chunk dictionaries.
        
        Args:
            chunks: List of chunk dicts with 'text' key
            
        Returns:
            Same chunks with added 'embedding' key
        """
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embed_text(texts)
        
        for chunk, embedding in zip(chunks, embeddings):
            chunk["embedding"] = embedding
        
        logger.info(f"Generated embeddings for {len(chunks)} chunks")
        return chunks


def generate_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> List[List[float]]:
    """
    Convenience function to generate embeddings for texts.
    
    Args:
        texts: List of text strings
        model_name: Embedding model to use
        
    Returns:
        List of embeddings (lists of floats)
    """
    generator = EmbeddingGenerator(model_name)
    return generator.embed_text(texts)
