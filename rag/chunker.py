"""
Text chunking and preprocessing module for RAG
"""
import re
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class NewsChunker:
    """
    Chunks news articles and metadata into suitable sizes for embedding and retrieval.
    """

    def __init__(self, chunk_size: int = 400, overlap: int = 50):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size of each chunk in characters
            overlap: Number of characters to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', '', text)
        return text.strip()

    def chunk_text(self, text: str, chunk_size: Optional[int] = None) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Override default chunk size
            
        Returns:
            List of text chunks
        """
        size = chunk_size or self.chunk_size
        text = self.clean_text(text)
        
        if len(text) <= size:
            return [text] if text else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + size
            
            # Try to break at sentence boundary if possible
            if end < len(text):
                last_period = text.rfind('.', start, end)
                if last_period > start + size // 2:
                    end = last_period + 1

            chunks.append(text[start:end])
            start = end - self.overlap

        return chunks

    def process_article(self, article: Dict) -> List[Dict]:
        """
        Process a single news article into chunks with metadata.
        
        Args:
            article: Article dict with keys: title, summary, link, published, source, ticker
            
        Returns:
            List of chunk dicts with text and metadata
        """
        chunks = []
        
        # Combine title and summary for processing
        full_text = f"{article.get('title', '')} {article.get('summary', '')}"
        
        # Chunk the text
        text_chunks = self.chunk_text(full_text)
        
        for i, chunk_text in enumerate(text_chunks):
            chunk_dict = {
                "text": chunk_text,
                "title": article.get("title", ""),
                "ticker": article.get("ticker", ""),
                "link": article.get("link", ""),
                "published": article.get("published", ""),
                "source": article.get("source", ""),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            }
            chunks.append(chunk_dict)
        
        return chunks

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Process multiple articles into chunks.
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Flattened list of chunk dictionaries
        """
        all_chunks = []
        for article in articles:
            chunks = self.process_article(article)
            all_chunks.extend(chunks)
        
        logger.info(f"Processed {len(articles)} articles into {len(all_chunks)} chunks")
        return all_chunks


def chunk_news_articles(articles: List[Dict], chunk_size: int = 400) -> List[Dict]:
    """
    Convenience function to chunk news articles.
    
    Args:
        articles: List of news articles
        chunk_size: Target chunk size in characters
        
    Returns:
        List of chunks with metadata
    """
    chunker = NewsChunker(chunk_size=chunk_size)
    return chunker.process_articles(articles)
