"""
Main RAG pipeline - orchestrates fetching, chunking, embedding, and retrieval
"""
from typing import List, Dict, Optional, Tuple
import logging
from datetime import datetime
import os

# Import our modules
from news.yahoo_fetcher import YahooFinanceFetcher
from rag.chunker import NewsChunker
from rag.embeddings import EmbeddingGenerator
from rag.vectorstore import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline for financial news retrieval and augmented generation.
    """

    def __init__(
        self,
        persist_dir: str = "./data/faiss",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 400,
    ):
        """
        Initialize RAG pipeline components.
        
        Args:
            persist_dir: Directory for FAISS storage
            embedding_model: Sentence-transformers model name
            chunk_size: Character size for text chunks
        """
        self.fetcher = YahooFinanceFetcher()
        self.chunker = NewsChunker(chunk_size=chunk_size)
        self.embedder = EmbeddingGenerator(embedding_model)
        self.vectorstore = FAISSVectorStore(persist_dir, self.embedder.embedding_dim)

        logger.info("RAG Pipeline initialized")

    def index_news_for_tickers(self, tickers: List[str], news_limit: int = 5) -> Dict:
        """
        Fetch news for tickers and index into vector database.
        
        Args:
            tickers: List of ticker symbols
            news_limit: Max articles per ticker
            
        Returns:
            Dictionary with indexing stats
        """
        logger.info(f"Starting indexing for tickers: {tickers}")

        all_articles = self.fetcher.fetch_news_for_tickers(tickers, limit=news_limit)

        all_chunks = []
        total_articles = 0
        successful_tickers = 0

        for ticker, articles in all_articles.items():
            if articles:
                successful_tickers += 1
                total_articles += len(articles)
                
                # Chunk articles
                chunks = self.chunker.process_articles(articles)
                
                # Add embeddings
                chunks = self.embedder.embed_chunks(chunks)
                
                # Store in vector DB
                self.vectorstore.add_chunks(chunks)
                all_chunks.extend(chunks)

                logger.info(f"Indexed {len(articles)} articles for {ticker} ({len(chunks)} chunks)")

        stats = {
            "tickers_processed": len(tickers),
            "tickers_successful": successful_tickers,
            "total_articles": total_articles,
            "total_chunks": len(all_chunks),
            "timestamp": datetime.now().isoformat(),
        }

        logger.info(f"Indexing complete: {stats}")
        return stats

    def retrieve_context(self, query: str, top_k: int = 5, filter_ticker: Optional[str] = None) -> List[Dict]:
        """
        Retrieve relevant context for a query.
        
        Args:
            query: User query text
            top_k: Number of results to retrieve
            filter_ticker: Optional ticker to filter results
            
        Returns:
            List of retrieved chunks with metadata
        """
        logger.info(f"Retrieving context for query: {query[:60]}...")

        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query)

        # Query vector store
        results = self.vectorstore.query_text(query, query_embedding, top_k=top_k)

        # Filter by ticker if specified
        if filter_ticker:
            results = [r for r in results if r["metadata"].get("ticker") == filter_ticker]
            logger.info(f"Filtered results to ticker {filter_ticker}: {len(results)} results")

        return results

    def format_context_for_llm(self, retrieved_chunks: List[Dict], max_tokens: int = 2000) -> str:
        """
        Format retrieved chunks for LLM context.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            max_tokens: Approximate max tokens (4 chars ~ 1 token)
            
        Returns:
            Formatted context string
        """
        if not retrieved_chunks:
            return ""

        context_parts = []
        total_length = 0
        max_chars = max_tokens * 4

        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk["metadata"]
            title = metadata.get("title", "")
            source = metadata.get("source", "")
            ticker = metadata.get("ticker", "")
            link = metadata.get("link", "")
            text = chunk["text"]

            # Format chunk
            part = f"""
[Source {i}] {ticker} - {title}
Source: {source}
Link: {link}
Content: {text}
---"""

            if total_length + len(part) > max_chars:
                break

            context_parts.append(part)
            total_length += len(part)

        context = "\n".join(context_parts)
        logger.info(f"Formatted context: {len(context)} chars from {len(context_parts)} chunks")
        return context

    def query_with_rag(
        self,
        query: str,
        llm_callback,
        top_k: int = 5,
        filter_ticker: Optional[str] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Execute a query with RAG context.
        
        Args:
            query: User query
            llm_callback: Function that takes (query, context) and returns LLM response
            top_k: Number of retrieved chunks
            filter_ticker: Optional ticker filter
            
        Returns:
            Tuple of (llm_response, retrieved_chunks)
        """
        # Retrieve context
        retrieved = self.retrieve_context(query, top_k, filter_ticker)

        # Format for LLM
        context = self.format_context_for_llm(retrieved)

        if not context:
            logger.warning("No context retrieved for query")
            context = "No relevant news found."

        # Call LLM with context
        llm_response = llm_callback(query, context)

        return llm_response, retrieved

    def get_stats(self) -> Dict:
        """
        Get current indexing statistics.
        
        Returns:
            Dictionary with stats
        """
        doc_count = self.vectorstore.count_documents()
        return {
            "indexed_documents": doc_count,
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dim,
        }


def create_pipeline(
    persist_dir: str = "./data/faiss",
    embedding_model: str = "all-MiniLM-L6-v2",
    chunk_size: int = 400,
) -> RAGPipeline:
    """
    Convenience function to create RAG pipeline.
    
    Args:
        persist_dir: Directory for FAISS storage
        embedding_model: Embedding model name
        chunk_size: Text chunk size
        
    Returns:
        Initialized RAGPipeline instance
    """
    return RAGPipeline(persist_dir, embedding_model, chunk_size)
