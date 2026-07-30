"""
Segmentierung (Chunking) für RAG — Exposé 3.2.

Verwendet den RecursiveCharacterTextSplitter (LangChain): trennt Text hierarchisch
nach Absätzen → Sätzen → Wörtern, statt an beliebigen Zeichengrenzen. Dadurch bleibt
die semantische Integrität erhalten. Für kurze RSS-Headlines entsteht i. d. R. ein
einzelner Chunk; für lange Dokumente (Quartalsberichte, FED-Protokolle) greift die
hierarchische Segmentierung.

Jeder Chunk wird mit Metadaten angereichert (Zeitstempel, Ticker, Quelle), damit das
Retrieval später gezielt nach Ticker und Zeitfenster filtern kann.
"""
import logging
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from news.base import published_to_epoch

logger = logging.getLogger(__name__)


class NewsChunker:
    """Segmentiert Nachrichtenartikel in überlappende, metadaten-angereicherte Chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def process_article(self, article: Dict) -> List[Dict]:
        """Ein Artikel → Liste von Chunk-Dicts (Text + Metadaten).

        Erwartetes Artikel-Schema (news.base): title, summary, link, published, source, ticker.
        """
        full_text = f"{article.get('title', '')}. {article.get('summary', '')}".strip()
        text_chunks = self._splitter.split_text(full_text) if full_text else []

        published = article.get("published", "")
        epoch = published_to_epoch(published)

        chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunks.append({
                "text": chunk_text,
                "title": article.get("title", ""),
                "ticker": article.get("ticker", ""),
                "link": article.get("link", ""),
                "published": published,              # ISO-String (Anzeige)
                "published_epoch": epoch,            # Unix-Sek. (Zeitfenster-Filter)
                "source": article.get("source", ""),
                "chunk_index": i,
                "total_chunks": len(text_chunks),
            })
        return chunks

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Mehrere Artikel → flache Liste aller Chunks."""
        all_chunks = []
        for article in articles:
            all_chunks.extend(self.process_article(article))
        logger.info(f"Processed {len(articles)} articles into {len(all_chunks)} chunks")
        return all_chunks
