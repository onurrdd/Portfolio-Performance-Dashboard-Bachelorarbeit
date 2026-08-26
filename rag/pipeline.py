"""
Zentrale RAG-Pipeline — orchestriert Quellen → Chunking → Embedding → Retrieval.

Die Quellen sind über die Registry news.base.get_sources() austauschbar: eine neue
Quelle (API/Feed/Scraper) implementiert news.base.NewsSource und wird dort eingetragen;
die Pipeline bleibt unverändert.

Metadaten-Filterung (Exposé 3.4): Retrieval kann nach Ticker und/oder Zeitfenster
eingeschränkt werden. Da FAISS keine native where-Filterung besitzt, wird ein größerer
Kandidaten-Pool geholt und anschließend in Python gefiltert (siehe RAGConfig.filter_pool_size).
"""
from typing import List, Dict, Optional, Set
import logging
from datetime import datetime, timedelta

from news.base import get_sources, published_to_epoch
from rag.config import RAGConfig, DEFAULT_CONFIG
from rag.cache import NewsCache
from rag.chunker import NewsChunker
from rag.embeddings import EmbeddingGenerator
from rag.vectorstore import FAISSVectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    """End-to-end RAG-Pipeline für die Retrieval-augmentierte Finanznachrichten-Analyse."""

    def __init__(self, sources: Optional[List] = None, config: RAGConfig = DEFAULT_CONFIG):
        """
        Args:
            sources: Liste von NewsSource-Objekten. None → Standard-Registry (get_sources()).
                     Erlaubt das Einschleusen eigener/mehrerer Quellen (Tests, Erweiterung).
            config:  Zentrale RAGConfig (Chunk-Größe, Retrieval-Parameter, Anomalie-Fenster).
        """
        self.config = config
        self.sources = sources if sources is not None else get_sources()
        self.cache = NewsCache(config.db_path)
        self.chunker = NewsChunker(chunk_size=config.chunk_size, overlap=config.chunk_overlap)
        self.embedder = EmbeddingGenerator(config.embedding_model)
        self.vectorstore = FAISSVectorStore(config.persist_dir, self.embedder.embedding_dim)

        logger.info(f"RAG Pipeline initialized ({len(self.sources)} source(s): "
                    f"{[getattr(s, 'name', '?') for s in self.sources]})")

    def index_news_for_tickers(self, tickers: List[str], news_limit: Optional[int] = None,
                               target_day: Optional[datetime] = None,
                               coverage_window_days: Optional[int] = None,
                               only_sources: Optional[Set[str]] = None) -> Dict:
        """Holt Nachrichten und indiziert nur NEUE Artikel.

        Cache-gesteuert und QUELLENSPEZIFISCH: Für jede Quelle wird einzeln geprüft, ob
        sie das Fenster [target_day ± window] bereits befüllt hat; nur die noch nicht
        abgedeckten Quellen werden angefragt. (Zuvor blockierte ein Treffer einer
        einzigen Quelle alle übrigen Quellen für dieses Fenster.)

        `only_sources`: Beschränkt den Abruf auf die genannten Quellennamen. Damit kann
        der automatische Hintergrund-Abruf auf kontingentfreie Quellen begrenzt werden,
        während kontingentierte Quellen dem manuell ausgelösten Abruf vorbehalten bleiben.

        target_day=None → heute. Bereits gecachte Artikel werden per `link` dedupliziert;
        nur neue Artikel werden embedded und in FAISS aufgenommen.
        """
        news_limit = news_limit or self.config.news_limit
        target_day = target_day or datetime.now()
        window = self.config.coverage_window_days if coverage_window_days is None else coverage_window_days
        logger.info(f"Starting indexing for tickers: {tickers} (target_day={target_day.date()}, ±{window}d)")

        total_new_articles = 0
        total_chunks = 0
        successful_tickers = 0
        skipped_tickers = 0

        win_start = target_day - timedelta(days=window)
        win_end = target_day + timedelta(days=window)

        # Quellen ohne Historie (supports_date_range = False, z. B. reines RSS) können
        # NUR etwas beitragen, wenn das Fenster die Gegenwart enthält — sonst liefern sie
        # zwangsläufig aktuelle Artikel, die der Fenster-Filter unten wieder verwirft.
        # Solche Quellen werden bei historischen Fenstern gar nicht erst angefragt
        # (spart je Anomalie und Ticker einen nutzlosen Netz-Request).
        window_covers_now = win_start <= datetime.now() <= win_end
        usable_sources = [
            s for s in self.sources
            # Default True: eine Quelle, die das Attribut nicht deklariert, wird wie bisher
            # immer angefragt — lieber ein überflüssiger Request als stilles Datenverlieren.
            if (window_covers_now or getattr(s, "supports_date_range", True))
            and (only_sources is None or getattr(s, "name", "") in only_sources)
        ]
        if len(usable_sources) < len(self.sources):
            skipped = [getattr(s, "name", "?") for s in self.sources if s not in usable_sources]
            logger.info(f"Historisches Fenster ({win_start.date()}…{win_end.date()}): "
                        f"Quelle(n) ohne Zeitraum-Unterstützung übersprungen: {skipped}")

        for ticker in tickers:
            raw_articles = []
            queried_any = False
            for src in usable_sources:
                src_name = getattr(src, "name", "?")
                # Abdeckung QUELLENSPEZIFISCH prüfen: hat diese Quelle das Fenster schon
                # befüllt, wird sie nicht erneut angefragt — andere Quellen aber schon.
                if self.cache.has_coverage(ticker, target_day, window, source=src_name):
                    logger.info(f"{ticker}/{src_name}: bereits abgedeckt "
                                f"(±{window}d um {target_day.date()}) — kein Fetch")
                    continue
                queried_any = True
                try:
                    # start/end übergeben: quellenspezifisch — tageshistorische Quellen
                    # (SEC EDGAR, Alpha Vantage) nutzen das Fenster gezielt; reine
                    # RSS-Feeds ohne Historie ignorieren es (liefern nur Aktuelles).
                    raw_articles.extend(src.fetch(ticker, news_limit, start=win_start, end=win_end))
                except Exception as e:
                    logger.warning(f"Source '{src_name}' failed for {ticker}: {e}")

            if not queried_any:
                skipped_tickers += 1
            if not raw_articles:
                continue

            # Zeitfenster-Filter VOR dem Cache-Schreiben: Quellen ohne Historie (z. B.
            # Yahoo-RSS) ignorieren start/end und liefern immer "aktuelle" Artikel, die
            # praktisch nie ins Anomaliefenster fallen. Solche Treffer werden gar nicht
            # erst in den Cache geschrieben (statt sie dort ungenutzt anzusammeln) — sie
            # könnten ohnehin nie in den Vektorindex gelangen (siehe Filter unten).
            epoch_from = int((target_day - timedelta(days=window)).timestamp())
            epoch_to = int((target_day + timedelta(days=window)).timestamp())
            in_window_articles = [
                a for a in raw_articles
                if epoch_from <= published_to_epoch(a.get("published", "")) <= epoch_to
            ]
            if not in_window_articles:
                continue

            new_articles = self.cache.upsert_articles(in_window_articles)  # dedup über link
            if new_articles:
                successful_tickers += 1
                total_new_articles += len(new_articles)
                chunks = self.chunker.process_articles(new_articles)
                chunks = self.embedder.embed_chunks(chunks)
                self.vectorstore.add_chunks(chunks)
                self.cache.mark_indexed([a.get("link") for a in new_articles])
                total_chunks += len(chunks)
                logger.info(f"Indexed {len(new_articles)} in-window articles for {ticker} "
                            f"({len(chunks)} chunks)")

        stats = {
            "tickers_processed": len(tickers),
            "tickers_successful": successful_tickers,
            "tickers_skipped": skipped_tickers,
            "total_articles": total_new_articles,   # NEU indizierte Artikel
            "total_chunks": total_chunks,
            "timestamp": datetime.now().isoformat(),
        }
        logger.info(f"Indexing complete: {stats}")
        return stats

    def retrieve_context(self, query: str, top_k: Optional[int] = None,
                         filter_ticker: Optional[str] = None,
                         date_from: Optional[datetime] = None,
                         date_to: Optional[datetime] = None) -> List[Dict]:
        """Retrieval mit optionaler Metadaten-Filterung (Ticker + Zeitfenster).

        Bei aktiven Filtern wird ein größerer Kandidaten-Pool geholt und danach
        gefiltert, damit trotz Filterung möglichst top_k Treffer übrig bleiben.
        """
        top_k = top_k or self.config.top_k
        logger.info(f"Retrieving for query: {query[:60]}... "
                    f"(ticker={filter_ticker}, from={date_from}, to={date_to})")

        has_filter = bool(filter_ticker or date_from or date_to)
        pool = self.config.filter_pool_size if has_filter else top_k

        query_embedding = self.embedder.embed_text(query)
        results = self.vectorstore.query_text(query, query_embedding, top_k=pool)

        if filter_ticker:
            results = [r for r in results if r["metadata"].get("ticker") == filter_ticker]

        if date_from or date_to:
            epoch_from = int(date_from.timestamp()) if date_from else None
            epoch_to = int(date_to.timestamp()) if date_to else None
            filtered = []
            for r in results:
                ep = r["metadata"].get("published_epoch", 0)
                if ep == 0:  # undatiert → bei Zeitfilter ausschließen
                    continue
                if epoch_from is not None and ep < epoch_from:
                    continue
                if epoch_to is not None and ep > epoch_to:
                    continue
                filtered.append(r)
            results = filtered

        return results[:top_k]

    def retrieve_for_anomaly(self, query: str, anomaly: Dict,
                             top_k: Optional[int] = None) -> List[Dict]:
        """Retrieval, das gezielt auf ein Anomalie-Ereignis abgestimmt ist.

        Filtert strikt nach dem verantwortlichen Ticker UND einem Zeitfenster
        (± config.anomaly_window_days) um das Anomaliedatum. KEIN Fallback: gibt es im
        Fenster keine passende Nachricht, wird eine leere Liste zurückgegeben (kein
        Rückgriff auf aktuelle, themenfremde Nachrichten).
        """
        top_k = top_k or self.config.top_k
        ticker = anomaly.get("responsible_ticker") or None
        date_from = date_to = None

        date_str = anomaly.get("date")
        if date_str:
            try:
                d = datetime.fromisoformat(str(date_str)[:10])
                w = timedelta(days=self.config.anomaly_window_days)
                date_from, date_to = d - w, d + w
            except ValueError:
                logger.warning(f"Unparsbares Anomaliedatum: {date_str}")

        return self.retrieve_context(query, top_k=top_k, filter_ticker=ticker,
                                     date_from=date_from, date_to=date_to)

    def format_context_for_llm(self, retrieved_chunks: List[Dict], max_tokens: int = 2000,
                               return_used: bool = False):
        """Formatiert die abgerufenen Chunks als Kontextblock für das LLM.

        `return_used=True` gibt zusätzlich die Liste der tatsächlich aufgenommenen Chunks
        zurück. Ohne das wäre die Budget-Kürzung unsichtbar: der Aufrufer wüsste nicht,
        wie viele Treffer es gar nicht in den Prompt geschafft haben, und würde stillschweigend
        eine Antwort auswerten, deren Kontext unvollständig ist.
        """
        if not retrieved_chunks:
            return ("", []) if return_used else ""

        context_parts = []
        used_chunks = []
        total_length = 0
        max_chars = max_tokens * 4

        for i, chunk in enumerate(retrieved_chunks, 1):
            metadata = chunk["metadata"]
            title = metadata.get("title", "")
            source = metadata.get("source", "")
            ticker = metadata.get("ticker", "")
            link = metadata.get("link", "")
            published = metadata.get("published", "")
            text = chunk["text"]

            part = f"""
[Source {i}] {ticker} - {title}
Published: {published}
Source: {source}
Link: {link}
Content: {text}
---"""
            if total_length + len(part) > max_chars:
                break
            context_parts.append(part)
            used_chunks.append(chunk)
            total_length += len(part)

        context = "\n".join(context_parts)
        dropped = len(retrieved_chunks) - len(context_parts)
        logger.info(f"Formatted context: {len(context)} chars from {len(context_parts)} chunks"
                    + (f" — {dropped} Chunk(s) wegen Budget verworfen" if dropped else ""))
        return (context, used_chunks) if return_used else context

    def get_stats(self) -> Dict:
        """Aktuelle Indexierungs-Statistik."""
        return {
            "indexed_documents": self.vectorstore.count_documents(),
            "cached_articles": self.cache.count(),
            "embedding_model": self.embedder.model_name,
            "embedding_dimension": self.embedder.embedding_dim,
        }
