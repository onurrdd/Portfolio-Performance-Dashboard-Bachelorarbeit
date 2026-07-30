"""
Verzögerte (lazy) Initialisierung der RAG-Pipeline.

Die Pipeline wird NICHT beim App-Start geladen, sondern erst beim ersten tatsächlichen
Gebrauch (erster Klick im RAG-Tab). So starten zuerst App, Ticker, Anomalie-Erkennung und
Naive-LLM ohne Verzögerung; erst danach — und nur bei Bedarf — lädt RAG das schwere
Embedding-Modell (torch / sentence-transformers).

Wichtig: An die Callbacks wird der Provider (stabile Referenz) übergeben, nicht der
Pipeline-Wert. Die Callbacks holen die Pipeline via get(); dadurch sehen alle Callbacks
dieselbe, einmalig gebaute Instanz — anders als beim Übergeben eines None-Werts per Closure.
"""
import logging
import threading

from rag.config import RAGConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


class RAGProvider:
    """Baut die RAGPipeline erst beim ersten get()-Aufruf (thread-sicher)."""

    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self._config = config
        self._pipeline = None
        self._failed = False
        self._lock = threading.Lock()

    def get(self):
        """Liefert die Pipeline (baut sie beim ersten Aufruf). None, falls Initialisierung fehlschlug."""
        if self._pipeline is not None or self._failed:
            return self._pipeline
        with self._lock:
            if self._pipeline is None and not self._failed:
                try:
                    from rag.pipeline import RAGPipeline  # schwerer Import (torch) erst hier
                    logger.info("Initialisiere RAG-Pipeline (lazy, erster Gebrauch) ...")
                    self._pipeline = RAGPipeline(config=self._config)
                    logger.info(f"RAG-Pipeline bereit: {self._pipeline.get_stats()}")
                except Exception as e:
                    self._failed = True
                    logger.warning(f"RAG nicht verfügbar: {e}")
        return self._pipeline

    @property
    def ready(self) -> bool:
        return self._pipeline is not None
