"""
Zentrale Konfiguration der RAG-Pipeline.

Ein einziger Ort für alle Stellschrauben (Chunk-Größe, Retrieval-Parameter,
Anomalie-Fenster) — erleichtert Thesis-Experimente (z. B. Chunk-Size-Ablationen),
ohne mehrere Module anfassen zu müssen.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class RAGConfig:
    # --- Segmentierung / Chunking (Exposé 3.2: 500–1000 / 50–100) ---
    # Für kurze RSS-Headlines faktisch ein No-Op (1 Chunk pro Artikel),
    # aber vorbereitet für lange Dokumente (Quartalsberichte, FED-Protokolle).
    chunk_size: int = 800
    chunk_overlap: int = 100

    # --- Vektorspeicher (FAISS, lokal, Windows-kompatibel) ---
    persist_dir: str = "./data/faiss"
    embedding_model: str = "all-MiniLM-L6-v2"

    # --- Persistenter Nachrichten-Cache (SQLite) ---
    db_path: str = "./data/news_cache.db"
    # Abdeckungsfenster (± Tage): gibt es für (Ticker, Tag) bereits Artikel, wird
    # nicht erneut aus dem Netz geholt. Beim manuellen Fetch ist Tag = heute.
    coverage_window_days: int = 3

    # --- Retrieval ---
    top_k: int = 5
    # FAISS besitzt keine native Metadaten-Filterung. Bei aktiven Filtern
    # (Ticker/Datum) holen wir zunächst einen größeren Kandidaten-Pool und
    # filtern anschließend in Python, damit top_k Treffer übrig bleiben.
    filter_pool_size: int = 50

    # --- Anomalie-basiertes Retrieval ---
    # Zeitfenster (± Tage) um ein Anomaliedatum für metadatengefiltertes Retrieval.
    anomaly_window_days: int = 3

    # --- Datenbeschaffung ---
    news_limit: int = 5


DEFAULT_CONFIG = RAGConfig()
