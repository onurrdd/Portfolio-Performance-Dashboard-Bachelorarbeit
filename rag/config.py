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
    # (Ticker/Datum) holen wir zunächst einen Kandidaten-Pool und filtern
    # anschließend in Python. Dieser Wert ist nur die UNTERGRENZE des Pools: bei
    # aktiven Filtern umfasst der Pool den gesamten Index, damit die Filterung
    # erschöpfend ist (siehe RAGPipeline.retrieve_context).
    filter_pool_size: int = 50

    # Garantierte SEC-EDGAR-Slots unter den top_k Treffern. Kurze, sprachlich
    # "saubere" Alpha-Vantage-Zusammenfassungen liegen der generischen Retrieval-
    # Anfrage im Embedding-Raum oft näher als der tatsächlich ursächliche Absatz
    # in einem 35.000-Zeichen-8-K — der Absatz verliert dann rein durch Distanz
    # gegen mehrere thematisch irrelevante Kurznachrichten. Dieser Wert reserviert
    # die besten `sec_slots` SEC-Treffer (falls im Zeitfenster vorhanden) IMMER
    # unter den top_k; die übrigen Slots bleiben nach globaler Distanz offen.
    # 0 = deaktiviert (bisheriges Verhalten).
    sec_slots: int = 2
    # Innerhalb der reservierten SEC-Slots werden Chunks aus diesen Abschnitten
    # (rag/chunker.py::_tag_sections) vor der reinen Distanz bevorzugt: Der
    # ereigniserklärende Text eines 8-K steht i. d. R. in OUTLOOK/FINANCIAL-
    # SUMMARY/SUMMARYHIGHLIGHTS, nicht in den (bereits per _EXCLUDED_SECTIONS
    # gefilterten) Tabellenabschnitten. Leeres Tupel = keine Bevorzugung
    # (nur Distanz entscheidet, wie bisher).
    preferred_sections: tuple = ("OUTLOOK", "FINANCIALSUMMARY", "SUMMARYHIGHLIGHTS")

    # --- Anomalie-basiertes Retrieval ---
    # Rückwärtiges Zeitfenster (Tage VOR dem Anomaliedatum) für metadatengefiltertes
    # Retrieval — deckt auch eine Meldung ab, die schon vor der Kursreaktion erschien.
    anomaly_window_days: int = 3
    # Vorwärtiges Fenster (Tage NACH dem Anomaliedatum), bewusst enger als das
    # rückwärtige: Eine erklärende Meldung erscheint am Ereignistag oder mit höchstens
    # einem Tag Verzug (Redaktions-/Meldelag). Ein Dokument, das erst mehrere Tage
    # SPÄTER erscheint, kann die Kursbewegung nicht verursacht haben — wird es dennoch
    # als „Erklärung" verwendet, ist das eine Rückwärtsverzerrung (Look-Ahead-Bias):
    # das Modell erklärt die Ursache mit einer Folge des Ereignisses.
    anomaly_window_days_after: int = 1

    # --- Datenbeschaffung ---
    news_limit: int = 5

    # --- Alpha Vantage (historische Nachrichtenartikel) ---
    # Harter Aus-Schalter. Das Free-Kontingent liegt bei 25 Anfragen/Tag, d. h.
    # die Quelle ist praktisch einschüssig. Der TATSÄCHLICH wirksame Wert ist
    # (SAVING_MODE and alpha_vantage_enabled): die Quelle geht nur ans Netz, wenn
    # der Sparmodus die Anomalieliste ohnehin auf eine Handvoll Fälle begrenzt
    # (siehe SAVING_MODE unten und news/alpha_vantage.py). Ein voller Lauf über
    # alle Anomalien holt Alpha-Vantage-Inhalte ausschließlich aus dem Cache.
    alpha_vantage_enabled: bool = True

    # --- SEC EDGAR (historische Filings, kein API-Key nötig) ---
    sec_forms: tuple = ("8-K", "10-Q", "10-K")
    # Obergrenze der aus einem Filing uebernommenen Zeichen, bevor der
    # RecursiveCharacterTextSplitter segmentiert. Die Grenze ist bewusst weit
    # oberhalb der typischen Dokumentlaenge angesetzt: Ein Quartals-Update (8-K
    # EX-99.1) umfasst rund 35.000 Zeichen, und der ereigniserklaerende Ausblick
    # steht regelmaessig im hinteren Teil des Dokuments — hinter Kennzahlen-
    # tabellen und Bilanzteilen. Eine knappe Grenze schneidet daher genau jene
    # Passage ab, die die Kursbewegung begruendet, waehrend die inhaltsarmen
    # Tabellen erhalten bleiben. Da die Segmentierung ohnehin nachgelagert
    # erfolgt und das Retrieval nur top_k Chunks in den Prompt uebernimmt,
    # vergroessert eine hoehere Grenze allein den durchsuchbaren Index, nicht
    # den Prompt.
    max_document_chars: int = 50000


DEFAULT_CONFIG = RAGConfig()


# --- Sparmodus (Entwicklungs-/Iterationsmodus, KEINE Pipeline-Stellschraube) -----
# Bewusst außerhalb der frozen RAGConfig: kein technischer Retrieval-Parameter,
# sondern der Versuchsumfang. Bei True arbeiten ALLE LLM-Pfade (Naive-Tab, RAG-Tab,
# Vergleichs-Tab, RAGAS) so, als bestünde das Portfolio nur aus einer kleinen,
# über den Knowledge-Cutoff ausgewogenen Auswahl von Anomalien (siehe
# callbacks/naive_llm.py::select_saving_mode_anomalies). Für diese wenigen Fälle
# darf auch Alpha Vantage ans Netz (Tageskontingent bleibt bei 2–4 Anfragen sicher).
# Für die in der Thesis zählenden Zahlen wird der Modus abgeschaltet und EIN Lauf
# über alle Anomalien gemacht — erst nachdem die RAG-Qualität an der kleinen
# Auswahl bestätigt ist.
#
# Aufrufer lesen IMMER als Modul-Attribut (rag_config.SAVING_MODE), nie über
# `from rag.config import SAVING_MODE` — eval_probe.py und test_eval_contract.py
# schalten den Modus zur Laufzeit ab, und nur die Attribut-Form lässt diese eine
# Zuweisung auf alle Verbraucher (auch news/alpha_vantage.py) durchschlagen.
SAVING_MODE = True
# Zielgröße der Auswahl; wird gleichmäßig auf die beiden Cutoff-Seiten verteilt.
SAVING_MODE_ANOMALY_COUNT = 4
# Feste Auswahl als ("YYYY-MM-DD|TICKER", ...). Leer = automatisch wählen. Sobald
# eine Auswahl steht, hierhin kopieren (der Hinweis unter der Antwort listet die
# Schlüssel): so bleibt sie über mehrere Läufe hinweg identisch — sonst kann eine
# durch Alpha Vantage gewachsene Cache-Abdeckung die Auswahl zwischen zwei Läufen
# verschieben und die Ergebnisse wären nicht mehr vergleichbar.
SAVING_MODE_PINNED = (
    "2021-01-27|GME",    # vor  Cutoff · Nachrichtenartikel
    "2024-01-25|TSLA",   # vor  Cutoff · Quartalsbericht (Archiv)
    "2025-06-05|TSLA",   # nach Cutoff · Nachrichtenartikel
    "2026-07-23|TSLA",   # nach Cutoff · Quartalsbericht (Archiv)
)
