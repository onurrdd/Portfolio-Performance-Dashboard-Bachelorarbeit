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
    # all-mpnet-base-v2 (768 Dim.) statt all-MiniLM-L6-v2 (384): Das kleinere Modell
    # stützt sich stärker auf oberflächliche Wortüberschneidung und stellte kurze
    # Meldungen, die die Wörter der Anfrage teilen, vor den Absatz, der dieselbe
    # Aussage mit anderen Worten trifft. Ein Modellwechsel ändert die Vektorlänge —
    # ein bestehender Index ist danach unbrauchbar und MUSS neu gebaut werden.
    embedding_model: str = "all-mpnet-base-v2"

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
    sec_slots: int = 0
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
    # Vorwärtiges Fenster (Tage NACH dem Anomaliedatum), enger als das rückwärtige.
    # Gesucht wird die Ursache, nicht die Folge — aber eine Ursache wird oft erst
    # danach verständlich beschrieben: Ein Bericht erscheint nach Börsenschluss in
    # nüchterner Unternehmenssprache, die einordnende Meldung („Tesla warnt vor
    # deutlich langsamerem Wachstum") folgt am nächsten oder übernächsten Tag. Ein
    # solcher Artikel ist keine Rückwärtsverzerrung: Das Ereignis liegt weiterhin
    # vor der Kursbewegung, nur seine Darstellung ist jünger — und für das Retrieval
    # brauchbarer als das Original. Zwei Tage sind der Erfahrungswert für den
    # redaktionellen Nachlauf; ein weiteres Fenster brächte vor allem Meldungen zu
    # späteren, unabhängigen Ereignissen.
    anomaly_window_days_after: int = 2

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
    # Nur 8-K: Ein 8-K meldet ein einzelnes Ereignis und ist entsprechend kurz;
    # 10-Q und 10-K sind Turnusberichte, die je Dokument ein Vielfaches an
    # Segmenten erzeugen. Da das Retrieval auf Segment- und nicht auf Dokument-
    # ebene ordnet, verdrängt ein Turnusbericht mit sechzig Segmenten die wenigen
    # Segmente des Ereignisberichts aus den top_k — auch dann, wenn er zum
    # Anomalietag nichts beiträgt. Beobachtet an MSFT 2023-04-26 (10-Q belegt die
    # Ränge 1–16, das 8-K mit den Quartalszahlen steht auf Rang 17) und an
    # AAPL 2026-07-31 (top-3 aus Verweisen auf Webseiten, Risikofaktoren und
    # "Changes in Internal Control").
    sec_forms: tuple = ("8-K",)
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

# --- Feste Anomalie-Stichprobe (EINE Quelle für Sparmodus, probe.py, eval_probe.py) ---
# PROBE_POOL: geordnete, feste Liste (Datum, Ticker) — die 2x2-Auswahl über
# Wissensschnitt (vor/nach GENERATOR_KNOWLEDGE_CUTOFF) und Quellentyp
# (Bericht aus dem Archiv vs. redaktioneller Nachrichtenartikel).
# PROBE_PICK: 1-basierte Nummern aus PROBE_POOL, die der Lauf tatsächlich
# bearbeitet — nicht zwingend zusammenhängend.
#   [1]          -> nur der erste Fall (aktive Experiment-Anomalie)
#   [1, 3]       -> erster und dritter Fall
#   [1, 2, 3, 4] -> die volle Stichprobe
# Der Dashboard-Sparmodus (callbacks/naive_llm.py::select_saving_mode_anomalies),
# probe.py und eval_probe.py lesen ausschließlich diese beiden Werte — eine
# Änderung hier schlägt überall gleich durch.
# Auswahl aus dem finalen Portfolio (portfolio_10_ticker_backup.csv): vier
# verschiedene Titel statt mehrfach desselben, je zwei Fälle vor und nach dem
# Wissensschnitt, Anstiege und Rückgänge gemischt. Für alle vier liegen bereits
# Quellen im Anomaliefenster im Cache — der Lauf kostet kein Alpha-Vantage-
# Kontingent. Ein Titel je Fall hält das Ergebnis unabhängig von den
# Besonderheiten einer einzelnen Aktie.
PROBE_POOL = (
    ("2021-01-27", "GME"),    # 1 · vor  Cutoff · +129,07 % (MAD-z +17,08)
    ("2023-04-26", "MSFT"),   # 2 · vor  Cutoff · +7,24 %   (MAD-z  +8,02)
    ("2025-12-02", "BA"),     # 3 · nach Cutoff · +10,15 %  (MAD-z +11,01)
    ("2026-07-31", "AAPL"),   # 4 · nach Cutoff · -7,35 %   (MAD-z  -6,43)
)
PROBE_PICK = [1, 2, 3, 4]

# Stabiler Schlüssel "YYYY-MM-DD|TICKER" der gewählten Fälle (chronologisch sortiert),
# wie ihn select_saving_mode_anomalies erwartet. Abgeleitet — nicht von Hand pflegen.
SAVING_MODE_PINNED = tuple(
    f"{d}|{t}" for d, t in sorted(
        (PROBE_POOL[i - 1] for i in PROBE_PICK), key=lambda dt: dt[0])
)
