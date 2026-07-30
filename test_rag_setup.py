#!/usr/bin/env python
"""
End-to-End-Smoke-Test der RAG-Pipeline (ohne Dash).

Prüft: Imports, Quellen-Modularität (2 Quellen), Indexierung, semantisches Retrieval,
Ticker-Filter, Zeitfenster-Filter und das anomalie-basierte Retrieval inkl. Fallback.

Nutzt ein isoliertes, temporäres FAISS-Verzeichnis, damit der echte Index
(data/faiss) unberührt bleibt.
"""
import sys
import os
import shutil
import tempfile
from datetime import datetime, timedelta

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

print("=" * 64)
print("RAG End-to-End Smoke Test")
print("=" * 64)

# --- [1] Imports -----------------------------------------------------------
print("\n[1] Imports ...")
try:
    from dataclasses import replace
    from rag.config import DEFAULT_CONFIG
    from rag.pipeline import RAGPipeline
    from news.base import get_sources, NewsSource
    from news.yahoo_fetcher import YahooRSSSource
    print("OK  RAG-Module importiert")
except Exception as e:
    print(f"FAIL Import: {e}")
    sys.exit(1)


# --- Test-Doppel: eine zweite, datierte Quelle (Modularitätsnachweis) ------
class FakeReportSource:
    """Simuliert eine historische/datierte Quelle (z. B. Quartalsbericht-Feed).

    Zeigt: (a) eine neue Quelle = 1 Klasse mit name+fetch, (b) datierte Artikel,
    damit der Zeitfenster-Filter tatsächlich greifen kann.
    """
    name = "fake_report"

    def __init__(self, published_dt: datetime):
        self._published = published_dt.isoformat()

    def fetch(self, ticker, limit=10, start=None, end=None):
        return [{
            "title": f"{ticker} Q2 earnings beat expectations",
            "summary": (f"{ticker} reported quarterly revenue above analyst estimates, "
                        "driven by strong product demand and margin expansion."),
            "link": f"https://example.com/{ticker}/report",  # ticker-spezifisch → keine Dedup-Kollision
            "published": self._published,
            "source": self.name,
            "ticker": ticker,
        }]


# --- [2] Quellen-Registry --------------------------------------------------
print("\n[2] Quellen-Registry ...")
sources = get_sources()
print(f"OK  get_sources() -> {[s.name for s in sources]}")
assert any(isinstance(s, YahooRSSSource) for s in sources), "YahooRSSSource fehlt in Registry"

# --- [3] Pipeline mit isoliertem Index + 2 Quellen -------------------------
print("\n[3] Pipeline-Init (isolierter Index, 2 Quellen) ...")
tmp_dir = tempfile.mkdtemp(prefix="rag_test_faiss_")
anomaly_day = datetime(2026, 7, 20)
try:
    cfg = replace(DEFAULT_CONFIG, persist_dir=tmp_dir,
                  db_path=os.path.join(tmp_dir, "cache.db"))
    pipeline = RAGPipeline(
        sources=[YahooRSSSource(), FakeReportSource(published_dt=anomaly_day)],
        config=cfg,
    )
    print(f"OK  Pipeline init: {pipeline.get_stats()}")

    # --- [4] Indexierung: NUR Fenster-Artikel landen im Index -------------
    print("\n[4] Indexierung (target_day = Anomalietag) ...")
    stats = pipeline.index_news_for_tickers(["AAPL", "MSFT"], news_limit=3, target_day=anomaly_day)
    print(f"OK  {stats}")
    assert stats["total_chunks"] > 0, "Keine Chunks erzeugt"
    # Nur die FakeReport-Artikel (im Fenster) werden indiziert; RSS (heute) nur gecacht.
    idx = pipeline.get_stats()["indexed_documents"]
    cached = pipeline.cache.count()
    print(f"    indexiert={idx}, gecacht={cached}")
    assert idx >= 2, "FakeReport (im Fenster) nicht indiziert?"
    # Schlüsselbeleg für den Fenster-Filter: JEDER indizierte Chunk liegt im Fenster.
    lo = int((anomaly_day - timedelta(days=3)).timestamp())
    hi = int((anomaly_day + timedelta(days=3)).timestamp())
    all_hits = pipeline.retrieve_context("news", top_k=20)
    assert all(lo <= h["metadata"].get("published_epoch", 0) <= hi for h in all_hits), \
        "Out-of-window Artikel wurde indiziert — Fenster-Filter greift nicht!"
    assert cached >= idx, "Cache sollte mind. so viele Artikel wie der Index enthalten"
    print("OK  Fenster-Filter: nur Fenster-Artikel indiziert, RSS nur gecacht")

    # --- [5] Semantisches Retrieval ---------------------------------------
    print("\n[5] Retrieval 'quarterly earnings revenue' (top_k=3) ...")
    res = pipeline.retrieve_context("quarterly earnings revenue", top_k=3)
    assert res, "Kein Retrieval-Ergebnis"
    top = res[0]["metadata"]
    print(f"OK  {len(res)} Treffer. Top: [{top.get('ticker')}] {top.get('title', '')[:50]}")
    assert isinstance(top.get("published_epoch"), int), "published_epoch fehlt/kein int"

    # --- [6] Ticker-Filter -------------------------------------------------
    print("\n[6] Ticker-Filter (AAPL) ...")
    res_aapl = pipeline.retrieve_context("earnings", top_k=5, filter_ticker="AAPL")
    assert res_aapl, "Kein AAPL-Treffer"
    assert all(r["metadata"].get("ticker") == "AAPL" for r in res_aapl), "Ticker-Filter undicht"
    print(f"OK  {len(res_aapl)} Treffer, alle AAPL")

    # --- [7] Zeitfenster-Filter -------------------------------------------
    print("\n[7] Zeitfenster-Filter (± 3 Tage um Anomalietag) ...")
    res_win = pipeline.retrieve_context("earnings", top_k=5,
                                        date_from=anomaly_day - timedelta(days=3),
                                        date_to=anomaly_day + timedelta(days=3))
    assert any(r["metadata"].get("source") == "fake_report" for r in res_win), \
        "Datierter Report nicht im Zeitfenster gefunden"
    assert all(r["metadata"].get("published_epoch", 0) != 0 for r in res_win), \
        "Undatierter Treffer im Zeitfilter durchgerutscht"
    print(f"OK  {len(res_win)} Treffer im Fenster")

    # --- [8] Anomalie-Retrieval: KEIN Fallback ----------------------------
    print("\n[8] retrieve_for_anomaly (ohne Fallback) ...")
    # 8a: Treffer im Fenster (verantwortlicher Ticker AAPL, Datum = anomaly_day)
    anomaly_hit = {"date": anomaly_day.strftime("%Y-%m-%d"), "responsible_ticker": "AAPL"}
    res_a = pipeline.retrieve_for_anomaly("earnings", anomaly_hit, top_k=3)
    assert res_a, "Anomalie-Retrieval leer trotz passendem Report"
    print(f"OK  {len(res_a)} Treffer im Anomaliefenster")
    # 8b: Fenster ohne passende Nachricht -> LEER (kein Fallback auf aktuelle News)
    anomaly_miss = {"date": "2020-01-15", "responsible_ticker": "AAPL"}
    res_b = pipeline.retrieve_for_anomaly("earnings", anomaly_miss, top_k=3)
    assert res_b == [], f"Kein Fallback erwartet, war {len(res_b)} Treffer"
    print("OK  Kein Fallback: außerhalb des Fensters leeres Ergebnis")

    # --- [9] Persistenter Cache: Abdeckung + Dedup + Skip -----------------
    print("\n[9] Nachrichten-Cache (Abdeckung / Dedup / Skip) ...")
    assert pipeline.cache.count() > 0, "Cache leer nach Indexierung"
    # 9a: Abdeckungs-Abfrage (Ticker + Datum)
    assert pipeline.cache.has_coverage("AAPL", anomaly_day, 3), \
        "has_coverage sollte True sein (FakeReport AAPL am Anomalietag)"
    assert not pipeline.cache.has_coverage("AAPL", datetime(2020, 1, 15), 3), \
        "has_coverage sollte False sein (kein Artikel 2020)"
    print("OK  has_coverage: Anomalietag=abgedeckt, 2020=nicht abgedeckt")
    # 9b: Re-Indexierung für den Anomalietag → beide Ticker aus Cache übersprungen
    stats2 = pipeline.index_news_for_tickers(["AAPL", "MSFT"], news_limit=3, target_day=anomaly_day)
    assert stats2["tickers_skipped"] == 2, f"Erwartet 2 übersprungen, war {stats2['tickers_skipped']}"
    assert stats2["total_articles"] == 0, "Kein Netz-Fetch erwartet (bereits abgedeckt)"
    print(f"OK  Skip funktioniert: {stats2['tickers_skipped']} Ticker aus Cache, 0 neue Artikel")

    print("\n" + "=" * 64)
    print("ALLE TESTS BESTANDEN")
    print("=" * 64)

except AssertionError as e:
    print(f"\nFAIL Assertion: {e}")
    sys.exit(1)
except Exception as e:
    print(f"\nFAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    shutil.rmtree(tmp_dir, ignore_errors=True)
