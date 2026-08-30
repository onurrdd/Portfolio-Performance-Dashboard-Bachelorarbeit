"""
Baut den FAISS-Vektorindex aus dem persistenten Cache (data/news_cache.db) neu auf.

Wird ausgeführt, wenn sich Fetching (news/sec_edgar.py) oder Chunking (rag/chunker.py)
inhaltlich ändern — der Cache bleibt die kanonische Quelle, der Index ist ein daraus
abgeleiteter Suchindex und kann jederzeit verworfen und neu gebaut werden.

Ablauf:
  1) Sicherung des bestehenden Index (data/faiss -> data/_backup_before_rebuild/).
  2) SEC-EDGAR-Zeilen aus dem Cache löschen (Fetcher hat sich geändert — alte Zeilen
     zeigen auf das primäre Formulardokument statt das Pressemitteilungs-Exhibit).
     Andere Quellen (z. B. Alpha Vantage) bleiben unangetastet.
  3) Vektorindex leeren (vectorstore.reset()).
  4) Verbleibende Cache-Artikel (nicht-SEC) mit dem AKTUELLEN Chunker (inkl.
     Boilerplate-Filter) neu chunken, embedden, indizieren — kein Netzabruf nötig,
     der Cache-Inhalt bleibt unverändert.
  5) SEC EDGAR für jede Einzeltitel-Anomalie NEU abrufen (exhibit-aware Fetcher) und
     indizieren — einziger Schritt mit Netzabruf, ausschließlich an SEC EDGAR
     (kotenlos; Alpha Vantage bleibt gesperrt, siehe RAGConfig.alpha_vantage_enabled).

Nutzung: python rebuild_index.py
"""
import os
import shutil
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

import utils.finance as fin
from auto_load import load_initial_positions
from utils.metrics import calculate_twr_metrics
from utils.anomaly import detect_anomalies
from callbacks.naive_llm import single_stock_anomalies
from rag.pipeline import RAGPipeline
from rag.config import DEFAULT_CONFIG


def _compute_single_stock_anomalies():
    """Berechnet die Einzeltitel-Anomalien des aktuell auto-geladenen Portfolios —
    derselbe Weg wie beim App-Start (auto_load -> Zeitreihe -> TWR -> Anomalieerkennung).
    ALLE Einzeltitel-Anomalien, nicht nur die im gepoolten Prompt gelisteten (siehe
    callbacks/naive_llm.py::select_prompt_anomalies) — die Auswertung (rag/evaluation.py)
    braucht Cache-Abdeckung für die volle Menge."""
    positions = load_initial_positions(fin.fetch_price_at_date)
    df, price_df = fin.calculate_portfolio_timeseries(positions)
    returns = calculate_twr_metrics(positions, price_df)['returns']
    spy_hist = fin.get_benchmark_history('SPY', df['Date'].min(), df['Date'].max())
    bench_pos = fin.build_synthetic_benchmark_positions(positions, spy_hist['Close'])
    spy_returns = calculate_twr_metrics(bench_pos, spy_hist['Close'].to_frame(name='SPY'))['returns']
    breaks = detect_anomalies(returns, spy_returns, positions, price_df)
    analysis_data = {'positions': positions, 'active_return_breaks': breaks}
    return single_stock_anomalies(analysis_data)


def main():
    print("=== 1) Sicherung des bestehenden Index ===")
    backup_dir = "./data/_backup_before_rebuild"
    if os.path.isdir(DEFAULT_CONFIG.persist_dir):
        if os.path.isdir(backup_dir):
            shutil.rmtree(backup_dir)
        shutil.copytree(DEFAULT_CONFIG.persist_dir, backup_dir)
        print(f"Gesichert: {DEFAULT_CONFIG.persist_dir} -> {backup_dir}")
    else:
        print("Kein bestehender Index gefunden — überspringe Sicherung.")

    pipeline = RAGPipeline()

    print("\n=== 2) SEC-EDGAR-Zeilen aus dem Cache löschen ===")
    deleted = pipeline.cache.delete_by_source("sec_edgar")
    print(f"{deleted} SEC-EDGAR-Zeile(n) gelöscht.")

    print("\n=== 3) Vektorindex leeren ===")
    before = pipeline.vectorstore.count_documents()
    pipeline.vectorstore.reset()
    print(f"Index geleert ({before} -> {pipeline.vectorstore.count_documents()} Chunks).")

    print("\n=== 4) Verbleibende Cache-Artikel neu chunken/embedden ===")
    remaining = pipeline.cache.all_articles_full()
    print(f"{len(remaining)} verbleibende Artikel im Cache (z. B. Alpha Vantage).")
    if remaining:
        chunks = pipeline.chunker.process_articles(remaining)
        chunks = pipeline.embedder.embed_chunks(chunks)
        pipeline.vectorstore.add_chunks(chunks)
        print(f"{len(chunks)} Chunks aus verbleibenden Artikeln indiziert "
              f"(Boilerplate-Filter kann Chunks pro Artikel reduzieren).")

    print("\n=== 5) SEC EDGAR je Einzeltitel-Anomalie neu abrufen ===")
    anomalies = _compute_single_stock_anomalies()
    print(f"{len(anomalies)} Einzeltitel-Anomalien im aktuellen Portfolio.")
    total_new, total_chunks = 0, 0
    for i, b in enumerate(anomalies, 1):
        ticker = b.get('responsible_ticker')
        date_str = b.get('date')
        if not (ticker and date_str):
            continue
        day = datetime.fromisoformat(str(date_str)[:10])
        stats = pipeline.index_news_for_tickers(
            [ticker], target_day=day,
            coverage_window_days=DEFAULT_CONFIG.anomaly_window_days,
            only_sources={"sec_edgar"})
        total_new += stats["total_articles"]
        total_chunks += stats["total_chunks"]
        print(f"  [{i}/{len(anomalies)}] {date_str} {ticker}: "
              f"+{stats['total_articles']} Artikel, +{stats['total_chunks']} Chunks")

    print("\n=== Zusammenfassung ===")
    print(f"SEC-EDGAR-Zeilen gelöscht:      {deleted}")
    print(f"Neue SEC-EDGAR-Artikel:         {total_new}")
    print(f"Neue SEC-EDGAR-Chunks:          {total_chunks}")
    print(f"Cache gesamt:                   {pipeline.cache.count()} Artikel")
    print(f"Vektorindex gesamt:             {pipeline.vectorstore.count_documents()} Chunks")


if __name__ == "__main__":
    main()
