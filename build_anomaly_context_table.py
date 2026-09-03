"""
Baut einen LLM-freien Snapshot des Retrieval-Ergebnisses für JEDE Einzeltitel-Anomalie
des aktuellen Portfolios: (Datum, Ticker, gecachte Quellen im Fenster, tatsaechlich an
den Prompt gehender Kontext-Block).

Zweck: Die Forschungsfrage "Erklaert die abgerufene Quelle die Kursbewegung ueberhaupt?"
laesst sich allein am Retrieval beurteilen, ohne einen LLM-Aufruf. Retrieval ist lokal
und kostenlos (FAISS + MiniLM + SQLite + SEC EDGAR); kein LLM-Kontingent wird
angefasst. Der Sparmodus (rag_config.SAVING_MODE) wird hier zur Laufzeit
abgeschaltet — es laufen ALLE Einzeltitel-Anomalien des Portfolios durch, nicht nur
die im UI-Prompt gelistete Auswahl, und Alpha Vantage bleibt damit gesperrt.

Ablauf je Anomalie:
  1) SEC-EDGAR-Abdeckung sicherstellen (identischer Aufruf wie rebuild_index.py Schritt 5;
     Cache-gesteuert, also beim Wiederholen kein erneuter Netzabruf). Alpha Vantage bleibt
     ueber RAGConfig.alpha_vantage_enabled gesperrt.
  2) Cache-Inhalt im Fenster [Datum - anomaly_window_days, Datum + anomaly_window_days_after]
     lesen (dieselbe Abfrage wie die "Kaynak"-Spalte der Anomalietage-Tabelle).
  3) Produktions-Retrieval (RAGPipeline.retrieve_for_anomaly, gleiche Query/top_k wie
     rag/evaluation.py) + format_context_for_llm -> der Text, der tatsaechlich an den
     Prompt gehen wuerde.

Ausgabe:
  - data/anomaly_context.db (SQLite, Tabelle anomaly_context) — bei jedem Lauf neu
    aufgebaut (Snapshot, kein inkrementelles Anhaengen).
  - data/anomaly_context_report.md — lesbarer Bericht fuer die manuelle Bewertung.

Nutzung: python build_anomaly_context_table.py
"""
import os
import json
import sqlite3
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from rag import config as rag_config
# Vollmodus erzwingen, BEVOR die Pipeline (und damit AlphaVantageNewsSource) gebaut
# wird: dieser Snapshot läuft über ALLE Anomalien; mit aktivem Alpha Vantage würde
# das Tageskontingent (25/Tag) sofort aufgebraucht.
rag_config.SAVING_MODE = False

from rebuild_index import _compute_single_stock_anomalies
from rag.pipeline import RAGPipeline
from rag.config import DEFAULT_CONFIG
from rag.evaluation import RETRIEVAL_QUERY, TOP_K

DB_PATH = "./data/anomaly_context.db"
REPORT_PATH = "./data/anomaly_context_report.md"

_SCHEMA = """
DROP TABLE IF EXISTS anomaly_context;
CREATE TABLE anomaly_context (
    date            TEXT,
    ticker          TEXT,
    own_return_pct  REAL,
    surprise_mad_z  REAL,
    n_cached        INTEGER,
    cached_articles TEXT,
    n_retrieved     INTEGER,
    prompt_context  TEXT,
    created_at      TEXT
);
"""


def _cached_articles_json(pipeline, ticker, day):
    """Cache-Inhalt im Anomalie-Fenster — identische Abfrage wie die 'Kaynak'-Spalte
    der Anomalietage-Tabelle (callbacks/rag.py::_anomaly_source_modal_body)."""
    articles = pipeline.cache.get_articles(
        ticker, day, DEFAULT_CONFIG.anomaly_window_days,
        window_days_after=DEFAULT_CONFIG.anomaly_window_days_after,
    )
    slim = [
        {
            "title": a.get("title") or "",
            "source": a.get("source") or "",
            "published": a.get("published") or "",
            "link": a.get("link") or "",
            "summary": a.get("summary") or "",
        }
        for a in articles
    ]
    return articles, json.dumps(slim, ensure_ascii=False)


def main():
    print("=== 1) Einzeltitel-Anomalien headless berechnen (ALLE, Sparmodus aus) ===")
    anomalies = _compute_single_stock_anomalies()
    print(f"{len(anomalies)} Einzeltitel-Anomalien im aktuellen Portfolio.")

    pipeline = RAGPipeline()

    os.makedirs("./data", exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.executescript(_SCHEMA)

    now = datetime.now().isoformat()
    report_sections = []
    n_no_cache = 0
    n_no_prompt_context = 0

    for i, b in enumerate(anomalies, 1):
        ticker = b.get("responsible_ticker")
        date_str = b.get("date")
        if not (ticker and date_str):
            continue
        try:
            day = datetime.fromisoformat(str(date_str)[:10])
        except ValueError:
            continue

        print(f"[{i}/{len(anomalies)}] {date_str} {ticker} ...")

        # 1) SEC-EDGAR-Abdeckung sicherstellen (Cache-gesteuert, kein AV-Abruf).
        pipeline.index_news_for_tickers(
            [ticker], target_day=day,
            coverage_window_days=DEFAULT_CONFIG.anomaly_window_days,
            only_sources={"sec_edgar"},
        )

        # 2) Cache-Inhalt im Fenster.
        cached_articles, cached_json = _cached_articles_json(pipeline, ticker, day)
        if not cached_articles:
            n_no_cache += 1

        # 3) Produktions-Retrieval + Formatierung — der Text, der tatsaechlich an
        #    den Prompt gehen wuerde (identisch zu rag/evaluation.py::generate_answer).
        chunks = pipeline.retrieve_for_anomaly(RETRIEVAL_QUERY, b, top_k=TOP_K)
        prompt_context = pipeline.format_context_for_llm(chunks, max_tokens=2000)
        if not prompt_context:
            n_no_prompt_context += 1

        own_return = b.get("ticker_own_return_pct", 0) or 0
        con.execute(
            "INSERT INTO anomaly_context (date, ticker, own_return_pct, surprise_mad_z, "
            "n_cached, cached_articles, n_retrieved, prompt_context, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(date_str), ticker, own_return, b.get("surprise_mad_z"),
             len(cached_articles), cached_json, len(chunks), prompt_context, now),
        )

        direction = "rose" if own_return >= 0 else "fell"
        lines = [
            f"## {date_str} — {ticker} {direction} {own_return:+.2f}%",
            "",
            f"Gecachte Quellen im Fenster: {len(cached_articles)}",
        ]
        for a in cached_articles:
            lines.append(
                f"- [{a.get('source', '')}] {a.get('published', '')[:10]} — "
                f"{a.get('title', '')} ({a.get('link', '')})"
            )
        lines.append("")
        lines.append(f"Retrievte Chunks (top_k={TOP_K}): {len(chunks)}")
        lines.append("")
        lines.append("### Prompt-Kontext (tatsaechlich an das LLM gehender Text)")
        lines.append("")
        lines.append(prompt_context if prompt_context else "*(leer — kein Kontext im Fenster gefunden)*")
        lines.append("")
        report_sections.append("\n".join(lines))

    con.commit()
    con.close()

    total = len(report_sections)
    summary = (
        f"# Anomaly-Context-Report\n\n"
        f"Erstellt: {now}\n\n"
        f"{total} Einzeltitel-Anomalien verarbeitet. "
        f"{n_no_cache} ohne gecachte Quelle im Fenster. "
        f"{n_no_prompt_context} ohne Prompt-Kontext (Retrieval leer oder außerhalb Ticker/Fenster-Filter).\n\n"
        "---\n\n"
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(summary + "\n\n---\n\n".join(report_sections))

    print("\n=== Zusammenfassung ===")
    print(f"Anomalien verarbeitet:         {total}")
    print(f"Ohne gecachte Quelle:          {n_no_cache}")
    print(f"Ohne Prompt-Kontext:           {n_no_prompt_context}")
    print(f"SQLite:                        {DB_PATH}")
    print(f"Report:                        {REPORT_PATH}")


if __name__ == "__main__":
    main()
