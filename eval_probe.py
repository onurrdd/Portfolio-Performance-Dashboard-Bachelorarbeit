"""Vollstaendiger Evaluationslauf (RAGAS + Naive-vs-RAG-Vergleich) auf der
KLEINEN, fest gewaehlten Anomalie-Stichprobe aus rag/config.py.

Zweck: Pruefen, ob die Evaluationsarchitektur (rag/evaluation.py) end-to-end
traegt — Kontextabruf, Antworterzeugung in beiden Bedingungen, Hakem-
Klassifikation, RAGAS-Metriken, Aggregation und Persistenz — ohne das
Tageskontingent des LLM-Anbieters mit allen Anomalien des Portfolios zu
verbrauchen. Ein vollstaendiger Lauf ueber alle Einzeltitel-Anomalien wuerde je
Anomalie rund ein Dutzend LLM-Aufrufe ausloesen; diese Stichprobe kostet einen
Bruchteil davon.

Die bearbeiteten Faelle stehen in rag/config.py (PROBE_POOL/PROBE_PICK) — EINE
Quelle fuer Dashboard-Sparmodus, probe.py und diesen Lauf. Die volle Auswahl
[1,2,3,4] deckt beide Seiten des Wissensschnitts und beide Quellentypen ab.

rag_config.SAVING_MODE wird fuer diesen Lauf zur Laufzeit abgeschaltet (die Datei
bleibt unveraendert): der Sparmodus waehlt sonst selbst und wuerde die hier
gesetzte Stichprobe (PROBE_KEYS) ueberschreiben. Alpha Vantage bleibt damit
ebenfalls gesperrt — die Stichprobe zieht ihren Kontext aus dem Cache.

Nutzung: python eval_probe.py
"""
import json
import os

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from rag import config as rag_config
# Sparmodus-Auswahl ausschalten, BEVOR die Pipeline (und damit AlphaVantageNewsSource)
# gebaut wird und BEVOR die Evaluation Prompts baut (siehe Docstring).
rag_config.SAVING_MODE = False

import utils.finance as fin
from auto_load import load_initial_positions
from utils.metrics import calculate_twr_metrics
from utils.anomaly import detect_anomalies
from rag.pipeline import RAGPipeline
from rag.evaluation import run_ragas_evaluation, save_eval_results
from callbacks.naive_llm import llm_api_key, LLM_PROVIDER

# Die bearbeitete Stichprobe kommt aus rag/config.py (PROBE_POOL/PROBE_PICK) —
# EINE Quelle für Dashboard-Sparmodus, probe.py und diesen Lauf. Auswahl ändern:
# dort PROBE_PICK anpassen, nicht hier.
PROBE_KEYS = {rag_config.PROBE_POOL[i - 1] for i in rag_config.PROBE_PICK}


def _build_analysis_data():
    """Baut analysis_data wie beim App-Start, beschraenkt die Anomalieliste
    danach auf die Stichprobe (dieselbe Struktur, die callbacks/charts.py
    erzeugt — die Evaluation liest daraus Kennzahlen und Anomalien)."""
    positions = load_initial_positions(fin.fetch_price_at_date)
    df, price_df = fin.calculate_portfolio_timeseries(positions)
    twr = calculate_twr_metrics(positions, price_df)
    returns = twr["returns"]

    spy_hist = fin.get_benchmark_history("SPY", df["Date"].min(), df["Date"].max())
    bench_pos = fin.build_synthetic_benchmark_positions(positions, spy_hist["Close"])
    spy_twr = calculate_twr_metrics(bench_pos, spy_hist["Close"].to_frame(name="SPY"))
    spy_returns = spy_twr["returns"]

    breaks = detect_anomalies(returns, spy_returns, positions, price_df)
    selected = [
        b for b in breaks
        if (str(b.get("date"))[:10], b.get("responsible_ticker")) in PROBE_KEYS
        and b.get("concentration") == "Hisseye özgü"
    ]

    def _num(v):
        try:
            f = float(v)
            return 0.0 if np.isnan(f) else f
        except (TypeError, ValueError):
            return 0.0

    def _last_sharpe(metrics):
        """calculate_twr_metrics liefert eine rollierende Sharpe-REIHE; der Prompt
        erwartet den aktuellen Punktwert (wie callbacks/charts.py: .iloc[-1])."""
        series = metrics.get("rolling_sharpe")
        if series is None or len(series) == 0:
            return 0.0
        return _num(series.iloc[-1])

    return {
        "positions": positions,
        "metrics": {
            "total_return": _num(twr.get("total_return")),
            "sortino": _num(twr.get("sortino")),
        },
        "rolling_sharpe": {"current": _last_sharpe(twr)},
        "benchmark": {
            "total_return": _num(spy_twr.get("total_return")),
            "sortino": _num(spy_twr.get("sortino")),
            "sharpe_current": _last_sharpe(spy_twr),
        },
        "active_return_breaks": selected,
    }


def main():
    print(f"LLM_PROVIDER={LLM_PROVIDER}")
    api_key = llm_api_key()
    if not api_key:
        raise SystemExit(f"Kein API-Schlüssel für Anbieter '{LLM_PROVIDER}' gesetzt.")

    print("=== 1) Stichprobe aufbauen ===")
    analysis_data = _build_analysis_data()
    breaks = analysis_data["active_return_breaks"]
    print(f"{len(breaks)} Anomalie(n) in der Stichprobe:")
    for b in breaks:
        print(f"  {b['date']} {b['responsible_ticker']} "
              f"{b.get('ticker_own_return_pct', 0):+.2f}%")
    if not breaks:
        raise SystemExit("Stichprobe leer — PROBE_KEYS passen nicht zu den erkannten Anomalien.")

    print("\n=== 2) Evaluation ausfuehren (LLM-Aufrufe) ===")
    pipeline = RAGPipeline()
    result = run_ragas_evaluation(pipeline, analysis_data, api_key)
    path = save_eval_results(result)

    print("\n=== 3) Ergebnis je Anomalie ===")
    for s in result["samples"]:
        print(f"\n--- {s['date']} {s['ticker']} ({s['status']}, "
              f"{s['n_contexts']} Kontext-Chunk(s)) ---")
        print(f"  Faithfulness      : {s['faithfulness']}")
        print(f"  Answer Relevancy  : {s['answer_relevancy']}")
        print(f"  Context Precision : {s['context_precision']}")
        print(f"  Spezifitaet  RAG/Naive : {s['specificity_rag']} / {s['specificity_naive']}")
        print(f"  Zitierbarkeit RAG/Naive: {s['citation_rag']} / {s['citation_naive']}")

    print("\n=== 4) Aggregat ===")
    for k, v in result["aggregate"].items():
        print(f"  {k:20s}: {v}")

    print("\n=== 5) Vergleich (Spezifitaet / Zitierbarkeit) ===")
    print(json.dumps(result["comparison"], indent=2, ensure_ascii=False))

    print(f"\n{result['n_skipped']} Anomalie(n) ohne Kontext. Ergebnis: {path}")


if __name__ == "__main__":
    main()
