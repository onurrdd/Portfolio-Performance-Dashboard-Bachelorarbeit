"""Baut ein breiter gestreutes Ersatzportfolio und vergleicht seine Anomalien mit
denen des aktuellen Portfolios.

Hintergrund: Im aktuellen Portfolio gehen ALLE Einzeltitel-Anomalien auf zwei
Titel zurueck (TSLA, GME), weil deren Gewicht die Portfoliorendite dominiert.
Die uebrigen Positionen erreichen die Anomalieschwelle nie. Fuer die Evaluation
ist das eine Einschraenkung der externen Validitaet: die Befunde stuetzen sich
faktisch auf zwei Unternehmen.

Dieses Skript entwirft ein Ersatzportfolio mit zehn Titeln zu annaehernd
gleichen Anfangsgewichten (8-12 %), das die sechs bisherigen Titel enthaelt und
um vier Titel aus anderen Sektoren ergaenzt. Anschliessend berechnet es fuer
BEIDE Portfolios die Einzeltitel-Anomalien und stellt sie gegenueber.

Es werden WEDER LLM- NOCH Nachrichten-API-Aufrufe ausgeloest: gerechnet wird nur
mit Kursdaten (yfinance) und dem lokalen Nachrichten-Cache, der lediglich
gelesen wird, um die vorhandene Quellenabdeckung auszuweisen.

Nutzung: python build_backup_portfolio.py
"""
import random
from datetime import datetime, timedelta

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

import utils.finance as fin
from auto_load import load_initial_positions
from utils.metrics import calculate_twr_metrics
from utils.anomaly import detect_anomalies
from callbacks.naive_llm import single_stock_anomalies
from rag.cache import NewsCache
from rag.config import DEFAULT_CONFIG

CSV_OUT = "portfolio_10_ticker_backup.csv"
REPORT_OUT = "data/backup_portfolio_report.md"

# Gemeinsames Kaufdatum fuer alle Positionen. Frueh genug gewaehlt, damit der
# Beobachtungszeitraum die markanten Ereignisfenster mehrerer Titel umfasst, und
# identisch fuer alle Positionen, damit die Anfangsgewichte exakt der Vorgabe
# entsprechen (bei gestaffelten Kaeufen waeren sie nur naeherungsweise zu treffen).
BUY_DATE = "2020-08-03"
CAPITAL = 200_000.0

# Die sechs Titel des aktuellen Portfolios bleiben enthalten, damit die
# Anomalien beider Portfolios ueberhaupt vergleichbar sind. Ergaenzt werden vier
# Titel aus bewusst anderen Sektoren (Finanzen, Energie, Gesundheit, Industrie),
# um die Konzentration auf Technologie/Automobil aufzubrechen.
TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "TSLA", "GME",  # bisher
    "JPM",   # Finanzen
    "XOM",   # Energie
    "PFE",   # Gesundheit
    "BA",    # Industrie / Luftfahrt
]

WEIGHT_MIN, WEIGHT_MAX = 0.08, 0.12
RANDOM_SEED = 20260829  # feste Saat -> reproduzierbare Gewichte


def _draw_weights(n, seed=RANDOM_SEED):
    """Zieht n Gewichte im Band [WEIGHT_MIN, WEIGHT_MAX], die sich zu 1 summieren.

    Gezogen wird gleichverteilt im Band und anschliessend normiert; die
    Normierung kann einzelne Werte minimal aus dem Band schieben, deshalb wird
    bis zur Einhaltung neu gezogen (bei zehn Titeln und diesem Band konvergiert
    das sofort, da der Bandmittelpunkt genau 1/n entspricht).
    """
    rng = random.Random(seed)
    for _ in range(1000):
        raw = [rng.uniform(WEIGHT_MIN, WEIGHT_MAX) for _ in range(n)]
        total = sum(raw)
        w = [x / total for x in raw]
        if all(WEIGHT_MIN <= x <= WEIGHT_MAX for x in w):
            return w
    raise RuntimeError("Keine gueltige Gewichtsziehung gefunden")


def _build_positions():
    """Erzeugt die Positionsliste des Ersatzportfolios und schreibt sie als CSV."""
    weights = _draw_weights(len(TICKERS))
    rows, positions = [], []
    for ticker, w in zip(TICKERS, weights):
        price = fin.fetch_price_at_date(ticker, BUY_DATE)
        if price is None or price <= 0:
            print(f"  WARNUNG: kein Kurs fuer {ticker} am {BUY_DATE} — uebersprungen")
            continue
        # Die Kursreihe ist splitbereinigt: `price` liegt bereits auf der HEUTIGEN
        # Stueckzahlskala. Die Positionsliste erwartet dagegen die damals
        # tatsaechlich gekaufte Stueckzahl, die spaeter ueber
        # adjust_shares_for_splits wieder auf die heutige Skala hochgerechnet
        # wird. Ohne die Division unten wuerde der Splitfaktor doppelt wirken und
        # Titel mit Splits (GME, NVDA, TSLA, AAPL) massiv uebergewichtet.
        effective = w * CAPITAL / float(price)      # Stueck auf heutiger Skala
        factor = fin.adjust_shares_for_splits(ticker, 1_000_000, BUY_DATE) / 1_000_000
        shares = int(round(effective / factor))     # Stueck auf Kaufzeitpunkt-Skala
        if shares <= 0:
            print(f"  WARNUNG: Stueckzahl 0 fuer {ticker} — uebersprungen")
            continue
        rows.append({"ticker": ticker, "shares": shares, "buy_date": BUY_DATE})
        positions.append({"ticker": ticker, "shares": shares,
                          "buy_date": BUY_DATE, "buy_price": float(price) * factor})
        print(f"  {ticker:6s} Zielgewicht {w*100:5.2f}%  Kurs(bereinigt) {float(price):9.2f}  "
              f"Splitfaktor {factor:5.1f}  Stueck {shares:7d}")
    pd.DataFrame(rows).to_csv(CSV_OUT, index=False)
    return positions, weights


def _anomalies_for(positions):
    """Einzeltitel-Anomalien fuer eine Positionsliste — derselbe Weg wie beim
    App-Start (Zeitreihe -> TWR -> angeglichene Benchmark -> Erkennung)."""
    df, price_df = fin.calculate_portfolio_timeseries(positions)
    returns = calculate_twr_metrics(positions, price_df)["returns"]
    spy_hist = fin.get_benchmark_history("SPY", df["Date"].min(), df["Date"].max())
    bench_pos = fin.build_synthetic_benchmark_positions(positions, spy_hist["Close"])
    spy_returns = calculate_twr_metrics(
        bench_pos, spy_hist["Close"].to_frame(name="SPY"))["returns"]
    breaks = detect_anomalies(returns, spy_returns, positions, price_df)
    return single_stock_anomalies({"active_return_breaks": breaks}), price_df


def _source_count(cache, ticker, date_str):
    """Anzahl bereits gecachter Quellen im Anomaliefenster (nur Lesezugriff)."""
    try:
        day = datetime.fromisoformat(str(date_str)[:10])
    except (ValueError, TypeError):
        return 0
    return len(cache.get_articles(
        ticker, day, DEFAULT_CONFIG.anomaly_window_days,
        window_days_after=DEFAULT_CONFIG.anomaly_window_days_after))


def _key(b):
    return (str(b.get("date"))[:10], b.get("responsible_ticker"))


def _by_ticker(anomalies):
    out = {}
    for b in anomalies:
        out[b["responsible_ticker"]] = out.get(b["responsible_ticker"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main():
    print("=== 1) Ersatzportfolio aufbauen ===")
    backup_positions, weights = _build_positions()
    print(f"CSV geschrieben: {CSV_OUT}")

    print("\n=== 2) Anomalien des Ersatzportfolios ===")
    backup_anomalies, backup_prices = _anomalies_for(backup_positions)
    print(f"{len(backup_anomalies)} Einzeltitel-Anomalie(n)")

    print("\n=== 3) Anomalien des aktuellen Portfolios ===")
    current_positions = load_initial_positions(fin.fetch_price_at_date)
    current_anomalies, _ = _anomalies_for(current_positions)
    print(f"{len(current_anomalies)} Einzeltitel-Anomalie(n)")

    print("\n=== 4) Bericht schreiben ===")
    cache = NewsCache(DEFAULT_CONFIG.db_path)
    backup_keys = {_key(b) for b in backup_anomalies}
    current_keys = {_key(b) for b in current_anomalies}
    shared = backup_keys & current_keys
    only_backup = backup_keys - current_keys
    only_current = current_keys - backup_keys

    backup_by_ticker = _by_ticker(backup_anomalies)
    current_by_ticker = _by_ticker(current_anomalies)

    lines = []
    a = lines.append
    a("# Ersatzportfolio — Anomalievergleich")
    a("")
    a(f"Erstellt: {datetime.now().isoformat(timespec='seconds')}  ")
    a(f"Kaufdatum aller Positionen: {BUY_DATE} · Anlagesumme: {CAPITAL:,.0f}")
    a("")
    a("Es wurden keine LLM- oder Nachrichten-API-Aufrufe ausgeloest. Die Spalte "
      "„Quellen im Fenster\" liest ausschliesslich den vorhandenen lokalen Cache.")
    a("")

    a("## Zusammensetzung des Ersatzportfolios")
    a("")
    a("| Titel | Zielgewicht | Kurs am Kaufdatum | Stueck | im aktuellen Portfolio |")
    a("|---|---:|---:|---:|---|")
    for p, w in zip(backup_positions, weights):
        in_cur = "ja" if p["ticker"] in {c["ticker"] for c in current_positions} else "nein (neu)"
        a(f"| {p['ticker']} | {w*100:.2f}% | {p['buy_price']:.2f} | {p['shares']} | {in_cur} |")
    a("")

    a("## Anomalien je Titel")
    a("")
    a("| Titel | Ersatzportfolio | Aktuelles Portfolio |")
    a("|---|---:|---:|")
    for t in sorted(set(backup_by_ticker) | set(current_by_ticker)):
        a(f"| {t} | {backup_by_ticker.get(t, 0)} | {current_by_ticker.get(t, 0)} |")
    a(f"| **Summe** | **{len(backup_anomalies)}** | **{len(current_anomalies)}** |")
    a("")
    a(f"Auslesende Titel: Ersatzportfolio **{len(backup_by_ticker)}**, "
      f"aktuelles Portfolio **{len(current_by_ticker)}**.")
    a("")

    a("## Ueberschneidung")
    a("")
    a(f"- Gemeinsame (Datum, Titel)-Paare: **{len(shared)}**")
    a(f"- Nur im Ersatzportfolio: **{len(only_backup)}**")
    a(f"- Nur im aktuellen Portfolio: **{len(only_current)}**")
    if backup_keys or current_keys:
        union = len(backup_keys | current_keys)
        a(f"- Jaccard-Aehnlichkeit (Schnitt/Vereinigung): **{len(shared)/union:.2f}**")
    a("")

    a("## Anomalien des Ersatzportfolios im Detail")
    a("")
    a("| # | Datum | Titel | Titel-Rendite | Titel (MAD-z) | Quellen im Fenster | auch im aktuellen Portfolio |")
    a("|---:|---|---|---:|---:|---:|---|")
    for i, b in enumerate(sorted(backup_anomalies, key=lambda x: str(x["date"])), 1):
        t = b["responsible_ticker"]
        d = str(b["date"])[:10]
        ret = b.get("ticker_own_return_pct") or 0.0
        mz = b.get("ticker_own_mad_z")
        mz_s = f"{mz:.2f}" if isinstance(mz, (int, float)) else "—"
        n_src = _source_count(cache, t, d)
        a(f"| {i} | {d} | {t} | {ret:+.2f}% | {mz_s} | {n_src} | "
          f"{'ja' if (d, t) in current_keys else 'nein'} |")
    a("")

    a("## Anomalien nur im aktuellen Portfolio")
    a("")
    if only_current:
        a("| Datum | Titel |")
        a("|---|---|")
        for d, t in sorted(only_current):
            a(f"| {d} | {t} |")
    else:
        a("(keine)")
    a("")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Bericht geschrieben: {REPORT_OUT}")

    print("\n=== Zusammenfassung ===")
    print(f"Ersatzportfolio:      {len(backup_anomalies)} Anomalien, "
          f"{len(backup_by_ticker)} ausloesende Titel {list(backup_by_ticker)}")
    print(f"Aktuelles Portfolio:  {len(current_anomalies)} Anomalien, "
          f"{len(current_by_ticker)} ausloesende Titel {list(current_by_ticker)}")
    print(f"Gemeinsam: {len(shared)} · nur Ersatz: {len(only_backup)} · "
          f"nur aktuell: {len(only_current)}")


if __name__ == "__main__":
    main()
