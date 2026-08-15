import os
import time
import sqlite3
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# Einfacher In-Memory-TTL-Cache für yfinance-Aufrufe. Lebt für die Dauer des
# laufenden Prozesses (python dashboard.py) — ein Browser-Refresh (F5) muss
# Marktdaten nicht erneut vom Netz holen. Die TTL verhindert dauerhaft
# veraltete Kurse während einer langen Sitzung.
_CACHE_TTL_SECONDS = 300
_cache = {}


def _cached(key, fetch_fn):
    now = time.time()
    hit = _cache.get(key)
    if hit is not None and now - hit[0] < _CACHE_TTL_SECONDS:
        return hit[1]
    value = fetch_fn()
    _cache[key] = (now, value)
    return value


# Persistenter Kurs-Cache (SQLite) — historische Tagesschlusskurse überleben einen
# Neustart. Ergänzt den In-Memory-TTL-Cache oben: der TTL-Cache verhindert wiederholte
# Netz-Aufrufe innerhalb einer laufenden Sitzung, dieser Cache verhindert sie über
# Neustarts/Portfolio-Wechsel hinweg. Vergangene Tage sind unveränderlich, daher ist
# Wiederverwendung unbegrenzt sicher; nur die Zeitspanne, die noch nicht im Cache liegt,
# wird nachgeholt (gleiches Abdeckungs-Prinzip wie rag/cache.py, dort für Nachrichten).
_PRICE_DB_PATH = "./data/price_cache.db"


def _price_db():
    os.makedirs(os.path.dirname(_PRICE_DB_PATH) or ".", exist_ok=True)
    con = sqlite3.connect(_PRICE_DB_PATH)
    con.execute("CREATE TABLE IF NOT EXISTS prices "
                "(ticker TEXT, date TEXT, close REAL, PRIMARY KEY (ticker, date))")
    con.execute("CREATE TABLE IF NOT EXISTS coverage "
                "(ticker TEXT PRIMARY KEY, min_date TEXT, max_date TEXT)")
    return con


def _fetch_and_store(con, ticker, start, end):
    hist = yf.Ticker(ticker).history(start=start, end=end)
    if hist.empty:
        return
    rows = [(ticker, d.strftime('%Y-%m-%d'), float(c)) for d, c in hist['Close'].items()]
    con.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?)", rows)


def get_price_history_cached(ticker, start, end):
    """Historische Tagesschlusskurse für [start,end], persistent gecacht (SQLite).

    Holt bei wiederholten Aufrufen nur die Zeitspanne nach, die noch nicht im Cache
    liegt — nicht die gesamte Historie erneut. Das heutige Datum wird bewusst NICHT
    als abgedeckt vermerkt (Kurs des laufenden Handelstags ist noch nicht endgültig),
    dadurch bleibt jede Sitzung mit einem kleinen, günstigen Nachhol-Fetch aktuell.
    """
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    yesterday = pd.Timestamp.now().normalize() - timedelta(days=1)

    con = _price_db()
    try:
        row = con.execute("SELECT min_date, max_date FROM coverage WHERE ticker=?",
                          (ticker,)).fetchone()
        if row is None:
            _fetch_and_store(con, ticker, start, end)
            persist_max = min(end, yesterday)
            if persist_max >= start:
                con.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?)",
                            (ticker, start.strftime('%Y-%m-%d'), persist_max.strftime('%Y-%m-%d')))
        else:
            cov_min, cov_max = pd.to_datetime(row[0]), pd.to_datetime(row[1])
            if start < cov_min:
                _fetch_and_store(con, ticker, start, cov_min)
            if end > cov_max:
                _fetch_and_store(con, ticker, cov_max, end)
            new_min = min(start, cov_min)
            new_max = max(cov_max, min(end, yesterday))
            if new_min != cov_min or new_max != cov_max:
                con.execute("INSERT OR REPLACE INTO coverage VALUES (?,?,?)",
                            (ticker, new_min.strftime('%Y-%m-%d'), new_max.strftime('%Y-%m-%d')))
        con.commit()
        df = pd.read_sql(
            "SELECT date, close FROM prices WHERE ticker=? AND date BETWEEN ? AND ? ORDER BY date",
            con, params=(ticker, start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')))
    finally:
        con.close()

    if df.empty:
        return pd.DataFrame(columns=['Close'])
    df['date'] = pd.to_datetime(df['date'])
    return df.set_index('date').rename(columns={'close': 'Close'})


def get_current_price(ticker):
    """Aktueller Schlusskurs (period='1d'), gecacht."""
    return _cached(f"current_price:{ticker}",
                  lambda: yf.Ticker(ticker).history(period="1d")['Close'].iloc[-1])


def get_benchmark_history(ticker, start, end):
    """Historische Kursreihe für einen Benchmark-Ticker (z. B. SPY), gecacht."""
    key = f"benchmark_hist:{ticker}:{pd.to_datetime(start).date()}:{pd.to_datetime(end).date()}"
    return _cached(key, lambda: get_price_history_cached(ticker, start, end))


def _get_splits(ticker):
    """Split-Historie eines Tickers, gecacht — von beiden adjust_*-Funktionen
    gemeinsam genutzt (vorher wurde sie pro Position zweimal einzeln geholt)."""
    return _cached(f"splits:{ticker}", lambda: yf.Ticker(ticker).splits)


def fetch_price_at_date(ticker, date):
    def _fetch():
        try:
            stock = yf.Ticker(ticker)
            start = pd.to_datetime(date) - timedelta(days=5)
            end = pd.to_datetime(date) + timedelta(days=5)
            hist = stock.history(start=start, end=end, auto_adjust=False)
            if not hist.empty:
                return hist['Close'].iloc[0]
            return None
        except Exception:
            return None
    return _cached(f"price_at_date:{ticker}:{date}", _fetch)


def adjust_shares_for_splits(ticker, shares, buy_date):
    splits = _get_splits(ticker)
    if splits.empty:
        return shares

    buy_date = pd.to_datetime(buy_date)
    if buy_date.tz is None and splits.index.tz is not None:
        buy_date = buy_date.tz_localize(splits.index.tz)

    adjusted_shares = shares
    for split_ratio in splits[splits.index > buy_date]:
        adjusted_shares *= split_ratio
    return int(adjusted_shares)


def adjust_price_for_splits(ticker, price, date):
    splits = _get_splits(ticker)
    if splits.empty:
        return price

    date = pd.to_datetime(date)
    if date.tz is None and splits.index.tz is not None:
        date = date.tz_localize(splits.index.tz)

    adjusted_price = price
    for split_ratio in splits[splits.index > date]:
        adjusted_price *= split_ratio
    return adjusted_price


def build_pme_positions(positions, benchmark_prices, benchmark_ticker='SPY'):
    """Public Market Equivalent (PME): simuliert für jede Position einen Kauf zum
    selben buy_date im Benchmark-Index, mit demselben investierten Dollarbetrag,
    damit die Kapitalzufluss-Zeitpunkte des Portfolios und des Benchmarks übereinstimmen."""
    pme_positions = []
    for pos in positions:
        buy_date = pd.to_datetime(pos['buy_date'])
        if benchmark_prices.index.tz is not None and buy_date.tz is None:
            buy_date = buy_date.tz_localize(benchmark_prices.index.tz)

        price_at_buy = benchmark_prices.asof(buy_date)
        if pd.isna(price_at_buy):
            continue

        invested_amount = pos['shares'] * pos['buy_price']
        pme_positions.append({
            'ticker': benchmark_ticker,
            'shares': invested_amount / price_at_buy,
            'buy_date': pos['buy_date']
        })
    return pme_positions


def calculate_portfolio_timeseries(positions):
    if not positions:
        return pd.DataFrame(), pd.DataFrame()

    dates = [pd.to_datetime(pos['buy_date']) for pos in positions]
    start_date = min(dates)
    end_date = datetime.now()

    all_data = {}
    for pos in positions:
        ticker = pos['ticker']
        try:
            # Cache-Schlüssel auf Tagesgranularität von start_date (end_date="jetzt"
            # wird durch die TTL selbst aktuell gehalten, nicht durch den Schlüssel).
            key = f"timeseries:{ticker}:{start_date.date()}"
            hist = _cached(key, lambda t=ticker, s=start_date, e=end_date: get_price_history_cached(t, s, e))
            if not hist.empty:
                all_data[ticker] = hist['Close']
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_data).ffill().bfill()

    # Split-bereinigte Stückzahl je Position (einmalig, nicht pro Tag): df-Spalten sind
    # split-adjustierte Kurse (yfinance-Standard, history() gibt splitbereinigte Closes
    # zurück), rohe pos['shares'] wäre damit inkonsistent (Maßstabsfehler um den
    # Split-Faktor) — analog zur Positions-Tabelle in callbacks/charts.py.
    positions_adj = [
        (pos, adjust_shares_for_splits(pos['ticker'], pos['shares'], pos['buy_date']))
        for pos in positions
    ]

    portfolio_values = []
    for date in df.index:
        total_value = 0
        for pos, shares in positions_adj:
            ticker = pos['ticker']
            buy_date = pd.to_datetime(pos['buy_date'])
            if date.tz is not None and buy_date.tz is None:
                buy_date = buy_date.tz_localize(date.tz)
            if ticker in df.columns and date >= buy_date:
                total_value += shares * df.loc[date, ticker]
        portfolio_values.append(total_value)

    result = pd.DataFrame({'Date': df.index, 'Portfolio_Value': portfolio_values})
    result['Returns'] = result['Portfolio_Value'].pct_change()
    return result, df
