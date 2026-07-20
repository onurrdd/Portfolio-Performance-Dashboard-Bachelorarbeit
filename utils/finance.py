import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


def fetch_price_at_date(ticker, date):
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


def adjust_shares_for_splits(ticker, shares, buy_date):
    stock = yf.Ticker(ticker)
    splits = stock.splits
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
    stock = yf.Ticker(ticker)
    splits = stock.splits
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
            stock = yf.Ticker(ticker)
            hist = stock.history(start=start_date, end=end_date)
            if not hist.empty:
                all_data[ticker] = hist['Close']
        except Exception:
            continue

    if not all_data:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_data).ffill().bfill()

    portfolio_values = []
    for date in df.index:
        total_value = 0
        for pos in positions:
            ticker = pos['ticker']
            buy_date = pd.to_datetime(pos['buy_date'])
            if date.tz is not None and buy_date.tz is None:
                buy_date = buy_date.tz_localize(date.tz)
            if ticker in df.columns and date >= buy_date:
                total_value += pos['shares'] * df.loc[date, ticker]
        portfolio_values.append(total_value)

    result = pd.DataFrame({'Date': df.index, 'Portfolio_Value': portfolio_values})
    result['Returns'] = result['Portfolio_Value'].pct_change()
    return result, df
