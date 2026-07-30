"""
Temporary auto-load: imports portfolio_5_ticker.csv on app startup.
"""
import os
import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "portfolio_5_ticker_older.csv")


def load_initial_positions(fetch_price_fn):
    if not os.path.exists(CSV_PATH):
        return []

    try:
        df = pd.read_csv(CSV_PATH)
        positions = []
        for _, row in df.iterrows():
            ticker = str(row['ticker']).upper()
            shares = int(row['shares'])
            buy_date = str(row['buy_date'])

            if 'buy_price' in df.columns and pd.notna(row.get('buy_price')):
                buy_price = float(row['buy_price'])
            else:
                buy_price = fetch_price_fn(ticker, buy_date)
                if buy_price is None:
                    continue

            positions.append({
                'ticker': ticker,
                'shares': shares,
                'buy_date': buy_date,
                'buy_price': buy_price
            })

        return positions

    except Exception as e:
        print(f"Auto-load error: {e}")
        return []
