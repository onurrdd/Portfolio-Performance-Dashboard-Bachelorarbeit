"""
Temporary auto-load: imports portfolio_5_ticker.csv on app startup.
"""
import os
import pandas as pd
import dash
from dash import Input, Output

CSV_PATH = os.path.join(os.path.dirname(__file__), "portfolio_5_ticker.csv")


def register(app, fetch_price_fn):
    @app.callback(
        Output('portfolio-store', 'data', allow_duplicate=True),
        Input('auto-load-trigger', 'n_intervals'),
        prevent_initial_call=True
    )
    def auto_load_portfolio(n_intervals):
        if not os.path.exists(CSV_PATH):
            return dash.no_update

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

            return {'positions': positions}

        except Exception as e:
            print(f"Auto-load error: {e}")
            return dash.no_update
