import base64
import io
import pandas as pd
import yfinance as yf
import dash_bootstrap_components as dbc
from dash import Input, Output, State, callback_context, html

from utils.finance import fetch_price_at_date, adjust_shares_for_splits


def _resolve_stored_data(stored_data, initial_positions, run_id):
    """Session-Daten von einem FRÜHEREN Prozesslauf (run_id stimmt nicht überein oder
    fehlt) -> CSV-Startpositionen. Sonst unverändert (auch wenn leer) — 'zuletzt war es
    so, also bleibt es so' innerhalb desselben Laufs, über beliebig viele F5 hinweg."""
    if not stored_data or stored_data.get('run_id') != run_id:
        return {'positions': initial_positions, 'run_id': run_id}
    return stored_data


def register(app, initial_positions, run_id):
    @app.callback(
        [Output('buy-section', 'style'),
         Output('sell-section', 'style'),
         Output('btn-toggle-buy', 'active'),
         Output('btn-toggle-sell', 'active')],
        [Input('btn-toggle-buy', 'n_clicks'),
         Input('btn-toggle-sell', 'n_clicks')]
    )
    def toggle_sections(buy_clicks, sell_clicks):
        ctx = callback_context
        if not ctx.triggered:
            return {'display': 'block'}, {'display': 'none', 'borderColor': '#9e6a03'}, True, False
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'btn-toggle-buy':
            return {'display': 'block'}, {'display': 'none', 'borderColor': '#9e6a03'}, True, False
        return {'display': 'none'}, {'display': 'block', 'borderColor': '#9e6a03'}, False, True

    @app.callback(
        [Output('portfolio-store', 'data'),
         Output('add-status', 'children'),
         Output('sell-status', 'children')],
        [Input('btn-add', 'n_clicks'),
         Input('btn-clear', 'n_clicks'),
         Input('btn-sell', 'n_clicks'),
         Input('upload-csv', 'contents'),
         Input('server-run-id', 'data')],
        [State('input-ticker', 'value'),
         State('input-shares', 'value'),
         State('input-date', 'value'),
         State('sell-ticker', 'value'),
         State('sell-shares', 'value'),
         State('upload-csv', 'filename'),
         State('portfolio-store', 'data')]
    )
    def manage_portfolio(add_clicks, clear_clicks, sell_clicks, csv_contents, _run_id_trigger,
                         ticker, shares, buy_date,
                         sell_ticker, sell_shares, csv_filename, stored_data):
        # Robust gegenüber Dashs Initial-Call-Verhalten: läuft unabhängig davon, was den
        # Callback konkret ausgelöst hat. Innerhalb desselben Prozesslaufs ist run_id
        # bereits aktuell, dieser Zweig ändert dann nichts mehr.
        stored_data = _resolve_stored_data(stored_data, initial_positions, run_id)

        ctx = callback_context
        if not ctx.triggered:
            return stored_data, "", ""

        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if button_id == 'server-run-id':
            return stored_data, "", ""

        if button_id == 'upload-csv' and csv_contents is not None:
            try:
                content_type, content_string = csv_contents.split(',')
                decoded = base64.b64decode(content_string)
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

                required_cols = ['ticker', 'shares', 'buy_date']
                if not all(col in df.columns for col in required_cols):
                    return stored_data, "", dbc.Alert(
                        f"CSV muss folgende Spalten enthalten: {', '.join(required_cols)}",
                        color="danger", duration=4000
                    )

                positions = []
                for _, row in df.iterrows():
                    ticker_val = str(row['ticker']).upper()
                    shares_val = int(row['shares'])
                    buy_date_val = str(row['buy_date'])

                    if 'buy_price' in df.columns and pd.notna(row['buy_price']):
                        buy_price_val = float(row['buy_price'])
                    else:
                        buy_price_val = fetch_price_at_date(ticker_val, buy_date_val)
                        if buy_price_val is None:
                            continue

                    positions.append({
                        'ticker': ticker_val,
                        'shares': shares_val,
                        'buy_date': buy_date_val,
                        'buy_price': buy_price_val
                    })

                return {'positions': positions, 'run_id': run_id}, dbc.Alert(
                    f"✓ {len(positions)} Positionen aus '{csv_filename}' geladen",
                    color="success", duration=3000
                ), ""

            except Exception as e:
                error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                return stored_data, "", dbc.Alert(
                    f"Fehler beim CSV-Import: {error_msg}",
                    color="danger", duration=4000
                )

        if button_id == 'btn-clear':
            return {'positions': [], 'run_id': run_id}, dbc.Alert("Portfolio geleert", color="info", duration=3000), ""

        if button_id == 'btn-sell':
            if not sell_ticker or not sell_shares:
                return stored_data, "", dbc.Alert("Bitte Ticker und Anzahl eingeben", color="warning", duration=3000)

            positions = stored_data.get('positions', [])
            sell_ticker = sell_ticker.upper()

            for pos in positions:
                if pos['ticker'] == sell_ticker:
                    current_shares = adjust_shares_for_splits(pos['ticker'], pos['shares'], pos['buy_date'])

                    if current_shares < sell_shares:
                        return stored_data, "", dbc.Alert(
                            f"Nur {current_shares} Aktien von {sell_ticker} verfügbar",
                            color="danger", duration=3000
                        )

                    split_ratio = current_shares / pos['shares'] if pos['shares'] > 0 else 1
                    original_shares_to_remove = sell_shares / split_ratio
                    pos['shares'] -= original_shares_to_remove

                    if pos['shares'] < 0.01:
                        positions.remove(pos)
                        return {'positions': positions, 'run_id': run_id}, "", dbc.Alert(
                            f"{sell_ticker} komplett verkauft und entfernt",
                            color="success", duration=3000
                        )

                    return {'positions': positions, 'run_id': run_id}, "", dbc.Alert(
                        f"{sell_shares} Aktien von {sell_ticker} verkauft (verbleibend: {int(current_shares - sell_shares)})",
                        color="success", duration=3000
                    )

            return stored_data, "", dbc.Alert(f"{sell_ticker} nicht im Portfolio gefunden", color="danger", duration=3000)

        if button_id == 'btn-add':
            if not ticker or not shares or not buy_date:
                return stored_data, dbc.Alert("Bitte alle Pflichtfelder ausfüllen", color="warning", duration=3000), ""

            try:
                test = yf.Ticker(ticker.upper())
                hist = test.history(period="1d")
                if hist.empty:
                    return stored_data, dbc.Alert(f"Ticker '{ticker}' nicht gefunden", color="danger", duration=3000), ""

                buy_price = fetch_price_at_date(ticker.upper(), buy_date)
                if buy_price is None:
                    return stored_data, dbc.Alert(f"Kein Preis für {ticker} am {buy_date} gefunden", color="danger", duration=3000), ""

                new_position = {
                    'ticker': ticker.upper(),
                    'shares': int(shares),
                    'buy_date': buy_date,
                    'buy_price': float(buy_price)
                }
                positions = stored_data.get('positions', [])
                positions.append(new_position)
                return {'positions': positions, 'run_id': run_id}, dbc.Alert(f"{ticker.upper()} erfolgreich hinzugefügt", color="success", duration=3000), ""

            except Exception as e:
                return stored_data, dbc.Alert(f"Fehler: {str(e)}", color="danger", duration=3000), ""

        return stored_data, "", ""
