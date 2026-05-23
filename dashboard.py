import subprocess
import sys
import io
import os

# Fix Windows encoding issues for UTF-8 characters
os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# All packages should be pre-installed via venv
# No auto-installation to avoid package conflicts


import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from groq import Groq
import os
import traceback
from dotenv import load_dotenv
import base64
import io
import logging

# RAG imports
try:
    from rag.pipeline import RAGPipeline
    RAG_AVAILABLE = True
except Exception as e:
    print(f"Warning: RAG modules not available: {e}")
    RAG_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize RAG Pipeline in background thread
rag_pipeline = None

def _init_rag_background():
    global rag_pipeline, RAG_AVAILABLE
    if not RAG_AVAILABLE:
        return
    try:
        persist_dir = os.path.join(os.getcwd(), "data", "faiss")
        rag_pipeline = RAGPipeline(persist_dir=persist_dir)
        logger.info("RAG Pipeline initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize RAG pipeline: {e}")
        RAG_AVAILABLE = False

if RAG_AVAILABLE:
    import threading
    threading.Thread(target=_init_rag_background, daemon=True).start()


# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Custom CSS for minimalist black/white design
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>Portfolio Performance Dashboard</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: #0d1117;
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                min-height: 100vh;
                color: #c9d1d9;
            }
            .main-container {
                background: #161b22;
                border-radius: 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                padding: 40px;
                margin: 20px auto;
                max-width: 1600px;
            }
            .metric-card {
                background: #0d1117;
                border: 1px solid #30363d;
                border-radius: 4px;
                padding: 24px;
                color: #c9d1d9;
                box-shadow: none;
                transition: border-color 0.2s ease;
                margin-bottom: 20px;
            }
            .metric-card:hover {
                border-color: #58a6ff;;
            }
            .metric-value {
                font-size: 2.5rem;
                font-weight: 600;
                margin: 10px 0;
                color: #f0f6fc;
            }
            .metric-label {
                font-size: 0.75rem;
                color: #8b949e;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                font-weight: 500;
            }
            .metric-description {
                font-size: 0.85rem;
                color: #8b949e;
                margin-top: 5px;
            }
            .card-custom {
                border-radius: 4px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                border: 1px solid #30363d;
                margin-bottom: 20px;
                background: #0d1117;
            }
            .card-header-custom {
                background: #21262d;
                color: #f0f6fc;
                border-radius: 4px 4px 0 0 !important;
                padding: 16px 20px;
                font-weight: 600;
                font-size: 0.95rem;
                letter-spacing: 0.3px;
                border-bottom: 1px solid #30363d;
            }
            .input-group-custom {
                background: #0d1117;
                border-radius: 4px;
                padding: 24px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.3);
                border: 1px solid #30363d;
            }
            .btn-custom {
                border-radius: 4px;
                padding: 10px 24px;
                font-weight: 500;
                transition: all 0.2s ease;
                font-size: 0.9rem;
            }
            .btn-custom:hover {
                transform: translateY(-1px);
                box-shadow: 0 2px 8px rgba(0,0,0,0.4);
            }
            .btn-primary {
                background: #238636;
                border: 1px solid #238636;
                color: white;
            }
            .btn-primary:hover {
                background: #2ea043;
                border: 1px solid #2ea043;
                color: white;
            }
            .btn-secondary {
                background: #21262d;
                border: 1px solid #30363d;
                color: #c9d1d9;
            }
            .btn-secondary:hover {
                background: #30363d;
                border: 1px solid #58a6ff;
                color: #f0f6fc;
            }
            .btn-danger {
                background: #21262d;
                border: 1px solid #da3633;
                color: #f85149;
            }
            .btn-danger:hover {
                background: #da3633;
                border: 1px solid #da3633;
                color: white;
            }
            h1 {
                color: #f0f6fc;
                font-weight: 700;
                margin-bottom: 8px;
                letter-spacing: -0.5px;
            }
            .text-muted {
                color: #8b949e !important;
            }
            .nav-tabs {
                border-bottom: 2px solid #30363d;
            }
            .nav-tabs .nav-link {
                border-radius: 0;
                font-weight: 500;
                color: #8b949e;
                border: none;
                border-bottom: 2px solid transparent;
                padding: 12px 24px;
                font-size: 0.9rem;
            }
            .nav-tabs .nav-link.active {
                background: transparent;
                color: #f0f6fc;
                border-bottom: 2px solid #58a6ff;
            }
            .nav-tabs .nav-link:hover {
                color: #f0f6fc;
                border-bottom: 2px solid #8b949e;
            }
            .table {
                color: #c9d1d9;
            }
            .table-striped tbody tr:nth-of-type(odd) {
                background-color: rgba(110, 118, 129, 0.05);
            }
            .table-bordered {
                border-color: #30363d;
            }
            .table-bordered td, .table-bordered th {
                border-color: #30363d;
            }
            .table-hover tbody tr:hover {
                background-color: rgba(110, 118, 129, 0.1);
            }
            .alert-info {
                background-color: #1a2332;
                border-color: #388bfd;
                color: #58a6ff;
            }
            .alert-success {
                background-color: #1a2e1a;
                border-color: #238636;
                color: #3fb950;
            }
            .alert-warning {
                background-color: #332b1a;
                border-color: #9e6a03;
                color: #d29922;
            }
            .alert-danger {
                background-color: #2e1a1a;
                border-color: #da3633;
                color: #f85149;
            }
            label {
                color: #c9d1d9;
            }
            .form-control {
                background-color: #0d1117;
                border-color: #30363d;
                color: #c9d1d9;
            }
            .form-control:focus {
                background-color: #0d1117;
                border-color: #58a6ff;
                color: #c9d1d9;
            }

h4:not(#pnl-value) {
    color: #f0f6fc !important;
}
           input::placeholder {
                color: #6e7681;
            }

#ai-analysis-output,
#ai-analysis-output * {
    color: #f0f6fc !important;
}
#ai-text-content {
    color: #f0f6fc !important;
}
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

def fetch_price_at_date(ticker, date):
    """Fetch unadjusted historical price for a ticker at a specific date"""
    try:
        stock = yf.Ticker(ticker)
        start = pd.to_datetime(date) - timedelta(days=5)
        end = pd.to_datetime(date) + timedelta(days=5)

        # ✅ auto_adjust=False gibt unadjusted (echte historische) Preise
        hist = stock.history(start=start, end=end, auto_adjust=False)

        if not hist.empty:
            return hist['Close'].iloc[0]
        return None
    except:
        return None

def adjust_shares_for_splits(ticker, shares, buy_date):
    """
    Passt die Anzahl der Aktien für alle Splits seit Kaufdatum an
    Adjustiert Aktienanzahl für alle Splits seit Kaufdatum.

   Beispiel: 10 NVDA Aktien gekauft 2022
   → 10:1 Split 2024 → jetzt 100 Aktien

    """
    stock = yf.Ticker(ticker)

    # Hole Split-Historie
    splits = stock.splits

    if splits.empty:
        return shares

    # Filtere nur Splits NACH dem Kaufdatum
    buy_date = pd.to_datetime(buy_date)
    if buy_date.tz is None and splits.index.tz is not None:
        buy_date = buy_date.tz_localize(splits.index.tz)

    relevant_splits = splits[splits.index > buy_date]

    # Multipliziere shares mit allen Split-Faktoren
    adjusted_shares = shares
    for split_ratio in relevant_splits:
        adjusted_shares *= split_ratio

    return int(adjusted_shares)

def adjust_price_for_splits(ticker, price, date):
    """
    Rekonstruiert echte historische Preise aus yfinance-Daten.

    WICHTIG: yfinance gibt historische Preise IMMER split-adjusted zurück,
    auch mit auto_adjust=False. Beispiel NVDA 2022:
    - Echter Marktpreis damals: ~$150
    - yfinance returniert: ~$15 (adjustiert für 10:1 Split 2024)

    Diese Funktion multipliziert mit allen Split-Ratios NACH dem Datum,
    um den Original-Preis zu rekonstruieren: $15 * 10 = $150

    Args:
        ticker: Stock symbol
        price: Split-adjusted price von yfinance
        date: Datum für das der Preis gilt

    Returns:
        Echter historischer Marktpreis (unadjusted)
    """
    stock = yf.Ticker(ticker)
    splits = stock.splits

    if splits.empty:
        return price

    date = pd.to_datetime(date)
    if date.tz is None and splits.index.tz is not None:
        date = date.tz_localize(splits.index.tz)

    relevant_splits = splits[splits.index > date]

    # Multipliziere Preis mit Split-Faktoren
    adjusted_price = price
    for split_ratio in relevant_splits:
        adjusted_price *= split_ratio

    return adjusted_price

def calculate_portfolio_timeseries(positions):
    """Calculate portfolio value over time with cash-adjusted returns"""
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
        except:
            continue

    if not all_data:
        return pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(all_data)
    df = df.ffill().bfill()

    # Make buy_dates timezone-aware to match yfinance data
    portfolio_values = []
    for date in df.index:
        total_value = 0
        for pos in positions:
            ticker = pos['ticker']
            shares = pos['shares']
            buy_date = pd.to_datetime(pos['buy_date'])

            # Make buy_date timezone-aware if date has timezone
            if date.tz is not None and buy_date.tz is None:
                buy_date = buy_date.tz_localize(date.tz)

            if ticker in df.columns and date >= buy_date:
                price = df.loc[date, ticker]
                total_value += shares * price

        portfolio_values.append(total_value)

    result = pd.DataFrame({
        'Date': df.index,
        'Portfolio_Value': portfolio_values
    })

    result['Returns'] = result['Portfolio_Value'].pct_change()

    return result, df

def calculate_sortino_ratio(returns, risk_free_rate=0.02):
    """Calculate Sortino Ratio"""
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0:
        return np.nan

    downside_std = np.sqrt(np.mean(downside_returns**2))

    if downside_std == 0:
        return np.nan

    return np.sqrt(252) * excess_returns.mean() / downside_std

def calculate_calmar_ratio(returns, portfolio_values, annualized_return=None):
    """Calculate Calmar Ratio"""

    # Wenn annualized_return nicht gegeben, berechne ihn aus Returns
    if annualized_return is None:
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
        years = len(returns) / 252
        annualized_return = (1 + total_return) ** (1/years) - 1

    print(f"\nDEBUG Calmar:")
    print(f"  Annualized return (input): {annualized_return*100:.2f}%")

    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min())

    print(f"  Max drawdown: {max_drawdown*100:.2f}%")

    if max_drawdown == 0:
        return np.nan

    calmar = annualized_return / max_drawdown
    print(f"  Calmar = {annualized_return*100:.2f}% / {max_drawdown*100:.2f}% = {calmar:.3f}")

    return calmar

def calculate_rolling_sharpe(returns, window=120, risk_free_rate=0.02):
    """Calculate Rolling Sharpe Ratio"""
    excess_returns = returns - risk_free_rate/252
    rolling_mean = excess_returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()

    rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std
    return rolling_sharpe

# Layout
app.layout = html.Div([
    html.Div([
        dbc.Container([
            # Header
            dbc.Row([
                dbc.Col([
                    html.H1("Portfolio Performance Dashboard", className="text-center mb-2 mt-4"),
                    html.P("Professionelle Analyse mit Sortino Ratio, Calmar Ratio und Rolling Sharpe Ratio",
                           className="text-center text-muted mb-4",
                           style={'fontSize': '1.1rem'})
                ])
            ]),

            # Toggle Button
            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Kaufen", id="btn-toggle-buy", color="success", className="btn-custom", active=True),
                        dbc.Button("Verkaufen", id="btn-toggle-sell", color="warning", className="btn-custom"),
                    ], className="mb-3")
                ])
            ]),

            # Input Section (Buy)
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Position hinzufügen", className="mb-3", style={'fontWeight': '600'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Ticker Symbol", style={'fontWeight': '500'}),
                                dbc.Input(id="input-ticker", placeholder="z.B. AAPL", type="text",
                                         style={'borderRadius': '4px'})
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Anzahl Aktien", style={'fontWeight': '500'}),
                                dbc.Input(id="input-shares", placeholder="z.B. 10", type="number",
                                         style={'borderRadius': '4px'})
                            ], width=2),
                            dbc.Col([
                                dbc.Label("Kaufdatum", style={'fontWeight': '500'}),
                                dbc.Input(id="input-date", placeholder="YYYY-MM-DD", type="text",
                                         value="2022-01-01", style={'borderRadius': '4px'})
                            ], width=4),
                            dbc.Col([
                                dbc.Label(" "),
                                dbc.Button("Hinzufügen", id="btn-add", color="primary",
                                          className="w-100 btn-custom")
                            ], width=2)
                        ]),
                        html.Div(id="add-status", className="mt-3")
                    ], className="input-group-custom mb-3", id="buy-section")
                ])
            ]),

            # Sell Section
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Position verkaufen (Teilverkauf)", className="mb-3", style={'fontWeight': '600'}),
                        dbc.Row([
                            dbc.Col([
                                dbc.Label("Ticker Symbol", style={'fontWeight': '500'}),
                                dbc.Input(id="sell-ticker", placeholder="z.B. AAPL", type="text",
                                         style={'borderRadius': '4px'})
                            ], width=4),
                            dbc.Col([
                                dbc.Label("Anzahl zu verkaufen", style={'fontWeight': '500'}),
                                dbc.Input(id="sell-shares", placeholder="z.B. 5", type="number",
                                         style={'borderRadius': '4px'})
                            ], width=3),
                            dbc.Col([
                                dbc.Label(" "),
                                dbc.Button("Verkaufen", id="btn-sell", color="warning",
                                          className="w-100 btn-custom")
                            ], width=2),
                            dbc.Col([
                                dbc.Label(" "),
                                html.Small("Reduziert die Anzahl der Aktien in deiner Position",
                                          className="text-muted d-block mt-2")
                            ], width=3)
                        ]),
                        html.Div(id="sell-status", className="mt-3")
                    ], className="input-group-custom mb-3",
                       style={'borderColor': '#9e6a03', 'display': 'none'},
                       id="sell-section")
                ])
            ]),

            # Control Buttons
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Upload(
                            id='upload-csv',
                            children=dbc.Button([
                                html.I(className="me-2"),
                                "Portfolio aus CSV laden"
                            ], color="secondary", className="btn-custom"),
                            multiple=False,
                            style={'display': 'inline-block', 'marginRight': '10px'}
                        ),
                        dbc.Button("Portfolio leeren", id="btn-clear",
                                  color="danger", className="btn-custom",
                                  style={'display': 'inline-block'}),
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], className="mb-3")
            ]),
            dbc.Row([
                dbc.Col([
                    html.Div(id='upload-status')
                ], width=12)
            ]),

            # Tabs
            dbc.Tabs([
                # Overview Tab
                dbc.Tab(label="Overview", tab_id="tab-overview", children=[
                    html.Div([
                        # Portfolio Table
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="portfolio-table")
                            ])
                        ], className="mt-4 mb-4"),

                        # Key Metrics
                        dbc.Row([
                            dbc.Col([
                                html.Div(id="key-metrics")
                            ])
                        ], className="mb-4"),

                        # Main Chart
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="portfolio-value-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ], width=12)
                        ], className="mb-3"),
                    ])
                ]),

                # Analytics Tab
                dbc.Tab(label="Analytics", tab_id="tab-analytics", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="rolling-sharpe-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="drawdown-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ], width=6)
                        ], className="mt-4 mb-3"),

                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="allocation-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ], width=6),
                            dbc.Col([
                                dcc.Graph(id="correlation-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ], width=6)
                        ], className="mb-3"),
                    ])
                ]),

                # Benchmark Comparison Tab
                dbc.Tab(label="Benchmark Comparison", tab_id="tab-benchmark", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dcc.Graph(id="benchmark-chart",
                                         config={'displayModeBar': True, 'displaylogo': False})
                            ])
                        ], className="mt-4 mb-3"),

                        dbc.Row([
                            dbc.Col([
                                html.Div(id="benchmark-metrics")
                            ])
                        ], className="mb-4"),
                    ])
                ]),

                # AI Risk Analysis Tab
                dbc.Tab(label="AI Risk Analysis", tab_id="tab-ai", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("AI-gestützte Portfolio-Analyse", className="card-header-custom"),
                                    dbc.CardBody([
                                        html.P("Lass die AI dein Portfolio analysieren und Risikobewertungen erstellen.",
                                               className="text-muted mb-3"),
                                        dbc.Button("Portfolio analysieren", id="btn-analyze",
                                                  color="primary", className="btn-custom mb-3", size="lg"),
                                        dcc.Loading(
                                            id="loading-ai",
                                            type="default",
                                            children=html.Div(id="ai-analysis-output", style={'whiteSpace': 'pre-wrap'})
                                        )
                                    ])
                                ], className="card-custom")
                            ])
                        ], className="mt-4"),
                    ])
                ]),

                # RAG News Analysis Tab
                dbc.Tab(label="Nachrichten & RAG", tab_id="tab-rag", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("Yahoo Finance Nachrichten - RAG", className="card-header-custom"),
                                    dbc.CardBody([
                                        html.P("Holen Sie sich die neuesten Nachrichten von Yahoo Finance und stellen Sie Fragen mit RAG-gestützter Kontextanalyse.",
                                               className="text-muted mb-3"),
                                        
                                        # News Fetching Section
                                        html.Div([
                                            html.H5("1. Nachrichten indizieren", style={'fontSize': '1.1rem', 'fontWeight': '600'}),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Ticker (komma-getrennt, z.B. AAPL,MSFT,TSLA)", style={'fontWeight': '500'}),
                                                    dbc.Input(id="rag-ticker-input", 
                                                             placeholder="z.B. AAPL,MSFT,TSLA",
                                                             type="text",
                                                             value="AAPL,MSFT,TSLA",
                                                             style={'borderRadius': '4px', 'marginBottom': '10px'})
                                                ], width=8),
                                                dbc.Col([
                                                    dbc.Label("Pro Ticker: "),
                                                    dbc.Input(id="rag-limit-input", 
                                                             placeholder="5",
                                                             type="number",
                                                             value=5,
                                                             min=1,
                                                             max=20,
                                                             style={'borderRadius': '4px'})
                                                ], width=4)
                                            ]),
                                            dbc.Button("Nachrichten abrufen & indizieren", 
                                                      id="btn-rag-fetch",
                                                      color="primary", 
                                                      className="btn-custom mb-3",
                                                      size="md",
                                                      style={'marginTop': '10px'}),
                                            html.Div(id="rag-fetch-status", className="mb-3"),
                                        ], className="mb-4", style={'padding': '15px', 'border': '1px solid #30363d', 'borderRadius': '4px'}),

                                        # RAG Query Section
                                        html.Div([
                                            html.H5("2. Nachrichten durchsuchen & Fragen beantworten", style={'fontSize': '1.1rem', 'fontWeight': '600'}),
                                            dbc.Row([
                                                dbc.Col([
                                                    dbc.Label("Frage eingeben:", style={'fontWeight': '500'}),
                                                    dbc.Input(id="rag-query-input",
                                                             placeholder="z.B. Was sind die neuesten Nachrichten über Apple?",
                                                             type="text",
                                                             style={'borderRadius': '4px', 'marginBottom': '10px'})
                                                ], width=8),
                                                dbc.Col([
                                                    dbc.Label("Top-K Ergebnisse: "),
                                                    dbc.Input(id="rag-topk-input",
                                                             placeholder="5",
                                                             type="number",
                                                             value=5,
                                                             min=1,
                                                             max=20,
                                                             style={'borderRadius': '4px'})
                                                ], width=4)
                                            ]),
                                            dbc.Button("Mit RAG abfragen",
                                                      id="btn-rag-query",
                                                      color="success",
                                                      className="btn-custom mb-3",
                                                      size="md",
                                                      style={'marginTop': '10px'}),
                                            html.Div(id="rag-query-status", className="mb-3"),
                                        ], className="mb-3", style={'padding': '15px', 'border': '1px solid #30363d', 'borderRadius': '4px'}),

                                        # Results Section
                                        html.Div([
                                            html.H5("RAG Ergebnisse", style={'fontSize': '1.1rem', 'fontWeight': '600', 'marginTop': '20px'}),
                                            dcc.Loading(
                                                id="loading-rag",
                                                type="default",
                                                children=html.Div(id="rag-results-output", style={'whiteSpace': 'pre-wrap', 'minHeight': '100px'})
                                            )
                                        ], style={'marginTop': '20px'})
                                    ])
                                ], className="card-custom")
                            ])
                        ], className="mt-4"),
                    ])
                ]),
            ], id="tabs", active_tab="tab-overview"),

            dcc.Store(id='portfolio-store', data={'positions': []}),
            dcc.Store(id='analysis-data', data={}),
            dcc.Store(id='rag-status', data={'indexed_tickers': [], 'doc_count': 0}),

        ], fluid=True, className="main-container")
    ])
])

def calculate_twr_metrics(positions, price_df):
    """
    Berechnet Portfolio-Metriken mit Time-Weighted Return (TWR) Methode.

    TWR ist der Industrie-Standard für Performance-Messung, da er
    Verzerrungen durch Cash-Flows (Käufe/Verkäufe) eliminiert.

    Methodik:
    1. Teile Timeline in Sub-Perioden (zwischen Käufen)
    2. In jeder Periode sind Holdings konstant (kein Cash-Flow)
    3. Berechne Daily Returns innerhalb jeder Periode
    4. Erster Return: Tag 1 vs Tag 0 (nach Kauf vs Kauftag)
    5. Geometrische Verknüpfung: Total Return = Product(1 + daily_returns) - 1

    Dies entspricht GIPS (Global Investment Performance Standards).
    """
    if not positions:
        return None

    # Sortiere Positionen nach Kaufdatum
    positions_sorted = sorted(positions, key=lambda x: pd.to_datetime(x['buy_date']))

    # Definiere Sub-Perioden (zwischen Käufen)
    buy_dates = [pd.to_datetime(p['buy_date']) for p in positions_sorted]

    # Make ALL dates timezone-aware to match price_df
    if not price_df.empty and price_df.index.tz is not None:
        # Localize buy_dates to match price_df timezone
        buy_dates = [d.tz_localize(price_df.index.tz) if d.tz is None else d.tz_convert(price_df.index.tz)
                     for d in buy_dates]
        # Also make current time timezone-aware
        tz = price_df.index.tz
        now_aware = pd.Timestamp.now(tz=tz)
        buy_dates.append(now_aware)
    else:
        buy_dates.append(pd.Timestamp.now())

    # Initialize lists for collecting data across all periods
    all_daily_returns = []
    all_portfolio_values = []
    all_dates = []

    # Für jede Sub-Periode
    for i in range(len(buy_dates) - 1):
        period_start = buy_dates[i]
        period_end = buy_dates[i + 1]

        # Welche Positionen werden in dieser Periode gehalten?
        active_positions = []
        for p in positions_sorted:
            pos_buy_date = pd.to_datetime(p['buy_date'])
            # Make timezone-aware if needed
            if price_df.index.tz is not None and pos_buy_date.tz is None:
                pos_buy_date = pos_buy_date.tz_localize(price_df.index.tz)

            if pos_buy_date <= period_start:
                active_positions.append(p)

        if not active_positions:
            continue

        # Hole Daten für diese Periode
        period_mask = (price_df.index >= period_start) & (price_df.index < period_end)
        period_dates = price_df.index[period_mask]

        if len(period_dates) == 0:
            continue

        # Berechne tägliche Portfolio-Werte für diese Periode
        period_values = []
        period_dates_valid = []

        for date in period_dates:
            value = 0

            for pos in active_positions:
                ticker = pos['ticker']
                if ticker in price_df.columns:
                    try:
                        price = price_df.loc[date, ticker]
                        value += pos['shares'] * price
                    except:
                        continue

            if value > 0:
                period_values.append(value)
                period_dates_valid.append(date)

        # Speichere die Werte und Daten
        all_portfolio_values.extend(period_values)
        all_dates.extend(period_dates_valid)

        # Berechne Returns für diese Periode
        # WICHTIG: Startet bei Index 1 (nicht 2!)
        # Erster Return: Tag 1 vs Tag 0 (erster Tag nach Kauf vs Kauftag)
        print(f"\nDEBUG Period {i}:")
        print(f"  Period: {period_start.date()} → {period_end.date()}")
        print(f"  Active positions: {len(active_positions)}")
        print(f"  Valid dates in period: {len(period_values)}")

        if len(period_values) >= 2:
            # Index des ersten Wertes dieser Periode in all_portfolio_values
            period_start_idx = len(all_portfolio_values) - len(period_values)

            print(f"  Period start index in all_portfolio_values: {period_start_idx}")
            print(f"  Will calculate returns from index {period_start_idx + 1} to {len(all_portfolio_values)}")

            # ✅ KORRIGIERT: +1 statt +2
            # Berechne Returns ab dem zweiten Tag der Periode (Index period_start_idx + 1)
            for j in range(period_start_idx + 1, len(all_portfolio_values)):
                if all_portfolio_values[j-1] > 0:
                    ret = (all_portfolio_values[j] / all_portfolio_values[j-1]) - 1
                    all_daily_returns.append(ret)

                    # Debug für erste paar Returns
                    if len(all_daily_returns) <= 3:
                        print(f"    Return {len(all_daily_returns)}: {all_portfolio_values[j]:.2f} / {all_portfolio_values[j-1]:.2f} - 1 = {ret:.6f}")

    # Konvertiere zu Pandas Series
    if len(all_daily_returns) > 0 and len(all_dates) > len(all_daily_returns):
        # Die Returns-Dates starten einen Tag später als portfolio_values
        returns_dates = all_dates[1:len(all_daily_returns)+1]
    else:
        returns_dates = all_dates[:len(all_daily_returns)]

    returns_series = pd.Series(all_daily_returns, index=returns_dates)
    portfolio_series = pd.Series(all_portfolio_values, index=all_dates)

    # Berechne Metriken
    if len(returns_series) == 0:
        return None

    # Volatilität (annualisiert)
    volatility = returns_series.std() * np.sqrt(252) * 100

    # Sortino Ratio
    sortino = calculate_sortino_ratio(returns_series)

    # Rolling Sharpe (120-Tage Fenster)
    rolling_sharpe = calculate_rolling_sharpe(returns_series)

    # Max Drawdown
    cumulative = (1 + returns_series).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = abs(drawdown.min()) * 100

    # Total Return (geometrische Verknüpfung aller taeglichen Returns)
    total_return = ((1 + returns_series).prod() - 1) * 100

    # Annualized Return
    years = len(returns_series) / 252
    annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0

    # Calmar Ratio (verwendet annualized_return)
    calmar = calculate_calmar_ratio(returns_series, portfolio_series, annualized_return=annualized_return/100)

    # DEBUG OUTPUT
    print(f"\n{'='*60}")
    print(f"TWR METRICS SUMMARY:")
    print(f"{'='*60}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Annualized Return: {annualized_return:.2f}%")
    print(f"  Number of daily returns: {len(returns_series)}")
    print(f"  Number of portfolio values: {len(all_portfolio_values)}")
    print(f"  First return date: {returns_series.index[0] if len(returns_series) > 0 else 'N/A'}")
    print(f"  Last return date: {returns_series.index[-1] if len(returns_series) > 0 else 'N/A'}")
    print(f"  Returns mean (daily): {returns_series.mean():.6f}")
    print(f"  Returns std (daily): {returns_series.std():.6f}")
    print(f"  Volatility (annualized): {volatility:.2f}%")
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Sortino Ratio: {sortino:.3f}")
    print(f"  Calmar Ratio: {calmar:.3f}")
    print(f"{'='*60}\n")

    return {
        'returns': returns_series,
        'portfolio_values': portfolio_series,
        'volatility': volatility,
        'sortino': sortino,
        'calmar': calmar,
        'rolling_sharpe': rolling_sharpe,
        'max_drawdown': max_drawdown,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'drawdown_series': drawdown,
        'num_days': len(returns_series)
    }


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
        # Default: Buy section visible
        return {'display': 'block'}, {'display': 'none', 'borderColor': '#9e6a03'}, True, False

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'btn-toggle-buy':
        return {'display': 'block'}, {'display': 'none', 'borderColor': '#9e6a03'}, True, False
    else:
        return {'display': 'none'}, {'display': 'block', 'borderColor': '#9e6a03'}, False, True

@app.callback(
    [Output('portfolio-store', 'data'),
     Output('add-status', 'children'),
     Output('sell-status', 'children')],
    [Input('btn-add', 'n_clicks'),
     Input('btn-clear', 'n_clicks'),
     Input('btn-sell', 'n_clicks'),
     Input('upload-csv', 'contents')],
    [State('input-ticker', 'value'),
     State('input-shares', 'value'),
     State('input-date', 'value'),
     State('sell-ticker', 'value'),
     State('sell-shares', 'value'),
     State('upload-csv', 'filename'),
     State('portfolio-store', 'data')]
)
def manage_portfolio(add_clicks, clear_clicks, sell_clicks, csv_contents,
                     ticker, shares, buy_date,
                     sell_ticker, sell_shares, csv_filename, stored_data):
    ctx = callback_context

    if not ctx.triggered:
        return stored_data, "", ""

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    # CSV Upload
    if button_id == 'upload-csv' and csv_contents is not None:
        try:
            # Decode CSV
            content_type, content_string = csv_contents.split(',')
            decoded = base64.b64decode(content_string)
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))

            # Validate columns
            required_cols = ['ticker', 'shares', 'buy_date']
            if not all(col in df.columns for col in required_cols):
                return stored_data, "", dbc.Alert(
                    f"CSV muss folgende Spalten enthalten: {', '.join(required_cols)}",
                    color="danger", duration=4000
                )

            # Build positions from CSV
            positions = []
            for _, row in df.iterrows():
                ticker_val = str(row['ticker']).upper()
                shares_val = int(row['shares'])
                buy_date_val = str(row['buy_date'])

                # Get buy price from CSV or fetch it
                if 'buy_price' in df.columns and pd.notna(row['buy_price']):
                    buy_price_val = float(row['buy_price'])
                else:
                    buy_price_val = fetch_price_at_date(ticker_val, buy_date_val)
                    if buy_price_val is None:
                        continue  # Skip if price not found

                positions.append({
                    'ticker': ticker_val,
                    'shares': shares_val,
                    'buy_date': buy_date_val,
                    'buy_price': buy_price_val
                })

            return {'positions': positions}, dbc.Alert(
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
        return {'positions': []}, dbc.Alert("Portfolio geleert", color="info", duration=3000), ""

    if button_id == 'btn-sell':
        if not sell_ticker or not sell_shares:
            return stored_data, "", dbc.Alert("Bitte Ticker und Anzahl eingeben", color="warning", duration=3000)

        positions = stored_data.get('positions', [])
        sell_ticker = sell_ticker.upper()

        # Find position
        found = False
        for pos in positions:
            if pos['ticker'] == sell_ticker:
                found = True

                # Berechne aktuelle Shares nach Splits
                current_shares = adjust_shares_for_splits(pos['ticker'], pos['shares'], pos['buy_date'])

                if current_shares < sell_shares:
                    return stored_data, "", dbc.Alert(
                        f"Nur {current_shares} Aktien von {sell_ticker} verfügbar",
                        color="danger", duration=3000
                    )

                split_ratio = current_shares / pos['shares'] if pos['shares'] > 0 else 1
                original_shares_to_remove = sell_shares / split_ratio
                pos['shares'] -= original_shares_to_remove

                # Remove position if shares = 0
                if pos['shares'] < 0.01:
                    positions.remove(pos)
                    return {'positions': positions}, "", dbc.Alert(
                        f"{sell_ticker} komplett verkauft und entfernt",
                        color="success", duration=3000
                    )

                return {'positions': positions}, "", dbc.Alert(
                    f"{sell_shares} Aktien von {sell_ticker} verkauft (verbleibend: {int(current_shares - sell_shares)})",
                    color="success", duration=3000
                )

        if not found:
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

            return {'positions': positions}, dbc.Alert(f"{ticker.upper()} erfolgreich hinzugefügt", color="success", duration=3000), ""

        except Exception as e:
            return stored_data, dbc.Alert(f"Fehler: {str(e)}", color="danger", duration=3000), ""

    return stored_data, "", ""
@app.callback(
    [Output('portfolio-table', 'children'),
     Output('key-metrics', 'children'),
     Output('portfolio-value-chart', 'figure'),
     Output('rolling-sharpe-chart', 'figure'),
     Output('drawdown-chart', 'figure'),
     Output('allocation-chart', 'figure'),
     Output('correlation-chart', 'figure'),
     Output('benchmark-chart', 'figure'),
     Output('benchmark-metrics', 'children'),
     Output('analysis-data', 'data')],
    [Input('portfolio-store', 'data')]
)
def update_dashboard(stored_data):
    positions = stored_data.get('positions', [])

    empty_fig = go.Figure()
    empty_fig.update_layout(
        title="Keine Daten verfügbar",
        xaxis={'visible': False},
        yaxis={'visible': False},
        annotations=[{
            'text': 'Füge Positionen hinzu um die Analyse zu starten',
            'xref': 'paper',
            'yref': 'paper',
            'showarrow': False,
            'font': {'size': 16, 'color': '#8b949e'}
        }],
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#0d1117'
    )

    if not positions:
        return (
            dbc.Alert("Portfolio ist leer. Füge Positionen hinzu", color="info"),
            "",
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "", {}
        )

    try:
        # Calculate portfolio timeseries
        df, price_df = calculate_portfolio_timeseries(positions)

        if df.empty:
            return (
                dbc.Alert("Fehler beim Laden der Daten", color="danger"),
                "", empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, "", {}
            )

        # Get current prices and calculate P&L
        current_data = []
        total_current_value = 0
        total_invested = 0
        allocation_data = []

        for pos in positions:
            try:
                ticker_obj = yf.Ticker(pos['ticker'])
                current_price = ticker_obj.history(period="1d")['Close'].iloc[-1]
                current_shares = adjust_shares_for_splits(
                    pos['ticker'],
                    pos['shares'],
                    pos['buy_date']
                )
                display_buy_price = adjust_price_for_splits(
                    pos['ticker'],
                    pos['buy_price'],
                    pos['buy_date']
                )

                current_value = current_shares * current_price
                invested = pos['shares'] * display_buy_price  #
                pnl = current_value - invested
                pnl_pct = (pnl / invested) * 100 if invested > 0 else 0

                total_current_value += current_value
                total_invested += invested

                allocation_data.append({
                    'Ticker': pos['ticker'],
                    'Value': current_value
                })

                current_data.append({
                    'Ticker': pos['ticker'],
                    'Shares': f"{current_shares}",
                    'Buy Date': pos['buy_date'],
                    'Buy Price': f"${display_buy_price:.2f}",
                    'Current': f"${current_price:.2f}",
                    'Value': f"${current_value:.2f}",
                    'P&L': f"${pnl:.2f} ({pnl_pct:+.1f}%)"
                })
            except:
                continue

        # Create enhanced table
        table = dbc.Table.from_dataframe(
            pd.DataFrame(current_data),
            striped=True,
            bordered=True,
            hover=True,
            responsive=True,
            className="mb-0"
        )

        total_pnl = total_current_value - total_invested
        total_pnl_pct = ((total_current_value/total_invested - 1) * 100) if total_invested > 0 else 0

        portfolio_table = dbc.Card([
            dbc.CardHeader("Aktuelles Portfolio", className="card-header-custom"),
            dbc.CardBody([
                table,
                html.Hr(),
                dbc.Row([
                    dbc.Col([
                        html.H6("Gesamt investiert", className="text-muted"),
                        html.H4(f"${total_invested:,.2f}", style={'fontWeight': '700'})
                    ], width=4),
                    dbc.Col([
                        html.H6("Aktueller Wert", className="text-muted"),
                        html.H4(f"${total_current_value:,.2f}", style={'fontWeight': '700'})
                    ], width=4),
                    dbc.Col([
                        html.H6("Gesamt P&L", className="text-muted"),
                        html.H4(f"${total_pnl:,.2f} ({total_pnl_pct:+.1f}%)",
                               id="pnl-value",
                               style={'fontWeight': '700',
                                      'color': '#28a745' if total_pnl > 0 else '#dc3545'})
                    ], width=4),
                ])
            ])
        ], className="card-custom")

        # Calculate metrics using TWR method
        twr_metrics = calculate_twr_metrics(positions, price_df)

        if twr_metrics is None:
            # Fallback to old method if TWR fails
            returns = df['Returns'].dropna()
            portfolio_values = df['Portfolio_Value']

            sortino = calculate_sortino_ratio(returns)
            calmar = calculate_calmar_ratio(returns, portfolio_values)
            rolling_sharpe = calculate_rolling_sharpe(returns)

            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = abs(drawdown.min()) * 100

            total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
            years = len(returns) / 252
            annualized_return = ((1 + total_return/100) ** (1/years) - 1) * 100 if years > 0 else 0
            volatility = returns.std() * np.sqrt(252) * 100
        else:
            # Use TWR metrics
            returns = twr_metrics['returns']
            portfolio_values = twr_metrics['portfolio_values']
            sortino = twr_metrics['sortino']
            calmar = twr_metrics['calmar']
            rolling_sharpe = twr_metrics['rolling_sharpe']
            max_drawdown = twr_metrics['max_drawdown']
            total_return = twr_metrics['total_return']
            annualized_return = twr_metrics['annualized_return']
            volatility = twr_metrics['volatility']
            drawdown = twr_metrics['drawdown_series']

            # ✅ DEBUG: Zeige Drawdown-Details
            print(f"\n=== MAX DRAWDOWN DEBUG ===")
            print(f"Portfolio Value: ${portfolio_values.iloc[-1]:,.2f}")
            print(f"Max Drawdown: {max_drawdown:.2f}%")
            print(f"Drawdown an letztem Tag: {drawdown.iloc[-1]*100:.2f}%")
            print(f"Anzahl Positionen: {len(positions)}")

            # Finde wo der Max Drawdown war - AUF CUMULATIVE!
            cumulative = (1 + returns).cumprod()
            running_max_cum = cumulative.expanding().max()
            drawdown_cum = (cumulative - running_max_cum) / running_max_cum

            max_dd_idx = drawdown_cum.idxmin()
            max_dd_date = max_dd_idx
            max_dd_cumulative = cumulative.loc[max_dd_idx]

            # Finde den Peak davor
            peak_idx = running_max_cum.loc[:max_dd_idx].idxmax()
            peak_cumulative = cumulative.loc[peak_idx]

            print(f"Max DD Datum: {max_dd_date.date()}")
            print(f"Cumulative return beim Max DD: {max_dd_cumulative:.4f}")
            print(f"Peak davor (Datum): {peak_idx.date()}")
            print(f"Peak cumulative: {peak_cumulative:.4f}")
            print(f"Drop: {peak_cumulative:.4f} → {max_dd_cumulative:.4f} = {((max_dd_cumulative/peak_cumulative - 1)*100):.2f}%")
            print("=" * 40 + "\n")
        # Enhanced metrics cards
        metrics = dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div("Sortino Ratio", className="metric-label"),
                    html.Div(f"{sortino:.3f}" if not np.isnan(sortino) else "N/A", className="metric-value"),
                    html.Div("Risikoadjustierte Performance", className="metric-description")
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("Calmar Ratio", className="metric-label"),
                    html.Div(f"{calmar:.3f}" if not np.isnan(calmar) else "N/A", className="metric-value"),
                    html.Div("Return / Max Drawdown", className="metric-description")
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("Max Drawdown", className="metric-label"),
                    html.Div(f"{max_drawdown:.2f}%", className="metric-value"),
                    html.Div("Maximaler Verlust vom Peak", className="metric-description")
                ], className="metric-card")
            ], width=3),
            dbc.Col([
                html.Div([
                    html.Div("Volatilität (Ann.)", className="metric-label"),
                    html.Div(f"{volatility:.2f}%", className="metric-value"),
                    html.Div("Annualisierte Standardabweichung", className="metric-description")
                ], className="metric-card")
            ], width=3),
        ], className="mb-3")

        # Portfolio value chart
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=df['Date'],
            y=df['Portfolio_Value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='#58a6ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(88, 166, 255, 0.1)'
        ))
        fig1.update_layout(
            title={
                'text': "Portfolio-Wert Entwicklung",
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            xaxis_title="Datum",
            yaxis_title="Wert (USD)",
            hovermode='x unified',
            template='plotly_dark',
            height=500,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#c9d1d9')
        )

        # Rolling Sharpe chart

        fig2 = go.Figure()

        # Verwende portfolio_values Index wenn verfügbar
        if twr_metrics is not None and hasattr(portfolio_values, 'index'):
            sharpe_dates = portfolio_values.index
        else:
            sharpe_dates = df['Date']

        # Filtere NaN-Werte heraus
        rolling_sharpe_valid = rolling_sharpe.dropna()

        fig2.add_trace(go.Scatter(
            x=rolling_sharpe_valid.index,
            y=rolling_sharpe_valid.values,
            mode='lines',
            name='Rolling Sharpe (120d)',
            line=dict(color='#79c0ff', width=2)
        ))
        fig2.add_hline(y=0, line_dash="dash", line_color="#30363d", opacity=0.5)
        fig2.add_hline(y=1, line_dash="dot", line_color="#8b949e", opacity=0.3, annotation_text="Sharpe = 1")
        fig2.update_layout(
            title={
                'text': "Rolling Sharpe Ratio (60 Tage)",
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            xaxis_title="Datum",
            yaxis_title="Sharpe Ratio",
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#c9d1d9')
        )

        # Drawdown chart
        fig3 = go.Figure()

        # Verwende drawdown_dates wenn verfügbar
        if twr_metrics is not None and hasattr(drawdown, 'index'):
            drawdown_dates = drawdown.index
        else:
            drawdown_dates = df['Date']

        fig3.add_trace(go.Scatter(
            x=drawdown_dates,
            y=drawdown * 100,
            mode='lines',
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='#8b949e', width=2),
            fillcolor='rgba(139, 148, 158, 0.2)'
        ))
        fig3.update_layout(
            title={
                'text': "Portfolio Drawdown",
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            xaxis_title="Datum",
            yaxis_title="Drawdown (%)",
            hovermode='x unified',
            template='plotly_dark',
            height=400,
            paper_bgcolor='#0d1117',
            plot_bgcolor='#0d1117',
            font=dict(color='#c9d1d9')
        )

        # Asset Allocation Pie Chart
        fig4 = go.Figure(data=[go.Pie(
            labels=[item['Ticker'] for item in allocation_data],
            values=[item['Value'] for item in allocation_data],
            hole=0.4,
            marker=dict(colors=['#58a6ff', '#79c0ff', '#a5d6ff', '#c9d1d9', '#8b949e', '#6e7681', '#484f58', '#30363d']),
            textinfo='label+percent',
            textfont_size=12
        )])
        fig4.update_layout(
            title={
                'text': "Asset Allocation (nach Marktwert)",
                'font': {'size': 18, 'color': '#f0f6fc'}
            },
            template='plotly_dark',
            height=400,
            showlegend=True,
            paper_bgcolor='#0d1117',
            font=dict(color='#c9d1d9')
        )

        # Correlation Matrix
        if not price_df.empty and len(price_df.columns) > 1:
            returns_df = price_df.pct_change().dropna()
            corr_matrix = returns_df.corr()

            # Calculate individual asset volatilities
            asset_volatilities = {}
            for ticker in price_df.columns:
                asset_returns = price_df[ticker].pct_change().dropna()
                asset_vol = asset_returns.std() * np.sqrt(252) * 100
                asset_volatilities[ticker] = asset_vol

            fig5 = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Blues',
                zmid=0,
                text=corr_matrix.values,
                texttemplate='%{text:.2f}',
                textfont={"size": 10, "color": "#c9d1d9"},
                colorbar=dict(title="Korrelation")
            ))
            fig5.update_layout(
                title={
                    'text': "Korrelationsmatrix (Returns)",
                    'font': {'size': 18, 'color': '#f0f6fc'}
                },
                template='plotly_dark',
                height=400,
                xaxis={'side': 'bottom'},
                paper_bgcolor='#0d1117',
                plot_bgcolor='#0d1117',
                font=dict(color='#c9d1d9')
            )
        else:
            fig5 = empty_fig
            corr_matrix = pd.DataFrame()
            asset_volatilities = {}

        # Benchmark Comparison (SPY)
        try:
            spy = yf.Ticker('SPY')
            spy_hist = spy.history(start=df['Date'].min(), end=df['Date'].max())

            if not spy_hist.empty:
                portfolio_normalized = (df['Portfolio_Value'] / df['Portfolio_Value'].iloc[0]) * 100
                spy_normalized = (spy_hist['Close'] / spy_hist['Close'].iloc[0]) * 100

                fig6 = go.Figure()
                fig6.add_trace(go.Scatter(
                    x=df['Date'],
                    y=portfolio_normalized,
                    mode='lines',
                    name='Portfolio',
                    line=dict(color='#58a6ff', width=2)
                ))
                fig6.add_trace(go.Scatter(
                    x=spy_hist.index,
                    y=spy_normalized,
                    mode='lines',
                    name='S&P 500 (SPY)',
                    line=dict(color='#8b949e', width=2, dash='dash')
                ))
                fig6.update_layout(
                    title={
                        'text': "Portfolio vs. S&P 500 Benchmark (Normalisiert auf 100)",
                        'font': {'size': 18, 'color': '#f0f6fc'}
                    },
                    xaxis_title="Datum",
                    yaxis_title="Indexiert (Start = 100)",
                    hovermode='x unified',
                    template='plotly_dark',
                    height=500,
                    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                    paper_bgcolor='#0d1117',
                    plot_bgcolor='#0d1117',
                    font=dict(color='#c9d1d9')
                )

                # Benchmark metrics
                spy_returns = spy_hist['Close'].pct_change().dropna()
                spy_total_return = (spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0] - 1) * 100
                spy_volatility = spy_returns.std() * np.sqrt(252) * 100

                portfolio_vs_spy = total_return - spy_total_return

                benchmark_metrics = dbc.Row([
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Portfolio Return (TWR)", className="text-muted"),
                                html.H3(f"{total_return:.2f}%", style={'color': '#f0f6fc', 'fontWeight': '700'}),
                            ])
                        ], className="card-custom")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("S&P 500 Return", className="text-muted"),
                                html.H3(f"{spy_total_return:.2f}%", style={'color': '#8b949e', 'fontWeight': '700'}),
                            ])
                        ], className="card-custom")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Outperformance", className="text-muted"),
                                html.H3(f"{portfolio_vs_spy:+.2f}%",
                                       style={'color': '#28a745' if portfolio_vs_spy > 0 else '#dc3545',
                                              'fontWeight': '700'}),
                            ])
                        ], className="card-custom")
                    ], width=3),
                    dbc.Col([
                        dbc.Card([
                            dbc.CardBody([
                                html.H5("Portfolio Volatilität", className="text-muted"),
                                html.H3(f"{volatility:.2f}%", style={'color': '#8b949e', 'fontWeight': '700'}),
                            ])
                        ], className="card-custom")
                    ], width=3),
                ])
            else:
                fig6 = empty_fig
                benchmark_metrics = dbc.Alert("Benchmark-Daten nicht verfügbar", color="warning")
        except:
            fig6 = empty_fig
            benchmark_metrics = dbc.Alert("Fehler beim Laden der Benchmark-Daten", color="danger")

        # Prepare analysis data for AI
        # Korrelations-Statistiken berechnen
        if not corr_matrix.empty:
            # Obere Dreiecks-Matrix ohne Diagonale (vermeidet Duplikate)
            upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

            correlation_stats = {
                'avg_correlation': float(upper_triangle.mean()),
                'max_correlation': float(upper_triangle.max()),
                'min_correlation': float(upper_triangle.min()),
                'high_correlations': []
            }

            # Finde starke Korrelationen (>0.7)
            for i in range(len(corr_matrix)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_value = corr_matrix.iloc[i, j]
                    if abs(corr_value) > 0.7:
                        ticker1 = corr_matrix.index[i]
                        ticker2 = corr_matrix.columns[j]
                        correlation_stats['high_correlations'].append(
                            f"{ticker1}-{ticker2}: {corr_value:.2f}"
                        )
        else:
            correlation_stats = {
                'avg_correlation': 0,
                'max_correlation': 0,
                'min_correlation': 0,
                'high_correlations': []
            }

        analysis_data = {
            'positions': positions,
            'allocation': allocation_data,
            'metrics': {
                'sortino': float(sortino) if not np.isnan(sortino) else 0,
                'calmar': float(calmar) if not np.isnan(calmar) else 0,
                'max_drawdown': float(max_drawdown),
                'volatility': float(volatility),
                'total_return': float(total_return),
                'annualized_return': float(annualized_return)
            },
            'rolling_sharpe': {
                'current': float(rolling_sharpe.iloc[-1]) if len(rolling_sharpe) > 0 and not pd.isna(rolling_sharpe.iloc[-1]) else 0,
                'mean': float(rolling_sharpe.mean()) if len(rolling_sharpe) > 0 and not pd.isna(rolling_sharpe.mean()) else 0,
                'min': float(rolling_sharpe.min()) if len(rolling_sharpe) > 0 and not pd.isna(rolling_sharpe.min()) else 0,
                'max': float(rolling_sharpe.max()) if len(rolling_sharpe) > 0 and not pd.isna(rolling_sharpe.max()) else 0,
                'std': float(rolling_sharpe.std()) if len(rolling_sharpe) > 0 and not pd.isna(rolling_sharpe.std()) else 0,
            },
            'correlation_stats': correlation_stats,
            'asset_volatilities': asset_volatilities,
            'correlation_matrix': corr_matrix.to_dict() if not corr_matrix.empty else {}
        }

        return portfolio_table, metrics, fig1, fig2, fig3, fig4, fig5, fig6, benchmark_metrics, analysis_data

    except Exception as e:
        print(f"ERROR in update_dashboard: {str(e)}")
        import traceback
        traceback.print_exc()

        return (
            dbc.Alert(f"Fehler beim Laden: {str(e)}", color="danger"),
            "",
            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
            "",
            {}
        )

# AI Analysis Callback (moved before app.run)
@app.callback(
    Output('ai-analysis-output', 'children'),
    [Input('btn-analyze', 'n_clicks')],
    [State('analysis-data', 'data')]
)
def analyze_portfolio_with_ai(n_clicks, analysis_data):
    if not n_clicks:
        return ""

    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

    try:
        # Initialize Groq API
        api_key = os.environ.get("GROQ_API_KEY")

        client = Groq(
            api_key=os.environ.get("GROQ_API_KEY"),
        )


        



        # Get metrics
        metrics = analysis_data.get('metrics', {})
        allocation = analysis_data.get('allocation', [])
        rolling_sharpe = analysis_data.get('rolling_sharpe', {})
        correlation_stats = analysis_data.get('correlation_stats', {})

        # Build comprehensive prompt
        prompt = f"""Analysiere diese Portfolio-Metriken und gib mir professionelles Feedback:

PERFORMANCE METRIKEN:
- Sortino Ratio: {metrics.get('sortino', 0):.3f}
- Calmar Ratio: {metrics.get('calmar', 0):.3f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
- Volatilität (annualisiert): {metrics.get('volatility', 0):.2f}%
- Gesamt Return: {metrics.get('total_return', 0):.2f}%
- Annualisierter Return: {metrics.get('annualized_return', 0):.2f}%

ROLLING SHARPE RATIO (60-Tage):
- Aktuell: {rolling_sharpe.get('current', 0):.3f}
- Durchschnitt: {rolling_sharpe.get('mean', 0):.3f}
- Range: {rolling_sharpe.get('min', 0):.3f} bis {rolling_sharpe.get('max', 0):.3f}
- Konsistenz (Std): {rolling_sharpe.get('std', 0):.3f}

KORRELATIONS-ANALYSE:
- Durchschnittliche Korrelation: {correlation_stats.get('avg_correlation', 0):.3f}
- Höchste Korrelation: {correlation_stats.get('max_correlation', 0):.3f}
- Niedrigste Korrelation: {correlation_stats.get('min_correlation', 0):.3f}
{'- Starke Korrelationen (>0.7): ' + ', '.join(correlation_stats.get('high_correlations', [])) if correlation_stats.get('high_correlations') else '- Keine besonders starken Korrelationen gefunden'}

ASSET VERTEILUNG:
{chr(10).join([f"- {a['Ticker']}: ${a['Value']:,.2f}" for a in allocation])}

Bitte bewerte diese Metriken kurz und prägnant:
1. Wie gut ist die risikoadjustierte Performance (Sortino & Calmar)?
2. Ist der Max Drawdown akzeptabel?
3. Wie ist das Verhältnis von Return zu Volatilität?
4. Wie stabil ist die Performance über Zeit? (Rolling Sharpe Analyse)
5. Ist das Portfolio gut diversifiziert? (Korrelations-Analyse)
6. Gibt es klare Warnsignale oder Stärken?

Halte die Antwort auf 8-10 Sätze. Sei konkret und direkt. Bearbeite die Fragen nacheinander"""

        # Call Groq API

        response = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model="llama-3.3-70b-versatile",
            max_tokens=2048,
        )

        # Extract response
        analysis_text = response.choices[0].message.content

        # Format output nicely
        return dbc.Card([
            dbc.CardBody([
                html.H5("AI Feedback zu deinen Metriken", className="mb-3", style={'color': '#f0f6fc'}),
               html.Div(analysis_text, id="ai-text-content", style={
    'whiteSpace': 'pre-wrap',
    'lineHeight': '1.6',
    'fontSize': '0.95rem',
    'color': '#f0f6fc'
}),
                html.Hr(),
                html.Small("Erstellt mit Groq", className="text-muted")
            ])
        ], className="card-custom mt-3")

    except Exception as e:
        return dbc.Alert([
            html.H5("Fehler bei der AI-Analyse", className="mb-2"),
            html.P(f"Fehlerdetails: {str(e)}"),
            html.Hr(),
            html.Small([
                "Mögliche Ursachen:",
                html.Ul([
                    html.Li("GOOGLE_API_KEY oder GEMINI_API_KEY nicht gesetzt"),
                    html.Li("Ungültiger API-Schlüssel"),
                    html.Li("Netzwerkprobleme"),
                    html.Li("API-Rate-Limit erreicht")
                ])
            ])
        ], color="danger")
    

# RAG Callbacks

# Callback: Fetch and Index News
@app.callback(
    [Output('rag-fetch-status', 'children'),
     Output('rag-status', 'data')],
    [Input('btn-rag-fetch', 'n_clicks')],
    [State('rag-ticker-input', 'value'),
     State('rag-limit-input', 'value')],
    prevent_initial_call=True
)
def fetch_and_index_news(n_clicks, tickers_str, limit):
    if not n_clicks or not rag_pipeline:
        return dbc.Alert("RAG Pipeline nicht verfügbar", color="warning"), {}

    if not tickers_str:
        return dbc.Alert("Bitte Ticker eingeben (komma-getrennt)", color="warning"), {}

    try:
        # Parse tickers
        tickers = [t.strip().upper() for t in tickers_str.split(',')]
        limit = int(limit) if limit else 5

        logger.info(f"Starting news indexing for tickers: {tickers}, limit: {limit}")

        # Index news
        stats = rag_pipeline.index_news_for_tickers(tickers, news_limit=limit)

        # Get vectorstore stats
        db_stats = rag_pipeline.get_stats()

        status_msg = dbc.Alert([
            html.H5("✅ Nachrichten erfolgreich indiziert!", className="mb-2"),
            dbc.Row([
                dbc.Col([
                    html.P(f"Ticker verarbeitet: {stats['tickers_successful']} / {stats['tickers_processed']}")
                ], width=6),
                dbc.Col([
                    html.P(f"Artikel: {stats['total_articles']}")
                ], width=6)
            ]),
            dbc.Row([
                dbc.Col([
                    html.P(f"Chunks erstellt: {stats['total_chunks']}")
                ], width=6),
                dbc.Col([
                    html.P(f"Gesamt indexiert: {db_stats['indexed_documents']} Dokumente")
                ], width=6)
            ]),
            html.Hr(),
            html.Small(f"Zeitstempel: {stats['timestamp']}")
        ], color="success", className="mb-3")

        rag_status_data = {
            'indexed_tickers': tickers,
            'doc_count': db_stats['indexed_documents'],
            'last_index_time': stats['timestamp']
        }

        return status_msg, rag_status_data

    except Exception as e:
        logger.error(f"Error indexing news: {e}")
        return dbc.Alert([
            html.H5("❌ Fehler beim Indizieren", color="danger"),
            html.P(f"Fehler: {str(e)}")
        ], color="danger"), {}


# Callback: RAG Query
@app.callback(
    Output('rag-results-output', 'children'),
    [Input('btn-rag-query', 'n_clicks')],
    [State('rag-query-input', 'value'),
     State('rag-topk-input', 'value'),
     State('rag-status', 'data')],
    prevent_initial_call=True
)
def query_with_rag(n_clicks, query, topk, rag_status):
    if not n_clicks or not rag_pipeline:
        return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")

    if not query:
        return dbc.Alert("Bitte eine Frage eingeben", color="warning")

    if rag_status.get('doc_count', 0) == 0:
        return dbc.Alert("Keine Nachrichten indexiert. Bitte zuerst Nachrichten abrufen.", color="info")

    try:
        topk = int(topk) if topk else 5
        logger.info(f"RAG query: '{query[:60]}...', top_k: {topk}")

        # Retrieve context
        retrieved = rag_pipeline.retrieve_context(query, top_k=topk)

        if not retrieved:
            return dbc.Alert("Keine relevanten Nachrichten gefunden", color="info")

        # Format context for LLM
        context = rag_pipeline.format_context_for_llm(retrieved, max_tokens=2000)

        # Create Groq prompt with RAG context
        prompt = f"""Du bist ein Finanzanalyst. Beantworte die folgende Frage basierend auf den gegebenen Nachrichten-Kontext.

KONTEXT (Nachrichten):
{context}

FRAGE: {query}

Bitte:
1. Beantworte die Frage basierend auf den gegebenen Nachrichten
2. Zitiere die Quellen (Links) wo relevant
3. Markiere besonders wichtige Punkte
4. Halte die Antwort prägnant (max 500 Wörter)"""

        # Call Groq API
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
            )
            llm_response = response.choices[0].message.content
        except Exception as e:
            logger.warning(f"Groq API error: {e}, using context summary instead")
            llm_response = f"Groq API nicht verfügbar. Hier sind die gefundenen Nachrichten:\n\n{context}"

        # Build result output
        result_components = [
            html.H5("🤖 RAG-gestützte Antwort", style={'color': '#58a6ff', 'marginTop': '20px', 'marginBottom': '10px'}),
            html.Div(llm_response, style={
                'whiteSpace': 'pre-wrap',
                'lineHeight': '1.6',
                'fontSize': '0.95rem',
                'color': '#f0f6fc',
                'marginBottom': '20px',
                'padding': '15px',
                'backgroundColor': '#0d1117',
                'borderLeft': '3px solid #58a6ff',
                'borderRadius': '4px'
            }),
            html.Hr(),
            html.H5("📰 Quelle (Top-K Nachrichten)", style={'color': '#79c0ff', 'marginBottom': '10px'}),
        ]

        # Add source cards
        for i, chunk in enumerate(retrieved, 1):
            metadata = chunk.get('metadata', {})
            title = metadata.get('title', 'N/A')
            ticker = metadata.get('ticker', 'N/A')
            link = metadata.get('link', '#')
            source = metadata.get('source', 'Unknown')
            text = chunk.get('text', '')

            result_components.append(
                dbc.Card([
                    dbc.CardBody([
                        dbc.Row([
                            dbc.Col([
                                html.H6(f"[{i}] {ticker} - {title[:60]}", style={'color': '#58a6ff'}),
                            ], width=9),
                            dbc.Col([
                                html.Small(source, className="text-muted")
                            ], width=3)
                        ]),
                        html.P(text[:200] + "...", className="text-muted small", style={'marginTop': '10px', 'marginBottom': '10px'}),
                        html.A("📌 Link zur Nachricht", href=link, target="_blank", className="small",
                               style={'color': '#79c0ff', 'textDecoration': 'underline'})
                    ])
                ], className="card-custom", style={'marginBottom': '10px'})
            )

        return html.Div(result_components)

    except Exception as e:
        logger.error(f"Error in RAG query: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.H5("❌ Fehler bei RAG-Abfrage"),
            html.P(f"Fehler: {str(e)}")
        ], color="danger")


if __name__ == '__main__':
    app.run(debug=True, port=8050)