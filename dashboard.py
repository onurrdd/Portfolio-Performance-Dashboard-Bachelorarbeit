import subprocess
import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import logging
from dotenv import load_dotenv

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- RAG (lazy) ---
# RAG wird NICHT beim Start geladen, sondern erst beim ersten Gebrauch im RAG-Tab
# (via RAGProvider.get()). Der Provider selbst ist billig und lädt weder torch noch
# das Embedding-Modell. So starten App, Ticker, Anomalie-Erkennung und Naive-LLM ohne
# Verzögerung; das schwere RAG-Laden passiert erst, wenn RAG tatsächlich benutzt wird.
from rag.provider import RAGProvider
rag_provider = RAGProvider()

import auto_load
from utils.finance import fetch_price_at_date

_initial_positions = auto_load.load_initial_positions(fetch_price_at_date)

# --- App ---
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

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
            .main-container { background: #161b22; border-radius: 0; box-shadow: 0 1px 3px rgba(0,0,0,0.3); padding: 40px; margin: 20px auto; max-width: 1600px; }
            .metric-card { background: #0d1117; border: 1px solid #30363d; border-radius: 4px; padding: 24px; color: #c9d1d9; box-shadow: none; transition: border-color 0.2s ease; margin-bottom: 20px; }
            .metric-card:hover { border-color: #58a6ff; }
            .metric-value { font-size: 2.5rem; font-weight: 600; margin: 10px 0; color: #f0f6fc; }
            .metric-label { font-size: 0.75rem; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; font-weight: 500; }
            .metric-description { font-size: 0.85rem; color: #8b949e; margin-top: 5px; }
            .card-custom { border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); border: 1px solid #30363d; margin-bottom: 20px; background: #0d1117; }
            .card-header-custom { background: #21262d; color: #f0f6fc; border-radius: 4px 4px 0 0 !important; padding: 16px 20px; font-weight: 600; font-size: 0.95rem; letter-spacing: 0.3px; border-bottom: 1px solid #30363d; }
            .input-group-custom { background: #0d1117; border-radius: 4px; padding: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.3); border: 1px solid #30363d; }
            .btn-custom { border-radius: 4px; padding: 10px 24px; font-weight: 500; transition: all 0.2s ease; font-size: 0.9rem; }
            .btn-custom:hover { transform: translateY(-1px); box-shadow: 0 2px 8px rgba(0,0,0,0.4); }
            .btn-primary { background: #238636; border: 1px solid #238636; color: white; }
            .btn-primary:hover { background: #2ea043; border: 1px solid #2ea043; color: white; }
            .btn-secondary { background: #21262d; border: 1px solid #30363d; color: #c9d1d9; }
            .btn-secondary:hover { background: #30363d; border: 1px solid #58a6ff; color: #f0f6fc; }
            .btn-danger { background: #21262d; border: 1px solid #da3633; color: #f85149; }
            .btn-danger:hover { background: #da3633; border: 1px solid #da3633; color: white; }
            h1 { color: #f0f6fc; font-weight: 700; margin-bottom: 8px; letter-spacing: -0.5px; }
            .text-muted { color: #8b949e !important; }
            .nav-tabs { border-bottom: 2px solid #30363d; display: flex; }
            .nav-tabs .nav-link { border-radius: 0; font-weight: 500; color: #8b949e; border: none; border-bottom: 2px solid transparent; padding: 12px 24px; font-size: 0.9rem; }
            .nav-tabs .nav-link.active { background: transparent; color: #f0f6fc; border-bottom: 2px solid #58a6ff; }
            .nav-tabs .nav-link:hover { color: #f0f6fc; border-bottom: 2px solid #8b949e; }
            .nav-tabs .nav-item:nth-last-child(2) { margin-left: auto; }
            .table { color: #c9d1d9; }
            .table-striped tbody tr:nth-of-type(odd) { background-color: rgba(110, 118, 129, 0.05); }
            .table-bordered { border-color: #30363d; }
            .table-bordered td, .table-bordered th { border-color: #30363d; }
            .table-hover tbody tr:hover { background-color: rgba(110, 118, 129, 0.1); }
            .alert-info { background-color: #1a2332; border-color: #388bfd; color: #58a6ff; }
            .alert-success { background-color: #1a2e1a; border-color: #238636; color: #3fb950; }
            .alert-warning { background-color: #332b1a; border-color: #9e6a03; color: #d29922; }
            .alert-danger { background-color: #2e1a1a; border-color: #da3633; color: #f85149; }
            label { color: #c9d1d9; }
            .form-control { background-color: #0d1117; border-color: #30363d; color: #c9d1d9; }
            .form-control:focus { background-color: #0d1117; border-color: #58a6ff; color: #c9d1d9; }
            h4:not(#pnl-value) { color: #f0f6fc !important; }
            input::placeholder { color: #6e7681; }
            #ai-analysis-output, #ai-analysis-output * { color: #f0f6fc !important; }
            #ai-text-content { color: #f0f6fc !important; }
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

app.layout = html.Div([
    html.Div([
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.H1("Portfolio Performance Dashboard", className="text-center mb-2 mt-4"),
                    html.P("Professionelle Analyse mit Sortino Ratio, Calmar Ratio und Rolling Sharpe Ratio",
                           className="text-center text-muted mb-4", style={'fontSize': '1.1rem'})
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.ButtonGroup([
                        dbc.Button("Kaufen", id="btn-toggle-buy", color="success", className="btn-custom", active=True),
                        dbc.Button("Verkaufen", id="btn-toggle-sell", color="warning", className="btn-custom"),
                    ], className="mb-3")
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Position hinzufügen", className="mb-3", style={'fontWeight': '600'}),
                        dbc.Row([
                            dbc.Col([dbc.Label("Ticker Symbol"), dbc.Input(id="input-ticker", placeholder="z.B. AAPL", type="text")], width=4),
                            dbc.Col([dbc.Label("Anzahl Aktien"), dbc.Input(id="input-shares", placeholder="z.B. 10", type="number")], width=2),
                            dbc.Col([dbc.Label("Kaufdatum"), dbc.Input(id="input-date", placeholder="YYYY-MM-DD", type="text", value="2022-01-01")], width=4),
                            dbc.Col([dbc.Label(" "), dbc.Button("Hinzufügen", id="btn-add", color="primary", className="w-100 btn-custom")], width=2)
                        ]),
                        html.Div(id="add-status", className="mt-3")
                    ], className="input-group-custom mb-3", id="buy-section")
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.H5("Position verkaufen (Teilverkauf)", className="mb-3", style={'fontWeight': '600'}),
                        dbc.Row([
                            dbc.Col([dbc.Label("Ticker Symbol"), dbc.Input(id="sell-ticker", placeholder="z.B. AAPL", type="text")], width=4),
                            dbc.Col([dbc.Label("Anzahl zu verkaufen"), dbc.Input(id="sell-shares", placeholder="z.B. 5", type="number")], width=3),
                            dbc.Col([dbc.Label(" "), dbc.Button("Verkaufen", id="btn-sell", color="warning", className="w-100 btn-custom")], width=2),
                            dbc.Col([dbc.Label(" "), html.Small("Reduziert die Anzahl der Aktien", className="text-muted d-block mt-2")], width=3)
                        ]),
                        html.Div(id="sell-status", className="mt-3")
                    ], className="input-group-custom mb-3", style={'borderColor': '#9e6a03', 'display': 'none'}, id="sell-section")
                ])
            ]),

            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Upload(
                            id='upload-csv',
                            children=dbc.Button("Portfolio aus CSV laden", color="secondary", className="btn-custom"),
                            multiple=False,
                            style={'display': 'inline-block', 'marginRight': '10px'}
                        ),
                        dbc.Button("Portfolio leeren", id="btn-clear", color="danger", className="btn-custom"),
                    ], style={'display': 'flex', 'alignItems': 'center'})
                ], className="mb-3")
            ]),
            dbc.Row([dbc.Col([html.Div(id='upload-status')], width=12)]),

            dbc.Tabs([
                dbc.Tab(label="Overview", tab_id="tab-overview", children=[
                    html.Div([
                        dbc.Row([dbc.Col([html.Div(id="portfolio-table")])], className="mt-4 mb-4"),
                        dbc.Row([dbc.Col([html.Div(id="key-metrics")])], className="mb-4"),
                        dbc.Row([dbc.Col([dcc.Graph(id="portfolio-value-chart", config={'displayModeBar': True, 'displaylogo': False})], width=12)], className="mb-3"),
                    ])
                ]),

                dbc.Tab(label="Analytics", tab_id="tab-analytics", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([dcc.Graph(id="rolling-sharpe-chart", config={'displayModeBar': True, 'displaylogo': False})], width=6),
                            dbc.Col([dcc.Graph(id="drawdown-chart", config={'displayModeBar': True, 'displaylogo': False})], width=6)
                        ], className="mt-4 mb-3"),
                        dbc.Row([
                            dbc.Col([dcc.Graph(id="allocation-chart", config={'displayModeBar': True, 'displaylogo': False})], width=6),
                            dbc.Col([dcc.Graph(id="correlation-chart", config={'displayModeBar': True, 'displaylogo': False})], width=6)
                        ], className="mb-3"),
                    ])
                ]),

                dbc.Tab(label="Benchmark Comparison", tab_id="tab-benchmark", children=[
                    html.Div([
                        dbc.Row([dbc.Col([dcc.Graph(id="benchmark-chart", config={'displayModeBar': True, 'displaylogo': False})])], className="mt-4 mb-3"),
                        dbc.Row([dbc.Col([html.Div(id="benchmark-metrics")])], className="mb-4"),
                    ])
                ]),

                dbc.Tab(label="AI Risk Analysis", tab_id="tab-ai", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("AI-gestützte Portfolio-Analyse", className="card-header-custom"),
                                    dbc.CardBody([
                                        html.P("Lass die AI dein Portfolio analysieren und Risikobewertungen erstellen.", className="text-muted mb-3"),
                                        dbc.Button("Portfolio analysieren", id="btn-analyze", color="primary", className="btn-custom mb-3", size="lg"),
                                        dcc.Loading(id="loading-ai", type="default",
                                                   children=html.Div(id="ai-analysis-output", style={'whiteSpace': 'pre-wrap'}))
                                    ])
                                ], className="card-custom")
                            ])
                        ], className="mt-4"),
                    ])
                ]),

                dbc.Tab(label="Naive LLM", tab_id="tab-naive-llm", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("AI-gestützte Portfolio-Analyse", className="card-header-custom"),
                                    dbc.CardBody([
                                        html.P("Lass die AI dein Portfolio analysieren und Risikobewertungen erstellen.", className="text-muted mb-3"),
                                        dbc.Button("Portfolio analysieren", id="btn-naive-llm-attribution", color="primary", className="btn-custom mb-3", size="lg"),
                                        dcc.Loading(id="loading-naive-llm", type="default",
                                                   children=html.Div(id="naive-llm-output", style={'whiteSpace': 'pre-wrap'}))
                                    ])
                                ], className="card-custom")
                            ])
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Anomalietage anzeigen (vorübergehend/Debug)", id="btn-toggle-breaks",
                                          color="secondary", className="btn-custom mb-3"),
                                dbc.Collapse(
                                    html.Div(id="breaks-table-container"),
                                    id="collapse-breaks",
                                    is_open=False,
                                )
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Prompt anzeigen (vorübergehend/Debug)", id="btn-toggle-naive-prompt",
                                          color="secondary", className="btn-custom mb-3"),
                                dbc.Collapse(
                                    html.Div(id="naive-prompt-container"),
                                    id="collapse-naive-prompt",
                                    is_open=False,
                                )
                            ])
                        ]),
                    ])
                ]),

                dbc.Tab(label="LLM mit RAG", tab_id="tab-rag", children=[
                    html.Div([
                        dbc.Row([
                            dbc.Col([
                                dbc.Card([
                                    dbc.CardHeader("AI-gestützte Portfolio-Analyse (RAG)", className="card-header-custom"),
                                    dbc.CardBody([
                                        html.P("Gleiche Analyse wie im Naive-LLM-Tab, jedoch mit abgerufenem Nachrichten-Kontext (RAG). Der Prompt ist identisch — so wird der Effekt des Retrievals sichtbar.", className="text-muted mb-3"),
                                        dbc.Button("Mit RAG analysieren", id="btn-rag-analyze", color="primary", className="btn-custom mb-3", size="lg"),
                                        dcc.Loading(id="loading-rag", type="default",
                                                   children=html.Div(id="rag-llm-output", style={'whiteSpace': 'pre-wrap'}))
                                    ])
                                ], className="card-custom")
                            ])
                        ], className="mt-4"),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Abgerufene Nachrichten anzeigen (vorübergehend/Debug)", id="btn-rag-toggle-news",
                                          color="secondary", className="btn-custom mb-3"),
                                dbc.Collapse(
                                    html.Div(id="rag-news-table-container"),
                                    id="collapse-rag-news",
                                    is_open=False,
                                )
                            ])
                        ]),
                        dbc.Row([
                            dbc.Col([
                                dbc.Button("Prompt anzeigen (vorübergehend/Debug)", id="btn-toggle-rag-prompt",
                                          color="secondary", className="btn-custom mb-3"),
                                dbc.Collapse(
                                    html.Div(id="rag-prompt-container"),
                                    id="collapse-rag-prompt",
                                    is_open=False,
                                )
                            ])
                        ]),
                    ])
                ]),
            ], id="tabs", active_tab="tab-overview"),

            dcc.Store(id='portfolio-store', data={'positions': _initial_positions}),
            dcc.Store(id='analysis-data', data={}),
            dcc.Store(id='rag-status', data={'indexed_tickers': [], 'doc_count': 0}),

        ], fluid=True, className="main-container")
    ])
])

# Register callbacks
from callbacks import register_all

register_all(app, rag_provider)

if __name__ == '__main__':
    app.run(debug=True, port=8050)
