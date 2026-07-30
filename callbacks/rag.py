import os
import logging
import threading
from datetime import datetime
import pandas as pd
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update

# Identischer Prompt-Builder wie im Naive-Tab — garantiert denselben Prompt.
from callbacks.naive_llm import build_portfolio_prompt, PROMPT_DEBUG_STYLE

logger = logging.getLogger(__name__)

MODEL = "llama-3.3-70b-versatile"
MAX_TOKENS = 4096

# Serialisiert JEDEN Zugriff auf Vektorspeicher/Cache (FAISS ist nicht thread-sicher).
# Auto-Fetch (Hintergrund) und manuelle Analyse dürfen den Store nicht gleichzeitig anfassen.
_rag_lock = threading.Lock()


def _single_stock_anomalies(analysis_data):
    """Einzeltitel-Anomalien (concentration == 'Hisseye özgü' mit verantwortlichem Ticker)."""
    return [
        b for b in (analysis_data or {}).get('active_return_breaks', [])
        if b.get('concentration') == 'Hisseye özgü' and b.get('responsible_ticker')
    ]


def _index_anomaly_news(pipeline, analysis_data):
    """Indiziert Nachrichten NUR für Einzeltitel-Anomalien: je Anomalie den verantwortlichen
    Ticker mit target_day = Anomaliedatum. Nur Nachricht im Fenster [Datum ± Fenster] landet
    im Vektorindex (siehe RAGPipeline.index_news_for_tickers)."""
    for b in _single_stock_anomalies(analysis_data):
        ticker = b.get('responsible_ticker')
        date_str = b.get('date')
        if not (ticker and date_str):
            continue
        try:
            day = datetime.fromisoformat(str(date_str)[:10])
        except ValueError:
            continue
        # Index-Fenster == Retrieval-Fenster (anomaly_window_days), damit indizierte und
        # später abgerufene Nachricht denselben Zeitraum abdecken.
        pipeline.index_news_for_tickers(
            [ticker], target_day=day,
            coverage_window_days=pipeline.config.anomaly_window_days)


def _background_index(rag_provider, analysis_data):
    """Hintergrund-Thread: lädt RAG (lazy) und indiziert Nachrichten NUR für Anomalie-Ticker
    im jeweiligen Anomaliedatum-Fenster. Mit der aktuellen RSS-Quelle liefert das für
    vergangene Anomalien (noch) keine Treffer; die Verdrahtung ist bereit, sobald eine
    datierte Quelle ergänzt wird."""
    if not _rag_lock.acquire(blocking=False):
        return  # ein Auto-Fetch läuft bereits
    try:
        pipeline = rag_provider.get()
        if pipeline:
            _index_anomaly_news(pipeline, analysis_data)
    except Exception as e:
        logger.warning(f"Auto-Fetch fehlgeschlagen: {e}")
    finally:
        _rag_lock.release()

# Anzeigesprache der Nachrichtentabelle (Debug)
NEWS_TABLE_HEADERS = ['#', 'Ticker', 'Datum', 'Titel', 'Quelle', 'Indexiert']


def _collect_rag_context(rag_pipeline, analysis_data):
    """Ruft Nachrichten-Kontext ab — NUR für Einzeltitel-Anomalien (Ticker + Zeitfenster).

    Kein Fallback und KEINE allgemeinen Nachrichten. Gibt es für eine Anomalie im Fenster
    keine passende Nachricht, bleibt der Kontext insoweit leer. Rückgabe: (kontext, chunks).
    """
    retrieved, seen = [], set()
    for b in _single_stock_anomalies(analysis_data):
        chunks = rag_pipeline.retrieve_for_anomaly(
            "reason for the stock price move, earnings, guidance or company news",
            b, top_k=3)
        for c in chunks:
            link = c.get('metadata', {}).get('link', '')
            key = link or c.get('text', '')[:80]
            if key not in seen:
                seen.add(key)
                retrieved.append(c)

    context = rag_pipeline.format_context_for_llm(retrieved, max_tokens=2000)
    return context, retrieved


def register(app, rag_provider):

    @app.callback(
        Output('rag-status', 'data'),
        Input('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def auto_fetch_news(analysis_data):
        """Automatischer Nachrichten-Abruf, sobald Portfolio + Anomalien berechnet sind.

        Läuft OHNE Klick auf 'Mit RAG analysieren'. Startet den Fetch in einem
        Hintergrund-Thread, damit die UI nicht blockiert wird (und RAG erst hier — nach
        App-Start, Tickern, Anomalien — lazy geladen wird). Der Cache verhindert
        wiederholtes Laden bei erneuten Auslösungen.
        """
        if not analysis_data or not _single_stock_anomalies(analysis_data):
            return no_update  # keine Einzeltitel-Anomalie → nichts abzurufen

        threading.Thread(
            target=_background_index,
            args=(rag_provider, analysis_data),
            daemon=True,
        ).start()
        return no_update

    @app.callback(
        Output('rag-llm-output', 'children'),
        [Input('btn-rag-analyze', 'n_clicks')],
        [State('analysis-data', 'data')],
        prevent_initial_call=True,
    )
    def analyze_with_rag(n_clicks, analysis_data):
        if not n_clicks:
            return ""
        # RAG wird hier beim ersten Klick geladen (lazy) — nicht beim App-Start.
        rag_pipeline = rag_provider.get()
        if not rag_pipeline:
            return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")
        if not analysis_data or not analysis_data.get('positions'):
            return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

        try:
            # 1+2) NUR anomalie-bezogene Nachrichten indizieren und abrufen — unter dem
            # Store-Lock (kein paralleler Zugriff mit dem Auto-Fetch-Thread).
            with _rag_lock:
                _index_anomaly_news(rag_pipeline, analysis_data)
                context, retrieved = _collect_rag_context(rag_pipeline, analysis_data)

            # 3) IDENTISCHER Prompt wie im Naive-Tab, plus abgerufener Nachrichten-Kontext.
            prompt = build_portfolio_prompt(analysis_data, news_context=context or None)

            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=MODEL,
                max_tokens=MAX_TOKENS,
            )
            llm_response = response.choices[0].message.content

            # Antwort
            components = [
                dbc.Card([dbc.CardBody([
                    html.H5("RAG-gestützte Analyse", className="mb-3", style={'color': '#f0f6fc'}),
                    html.Div(llm_response, style={
                        'whiteSpace': 'pre-wrap', 'lineHeight': '1.6', 'fontSize': '0.95rem',
                        'color': '#f0f6fc'
                    }),
                    html.Hr(),
                    html.Small(
                        f"Erstellt mit Groq ({MODEL}) · Gleicher Prompt wie Naive-LLM, zusätzlich "
                        f"{len(retrieved)} abgerufene Nachrichten-Snippets als Kontext.",
                        className="text-muted"
                    )
                ])], className="card-custom mt-3")
            ]

            # Quellen (Top-K)
            if retrieved:
                components.append(html.H5("📰 Verwendete Quellen", style={
                    'color': '#79c0ff', 'marginTop': '20px', 'marginBottom': '10px'}))
                for i, chunk in enumerate(retrieved, 1):
                    md = chunk.get('metadata', {})
                    not_dated = md.get('date_filtered') is False
                    components.append(dbc.Card([dbc.CardBody([
                        dbc.Row([
                            dbc.Col([html.H6(f"[{i}] {md.get('ticker', 'N/A')} - {md.get('title', 'N/A')[:60]}",
                                             style={'color': '#58a6ff'})], width=9),
                            dbc.Col([
                                html.Small(md.get('source', 'Unknown'), className="text-muted d-block"),
                                html.Small(md.get('published', '')[:10], className="text-muted")
                            ], width=3)
                        ]),
                        html.P(chunk.get('text', '')[:200] + "...",
                               className="text-muted small", style={'marginTop': '10px', 'marginBottom': '10px'}),
                        html.Div([
                            html.A("📌 Link zur Nachricht", href=md.get('link', '#'), target="_blank",
                                   className="small", style={'color': '#79c0ff', 'textDecoration': 'underline'}),
                            html.Span(" · Kontext nicht aus dem Anomaliefenster (Fallback)",
                                      className="small text-warning") if not_dated else None,
                        ])
                    ])], className="card-custom", style={'marginBottom': '10px'}))

            return html.Div(components)

        except Exception as e:
            logger.error(f"Error in RAG analysis: {e}")
            import traceback
            traceback.print_exc()
            return dbc.Alert([
                html.H5("Fehler bei der RAG-Analyse", className="mb-2"),
                html.P(f"Fehlerdetails: {str(e)}"),
            ], color="danger")

    @app.callback(
        Output('collapse-rag-prompt', 'is_open'),
        Output('rag-prompt-container', 'children'),
        Input('btn-toggle-rag-prompt', 'n_clicks'),
        State('collapse-rag-prompt', 'is_open'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def toggle_rag_prompt(n_clicks, is_open, analysis_data):
        """Zeigt den EXAKTEN Prompt, der an das LLM geht — inkl. abgerufenem Nachrichten-Kontext."""
        if is_open:
            return False, no_update  # Schließen: nicht neu berechnen/abrufen
        if not analysis_data or not analysis_data.get('positions'):
            return True, dbc.Alert("Keine Portfolio-Daten verfügbar.", color="warning")

        rag_pipeline = rag_provider.get()
        if not rag_pipeline:
            prompt = build_portfolio_prompt(analysis_data)
            return True, html.Pre(
                prompt + "\n\n[RAG nicht verfügbar — Nachrichten-Kontext fehlt]",
                style=PROMPT_DEBUG_STYLE)

        with _rag_lock:
            _index_anomaly_news(rag_pipeline, analysis_data)
            context, _ = _collect_rag_context(rag_pipeline, analysis_data)

        prompt = build_portfolio_prompt(analysis_data, news_context=context or None)
        return True, html.Pre(prompt, style=PROMPT_DEBUG_STYLE)

    @app.callback(
        Output('collapse-rag-news', 'is_open'),
        Output('rag-news-table-container', 'children'),
        Input('btn-rag-toggle-news', 'n_clicks'),
        State('collapse-rag-news', 'is_open'),
        prevent_initial_call=True,
    )
    def toggle_news_table(n_clicks, is_open):
        rag_pipeline = rag_provider.get()
        if not rag_pipeline:
            return (not is_open), dbc.Alert("RAG Pipeline nicht verfügbar", color="warning")

        articles = rag_pipeline.cache.all_articles(limit=200)
        if not articles:
            content = dbc.Alert("Noch keine Nachrichten im Cache. Klicke zuerst auf 'Mit RAG analysieren'.",
                                color="info")
        else:
            df = pd.DataFrame(articles)
            out = pd.DataFrame({
                'ticker': df['ticker'],
                'published': df['published'].apply(lambda x: str(x)[:10] if x else '—'),
                'title': df['title'].apply(lambda x: (x or '')[:80]),
                'source': df['source'],
                'indexed': df['indexed'].apply(lambda x: '✓' if x else '—'),
            })
            out.insert(0, '#', range(1, len(out) + 1))
            out.columns = NEWS_TABLE_HEADERS
            content = dbc.Table.from_dataframe(
                out, striped=True, bordered=True, hover=True, responsive=True, className="mt-2")

        return (not is_open), content
