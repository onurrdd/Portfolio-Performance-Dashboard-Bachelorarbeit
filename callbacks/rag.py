import os
import logging
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html

from prompts import build_advisor_prompt, performance_info_from_analysis_data

logger = logging.getLogger(__name__)


def register(app, rag_pipeline):
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
            tickers = [t.strip().upper() for t in tickers_str.split(',')]
            limit = int(limit) if limit else 5

            stats = rag_pipeline.index_news_for_tickers(tickers, news_limit=limit)
            db_stats = rag_pipeline.get_stats()

            status_msg = dbc.Alert([
                html.H5("✅ Nachrichten erfolgreich indiziert!", className="mb-2"),
                dbc.Row([
                    dbc.Col([html.P(f"Ticker verarbeitet: {stats['tickers_successful']} / {stats['tickers_processed']}")], width=6),
                    dbc.Col([html.P(f"Artikel: {stats['total_articles']}")], width=6)
                ]),
                dbc.Row([
                    dbc.Col([html.P(f"Chunks erstellt: {stats['total_chunks']}")], width=6),
                    dbc.Col([html.P(f"Gesamt indexiert: {db_stats['indexed_documents']} Dokumente")], width=6)
                ]),
                html.Hr(),
                html.Small(f"Zeitstempel: {stats['timestamp']}")
            ], color="success", className="mb-3")

            return status_msg, {
                'indexed_tickers': tickers,
                'doc_count': db_stats['indexed_documents'],
                'last_index_time': stats['timestamp']
            }

        except Exception as e:
            logger.error(f"Error indexing news: {e}")
            return dbc.Alert([
                html.H5("❌ Fehler beim Indizieren"),
                html.P(f"Fehler: {str(e)}")
            ], color="danger"), {}

    @app.callback(
        Output('rag-results-output', 'children'),
        [Input('btn-rag-query', 'n_clicks')],
        [State('rag-query-input', 'value'),
         State('rag-topk-input', 'value'),
         State('rag-status', 'data'),
         State('analysis-data', 'data')],
        prevent_initial_call=True
    )
    def query_with_rag(n_clicks, query, topk, rag_status, analysis_data):
        if not n_clicks or not rag_pipeline:
            return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")
        if not query:
            return dbc.Alert("Bitte eine Frage eingeben", color="warning")
        if rag_status.get('doc_count', 0) == 0:
            return dbc.Alert("Keine Nachrichten indexiert. Bitte zuerst Nachrichten abrufen.", color="info")

        try:
            topk = int(topk) if topk else 5
            retrieved = rag_pipeline.retrieve_context(query, top_k=topk)

            if not retrieved:
                return dbc.Alert("Keine relevanten Nachrichten gefunden", color="info")

            context = rag_pipeline.format_context_for_llm(retrieved, max_tokens=2000)

            performance_info = performance_info_from_analysis_data(analysis_data)
            prompt = build_advisor_prompt(performance_info=performance_info, news_context=context, user_question=query)

            try:
                client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
                response = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model="llama-3.3-70b-versatile",
                    max_tokens=2048,
                )
                llm_response = response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq API error: {e}")
                llm_response = f"Groq API nicht verfügbar. Hier sind die gefundenen Nachrichten:\n\n{context}"

            result_components = [
                html.H5("🤖 RAG-gestützte Antwort", style={'color': '#58a6ff', 'marginTop': '20px', 'marginBottom': '10px'}),
                html.Div(llm_response, style={
                    'whiteSpace': 'pre-wrap', 'lineHeight': '1.6', 'fontSize': '0.95rem',
                    'color': '#f0f6fc', 'marginBottom': '20px', 'padding': '15px',
                    'backgroundColor': '#0d1117', 'borderLeft': '3px solid #58a6ff', 'borderRadius': '4px'
                }),
                html.Hr(),
                html.H5("📰 Quelle (Top-K Nachrichten)", style={'color': '#79c0ff', 'marginBottom': '10px'}),
            ]

            for i, chunk in enumerate(retrieved, 1):
                metadata = chunk.get('metadata', {})
                result_components.append(
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([html.H6(f"[{i}] {metadata.get('ticker', 'N/A')} - {metadata.get('title', 'N/A')[:60]}",
                                                style={'color': '#58a6ff'})], width=9),
                                dbc.Col([html.Small(metadata.get('source', 'Unknown'), className="text-muted")], width=3)
                            ]),
                            html.P(chunk.get('text', '')[:200] + "...",
                                  className="text-muted small", style={'marginTop': '10px', 'marginBottom': '10px'}),
                            html.A("📌 Link zur Nachricht", href=metadata.get('link', '#'), target="_blank",
                                  className="small", style={'color': '#79c0ff', 'textDecoration': 'underline'})
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
