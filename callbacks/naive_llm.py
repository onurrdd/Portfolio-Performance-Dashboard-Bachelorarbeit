import os
import pandas as pd
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html

# Antwortsprache des Modells: "de", "en" oder "tr" (Prompt selbst bleibt immer Englisch)
RESPONSE_LANGUAGE = "tr"

PROMPT_TEMPLATE = """Analyze these portfolio metrics and give me professional feedback:

PERFORMANCE METRICS:
- Total Return: {total_return:.2f}%
- Sortino Ratio: {sortino:.3f}
- Sharpe Ratio (current): {sharpe:.3f}

SP500 METRICS:
- Total Return: {sp500_total_return:.2f}%
- Sortino Ratio: {sp500_sortino:.3f}
- Sharpe Ratio (current): {sp500_sharpe:.3f}

Please evaluate these 3 metrics individually:
1. Total Return
2. Sortino Ratio
3. Sharpe Ratio (current)

Then explain why the portfolio's performance is 
better or worse than the S&P 500 benchmark shown above (e.g. sector/stock exposure, timing of purchases, diversification, or general market conditions).
"""

RESPONSE_LANGUAGE_INSTRUCTIONS = {
    "de": "\nPlease respond in German.",
    "tr": "\nPlease respond in Turkish.",
}

# Anzeigesprache der Anomalie-Tabelle (unabhängig von RESPONSE_LANGUAGE): "tr" oder "de"
TABLE_LANGUAGE = "de"

TABLE_HEADERS = {
    "tr": ['#', 'Tarih', 'Gerçekleşen', 'Beklenen', 'Sürpriz', 'Sürpriz (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Sınıf', 'Sorumlu Ticker', 'Uyarı'],
    "de": ['#', 'Datum', 'Tatsächlich', 'Erwartet', 'Überraschung', 'Überraschung (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Klasse', 'Verantwortlicher Ticker', 'Warnung'],
}

CLASSIFICATION_LABELS = {
    "tr": {"Portföye özgü": "Portföye özgü", "Kopma": "Kopma"},
    "de": {"Portföye özgü": "Portfolio-spezifisch", "Kopma": "Entkopplung"},
}

CONCENTRATION_LABELS = {
    "tr": {"Hisseye özgü": "Hisseye özgü", "Faktör/Sektör": "Faktör/Sektör"},
    "de": {"Hisseye özgü": "Einzeltitel", "Faktör/Sektör": "Faktor/Sektor"},
}

TABLE_TEXT = {
    "tr": {"no_breaks": "Anomali günü tespit edilmedi.", "not_attributable": "— (atfedilemez)", "dash": "—"},
    "de": {"no_breaks": "Kein Anomalietag festgestellt.", "not_attributable": "— (nicht zuordenbar)", "dash": "—"},
}


def register(app):
    @app.callback(
        Output('naive-llm-output', 'children'),
        [Input('btn-naive-llm-attribution', 'n_clicks')],
        [State('analysis-data', 'data')]
    )
    def analyze_portfolio_with_ai(n_clicks, analysis_data):
        if not n_clicks:
            return ""

        if not analysis_data or not analysis_data.get('positions'):
            return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            metrics = analysis_data.get('metrics', {})
            allocation = analysis_data.get('allocation', [])
            rolling_sharpe = analysis_data.get('rolling_sharpe', {})
            benchmark = analysis_data.get('benchmark', {})

            allocation_text = chr(10).join([f"- {a['Ticker']}: ${a['Value']:,.2f}" for a in allocation])
            prompt = PROMPT_TEMPLATE.format(
                total_return=metrics.get('total_return', 0),
                sortino=metrics.get('sortino', 0),
                sharpe=rolling_sharpe.get('current', 0),
                allocation=allocation_text,
                sp500_total_return=benchmark.get('total_return', 0),
                sp500_sortino=benchmark.get('sortino', 0),
                sp500_sharpe=benchmark.get('sharpe_current', 0)
            )
            if RESPONSE_LANGUAGE in RESPONSE_LANGUAGE_INSTRUCTIONS:
                prompt += RESPONSE_LANGUAGE_INSTRUCTIONS[RESPONSE_LANGUAGE]

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
            )
            analysis_text = response.choices[0].message.content

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
                html.Small(["Mögliche Ursachen:", html.Ul([
                    html.Li("GROQ_API_KEY nicht gesetzt"),
                    html.Li("Ungültiger API-Schlüssel"),
                    html.Li("Netzwerkprobleme"),
                    html.Li("API-Rate-Limit erreicht")
                ])])
            ], color="danger")

    @app.callback(
        Output('collapse-breaks', 'is_open'),
        Output('breaks-table-container', 'children'),
        Input('btn-toggle-breaks', 'n_clicks'),
        State('collapse-breaks', 'is_open'),
        State('analysis-data', 'data'),
        prevent_initial_call=True
    )
    def toggle_breaks_table(n_clicks, is_open, analysis_data):
        breaks = (analysis_data or {}).get('active_return_breaks', [])
        lang = TABLE_LANGUAGE if TABLE_LANGUAGE in TABLE_HEADERS else "tr"
        text = TABLE_TEXT[lang]
        class_labels = CLASSIFICATION_LABELS[lang]
        conc_labels = CONCENTRATION_LABELS[lang]

        if not breaks:
            content = dbc.Alert(text["no_breaks"], color="info")
        else:
            df = pd.DataFrame(breaks)

            def _ticker_cell(row):
                if not row['responsible_ticker']:
                    return text["not_attributable"]
                conc = row.get('concentration')
                conc_label = f" · {conc_labels.get(conc, conc)}" if conc else ""
                return f"{row['responsible_ticker']} ({row['ticker_contribution_pct']:+.2f}%){conc_label}"

            out = pd.DataFrame({
                'date': df['date'],
                'actual': df['actual_return_pct'].apply(lambda x: f"{x:+.2f}%"),
                'expected': df['expected_return_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else text["dash"]),
                'surprise': df['surprise_pct'].apply(lambda x: f"{x:+.2f}%"),
                'surprise_z': df['surprise_mad_z'].apply(lambda x: f"{x:+.2f}"),
                'benchmark_z': df['benchmark_mad_z'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else text["dash"]),
                'beta': df['beta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else text["dash"]),
                'classification': df['classification'].apply(lambda c: class_labels.get(c, c)),
                'ticker': df.apply(_ticker_cell, axis=1),
                'flags': df['flags'].apply(lambda x: x if x else text["dash"]),
            })
            out.insert(0, '#', range(1, len(out) + 1))
            out.columns = TABLE_HEADERS[lang]
            content = dbc.Table.from_dataframe(
                out, striped=True, bordered=True, hover=True, responsive=True, className="mt-2"
            )

        return not is_open, content
