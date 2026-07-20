import os
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
