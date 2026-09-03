import os
from openai import OpenAI
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def register(app):
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
            client = OpenAI(base_url=_OPENROUTER_BASE_URL,
                            api_key=os.environ.get("OPENROUTER_API_KEY"))
            metrics = analysis_data.get('metrics', {})
            allocation = analysis_data.get('allocation', [])
            rolling_sharpe = analysis_data.get('rolling_sharpe', {})
            correlation_stats = analysis_data.get('correlation_stats', {})

            prompt = f"""Analysiere diese Portfolio-Metriken und gib mir professionelles Feedback:

PERFORMANCE METRIKEN:
- Sortino Ratio: {metrics.get('sortino', 0):.3f}
- Calmar Ratio: {metrics.get('calmar', 0):.3f}
- Max Drawdown: {metrics.get('max_drawdown', 0):.2f}%
- Volatilität (annualisiert): {metrics.get('volatility', 0):.2f}%
- Gesamt Return: {metrics.get('total_return', 0):.2f}%
- Annualisierter Return: {metrics.get('annualized_return', 0):.2f}%

ROLLING SHARPE RATIO (120-Tage):
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
"""

            response = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-120b",
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
                    html.Small("Erstellt mit OpenRouter", className="text-muted")
                ])
            ], className="card-custom mt-3")

        except Exception as e:
            return dbc.Alert([
                html.H5("Fehler bei der AI-Analyse", className="mb-2"),
                html.P(f"Fehlerdetails: {str(e)}"),
                html.Hr(),
                html.Small(["Mögliche Ursachen:", html.Ul([
                    html.Li("OPENROUTER_API_KEY nicht gesetzt"),
                    html.Li("Ungültiger API-Schlüssel"),
                    html.Li("Netzwerkprobleme"),
                    html.Li("API-Rate-Limit erreicht")
                ])])
            ], color="danger")
