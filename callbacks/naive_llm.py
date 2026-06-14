import os
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, html

# ── Buraya istediğin prompt'u yaz ──────────────────────────────────────────
PERFORMANCE_ATTRIBUTION_PROMPT = """x"""
# ───────────────────────────────────────────────────────────────────────────


def register(app):
    @app.callback(
        Output('naive-llm-output', 'children'),
        Input('btn-naive-llm-attribution', 'n_clicks'),
        prevent_initial_call=True
    )
    def run_performance_attribution(n_clicks):
        try:
            client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
            response = client.chat.completions.create(
                messages=[{"role": "user", "content": PERFORMANCE_ATTRIBUTION_PROMPT}],
                model="llama-3.3-70b-versatile",
                max_tokens=2048,
            )
            result_text = response.choices[0].message.content

            return dbc.Card([
                dbc.CardBody([
                    html.H5("Performance Attribution Analyse", className="mb-3", style={'color': '#f0f6fc'}),
                    html.Div(result_text, style={
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
                html.H5("Fehler", className="mb-2"),
                html.P(f"Fehlerdetails: {str(e)}"),
            ], color="danger")
