import os
import pandas as pd
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update

# Sprache der LLM-ANTWORT: "de", "en" oder "tr".
# Die Prompts selbst sind IMMER Englisch (Projektregel, siehe CLAUDE.md).
RESPONSE_LANGUAGE = "en"

# Flag: Teil 1 (Performance vs. Benchmark) temporär deaktiviert — Forschungsfrage fokussiert
# jetzt auf die kausale Erklärung von Anomalie-Ereignissen (Teil 2), nicht mehr auf die
# Interpretation von Kennzahlen. Code bewusst NICHT gelöscht, um Teil 1 bei Bedarf wieder
# zu aktivieren (True setzen).
INCLUDE_METRICS_PART = False

PART1_TEMPLATE = """=== PART 1: PERFORMANCE VS. BENCHMARK ===

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

Then explain why the portfolio's performance is better or worse than the S&P 500
benchmark shown above (e.g. sector/stock exposure, timing of purchases,
diversification, or general market conditions).

"""

PART2_TEMPLATE = """=== PART 2: SINGLE-STOCK ANOMALY EVENTS ===

Below is a list of single-stock anomaly events detected in this portfolio. Each event is a case where a specific stock showed an unusually large price move on a specific date — large both relative to its own historical volatility and relative to how the market moved that day.

ANOMALY EVENTS:
{anomaly_list}

For each event listed above, explain the reason for this price move on that
specific date.
Address every event individually; do not summarize them collectively.

For each explanation, also state what you are basing it on (e.g. a specific news report,
earnings release, or your own general knowledge of the event) — name the source if you are
citing one.

"""
# (or the previous trading day, since news can affect prices with a one-day lag)
# Give a specific, concrete, and definitive answer for each event. Do NOT give generic or
# vague explanations. Commit to a direct answer.
if INCLUDE_METRICS_PART:
    PROMPT_TEMPLATE = (
        'Analyze the portfolio below and give me professional feedback.\n'
        'Answer the following two parts separately, using "PART 1" and "PART 2" as headings.\n\n'
        + PART1_TEMPLATE + PART2_TEMPLATE
    )
else:
    PROMPT_TEMPLATE = (
        'Analyze the portfolio below and give me professional feedback.\n\n'
        + PART2_TEMPLATE
    )

# "en" fehlt bewusst: der Prompt ist bereits Englisch, eine Extra-Anweisung wäre redundant.
RESPONSE_LANGUAGE_INSTRUCTIONS = {
    "de": "\nPlease respond in German.",
    "tr": "\nPlease respond in Turkish.",
}

# Flag: True = Sicherheitsanweisung wird angehängt (kein Raten, "unbekannt" sagen).
#       False = naive/ungeschützter Prompt (zeigt Halluzinationsrisiko ohne RAG).
ANOMALY_PROMPT_SAFE_MODE = False

NO_ANOMALY_PLACEHOLDER = "(No single-stock anomaly events were detected for this portfolio.)"

PROMPT_ANOMALY_SAFETY_INSTRUCTION = """
Answer ONLY if you have specific, verifiable knowledge of a real, documented reason (e.g. an earnings report, guidance change, product announcement, analyst rating action, regulatory or legal news, M&A news, or a specific macroeconomic event) known to have affected this exact ticker on this exact date. If you are not confident about the specific cause for a given event, explicitly write "No reliable specific cause known for this date" instead of guessing. Do not invent a plausible-sounding explanation, and do not substitute general market or sector commentary for a specific reason you are not sure about."""

# Anzeigesprache der Anomalie-Tabelle (unabhängig von RESPONSE_LANGUAGE): "tr" oder "de"
TABLE_LANGUAGE = "de"

TABLE_HEADERS = {
    "tr": ['#', 'Tarih', 'Gerçekleşen', 'Beklenen', 'Sürpriz', 'Sürpriz (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Sorumlu Ticker', 'Hisse Getirisi',
           'Hisse Sürprizi', 'Hisse (MAD-z)', 'Uyarı'],
    "de": ['#', 'Datum', 'Tatsächlich', 'Erwartet', 'Überraschung', 'Überraschung (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Verantwortlicher Ticker', 'Titel-Rendite',
           'Titel-Residuum', 'Titel (MAD-z)', 'Warnung'],
}

CONCENTRATION_LABELS = {
    "tr": {"Hisseye özgü": "Hisseye özgü", "Dağılmış": "Dağılmış"},
    "de": {"Hisseye özgü": "Einzeltitel", "Dağılmış": "Verteilt"},
}

TABLE_TEXT = {
    "tr": {"no_breaks": "Anomali günü tespit edilmedi.", "not_attributable": "— (atfedilemez)", "dash": "—"},
    "de": {"no_breaks": "Kein Anomalietag festgestellt.", "not_attributable": "— (nicht zuordenbar)", "dash": "—"},
}


def _build_anomaly_list_text(analysis_data):
    """Baut die Anomalie-Liste (Teil 2 des Prompts) aus den einzeltitelbezogenen Ereignissen."""
    breaks = (analysis_data or {}).get('active_return_breaks', [])
    single_stock = [
        b for b in breaks
        if b.get('concentration') == 'Hisseye özgü' and b.get('responsible_ticker')
    ]
    if not single_stock:
        return NO_ANOMALY_PLACEHOLDER
    lines = []
    for b in single_stock:
        own_return = b.get('ticker_own_return_pct', 0) or 0
        direction = "rose" if own_return >= 0 else "fell"
        lines.append(
            f"- {b['date']}: {b['responsible_ticker']} {direction} {own_return:+.2f}%"
        )
    return chr(10).join(lines)


def build_portfolio_prompt(analysis_data, news_context=None):
    """Baut den Portfolio-Analyse-Prompt.

    Wird von BEIDEN LLM-Tabs genutzt, damit der Prompt identisch ist. Der einzige
    Unterschied zwischen Naive-LLM und RAG-LLM ist der optionale `news_context`, der
    beim RAG-Tab als abgerufener Nachrichten-Kontext angehängt wird — so lässt sich der
    Effekt des Retrievals isoliert beobachten.
    """
    metrics = (analysis_data or {}).get('metrics', {})
    rolling_sharpe = (analysis_data or {}).get('rolling_sharpe', {})
    benchmark = (analysis_data or {}).get('benchmark', {})

    prompt = PROMPT_TEMPLATE.format(
        total_return=metrics.get('total_return', 0),
        sortino=metrics.get('sortino', 0),
        sharpe=rolling_sharpe.get('current', 0),
        sp500_total_return=benchmark.get('total_return', 0),
        sp500_sortino=benchmark.get('sortino', 0),
        sp500_sharpe=benchmark.get('sharpe_current', 0),
        anomaly_list=_build_anomaly_list_text(analysis_data),
    )
    if news_context:
        prompt += (
            "\n\n=== RETRIEVED NEWS CONTEXT (RAG) ===\n"
            "The following news items were retrieved for the anomaly tickers and dates. "
            "Only use a retrieved item if it is genuinely relevant to explaining a specific "
            "event. If none of the retrieved items are relevant to a given event, say so "
            "explicitly instead of forcing a connection.\n"
            f"{news_context}"
        )
    if ANOMALY_PROMPT_SAFE_MODE:
        prompt += PROMPT_ANOMALY_SAFETY_INSTRUCTION
    if RESPONSE_LANGUAGE in RESPONSE_LANGUAGE_INSTRUCTIONS:
        prompt += RESPONSE_LANGUAGE_INSTRUCTIONS[RESPONSE_LANGUAGE]
    return prompt


# Anzeigestil für den Debug-Prompt (aufklappbar). Wird auch vom RAG-Tab genutzt.
PROMPT_DEBUG_STYLE = {
    'whiteSpace': 'pre-wrap', 'fontFamily': 'monospace', 'fontSize': '0.8rem',
    'color': '#c9d1d9', 'backgroundColor': '#0d1117', 'padding': '15px',
    'border': '1px solid #30363d', 'borderRadius': '6px',
    'maxHeight': '500px', 'overflowY': 'auto',
}


def _run_naive_analysis(analysis_data):
    """Führt die Naive-LLM-Analyse aus und baut die Ergebniskarte.

    Gemeinsame Logik für den Naive-LLM-Tab UND den Vergleichs-Tab (ein Trigger-Button
    löst dort beide Analysen aus) — identischer Code, ein einziger Ort zum Pflegen.
    """
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

    try:
        client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        # Identischer Prompt wie im RAG-Tab (build_portfolio_prompt) — hier ohne
        # Nachrichten-Kontext (naiv). So ist der einzige Unterschied das Retrieval.
        prompt = build_portfolio_prompt(analysis_data)

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            max_tokens=4096,
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
                html.Small(
                    "Erstellt mit Groq · Hinweis: Dieses Modell hat keinen Zugriff auf aktuelle "
                    "Nachrichten (kein RAG) — die Antworten zu den Anomalietagen basieren "
                    "ausschließlich auf dem allgemeinen Trainingswissen des Modells und sind "
                    "keine verifizierte Quelle.",
                    className="text-muted"
                )
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


def _naive_prompt_component(analysis_data):
    """Baut die Debug-Anzeige des exakten Naive-LLM-Prompts (ohne Nachrichten-Kontext).

    Gemeinsame Logik für den 'Prompt anzeigen'-Button im Naive-LLM-Tab UND im
    Vergleichs-Tab.
    """
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar.", color="warning")
    prompt = build_portfolio_prompt(analysis_data)
    return html.Pre(prompt, style=PROMPT_DEBUG_STYLE)


def register(app):
    @app.callback(
        Output('naive-llm-output', 'children'),
        [Input('btn-naive-llm-attribution', 'n_clicks')],
        [State('analysis-data', 'data')]
    )
    def analyze_portfolio_with_ai(n_clicks, analysis_data):
        if not n_clicks:
            return ""
        return _run_naive_analysis(analysis_data)

    @app.callback(
        Output('compare-naive-output', 'children'),
        Input('btn-compare-analyze', 'n_clicks'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def analyze_naive_compare(n_clicks, analysis_data):
        """Vergleichs-Tab, linke Seite — identische Logik wie der Naive-LLM-Tab,
        ausgelöst vom gemeinsamen 'btn-compare-analyze'-Button."""
        if not n_clicks:
            return ""
        return _run_naive_analysis(analysis_data)

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
        conc_labels = CONCENTRATION_LABELS[lang]

        if not breaks:
            content = dbc.Alert(text["no_breaks"], color="info")
        else:
            df = pd.DataFrame(breaks)

            def _ticker_cell(row):
                if pd.isna(row['responsible_ticker']):
                    return text["not_attributable"]
                conc = row.get('concentration')
                conc_label = f" · {conc_labels.get(conc, conc)}" if pd.notna(conc) else ""
                return f"{row['responsible_ticker']} ({row['ticker_contribution_pct']:+.2f}%){conc_label}"

            out = pd.DataFrame({
                'date': df['date'],
                'actual': df['actual_return_pct'].apply(lambda x: f"{x:+.2f}%"),
                'expected': df['expected_return_pct'].apply(lambda x: f"{x:+.2f}%" if pd.notna(x) else text["dash"]),
                'surprise': df['surprise_pct'].apply(lambda x: f"{x:+.2f}%"),
                'surprise_z': df['surprise_mad_z'].apply(lambda x: f"{x:+.2f}"),
                'benchmark_z': df['benchmark_mad_z'].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else text["dash"]),
                'beta': df['beta'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else text["dash"]),
                'ticker': df.apply(_ticker_cell, axis=1),
                'own_return': df['ticker_own_return_pct'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else text["dash"]),
                'own_residual': df['ticker_own_residual_pct'].apply(
                    lambda x: f"{x:+.2f}%" if pd.notna(x) else text["dash"]),
                'own_z': df['ticker_own_mad_z'].apply(
                    lambda x: f"{x:+.2f}" if pd.notna(x) else text["dash"]),
                'flags': df['flags'].apply(lambda x: x if x else text["dash"]),
            })
            out.insert(0, '#', range(1, len(out) + 1))
            out.columns = TABLE_HEADERS[lang]
            content = dbc.Table.from_dataframe(
                out, striped=True, bordered=True, hover=True, responsive=True, className="mt-2"
            )

        return not is_open, content

    @app.callback(
        Output('collapse-naive-prompt', 'is_open'),
        Output('naive-prompt-container', 'children'),
        Input('btn-toggle-naive-prompt', 'n_clicks'),
        State('collapse-naive-prompt', 'is_open'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def toggle_naive_prompt(n_clicks, is_open, analysis_data):
        """Zeigt den EXAKTEN Prompt, der an das LLM geht (ohne Nachrichten-Kontext)."""
        if is_open:
            return False, no_update  # Schließen: nicht neu berechnen
        return True, _naive_prompt_component(analysis_data)

    @app.callback(
        Output('collapse-compare-naive-prompt', 'is_open'),
        Output('compare-naive-prompt-container', 'children'),
        Input('btn-toggle-compare-naive-prompt', 'n_clicks'),
        State('collapse-compare-naive-prompt', 'is_open'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def toggle_compare_naive_prompt(n_clicks, is_open, analysis_data):
        """Vergleichs-Tab, linke Seite — gleicher 'Prompt anzeigen' wie im Naive-LLM-Tab."""
        if is_open:
            return False, no_update
        return True, _naive_prompt_component(analysis_data)
