import os
import logging
import threading
from datetime import datetime
import pandas as pd
from groq import Groq
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update, ALL, ctx

from rag.cache import NewsCache
from rag.config import DEFAULT_CONFIG

# Identischer Prompt-Builder UND identisches Modell wie im Naive-Tab — ein einziger
# Ort (GENERATOR_MODEL) verhindert ein stilles Auseinanderlaufen der beiden Modelle
# (siehe Vorfall: Groq hat llama-3.3-70b-versatile zwischenzeitlich entfernt).
from callbacks.naive_llm import (build_portfolio_prompt, PROMPT_DEBUG_STYLE,
                                 GENERATOR_MODEL, GENERATION_MAX_TOKENS)

logger = logging.getLogger(__name__)

MODEL = GENERATOR_MODEL
# Antwortbudget — aus callbacks/naive_llm.py übernommen, damit beide Bedingungen
# zwangsläufig denselben Wert benutzen (ein zweiter Literalwert könnte auseinanderlaufen
# und den Vergleich unbemerkt entwerten).
MAX_TOKENS = GENERATION_MAX_TOKENS

# Serialisiert JEDEN Zugriff auf Vektorspeicher/Cache (FAISS ist nicht thread-sicher).
# Auto-Fetch (Hintergrund) und manuelle Analyse dürfen den Store nicht gleichzeitig anfassen.
_rag_lock = threading.Lock()


def _single_stock_anomalies(analysis_data):
    """Einzeltitel-Anomalien (concentration == 'Hisseye özgü' mit verantwortlichem Ticker)."""
    return [
        b for b in (analysis_data or {}).get('active_return_breaks', [])
        if b.get('concentration') == 'Hisseye özgü' and b.get('responsible_ticker')
    ]


# Quellen ohne Tageskontingent — nur diese laufen im automatischen Hintergrund-Abruf.
# Kontingentierte Quellen (Alpha Vantage: 25 Anfragen/Tag) bleiben dem manuell
# ausgelösten Abruf vorbehalten, damit ein bloßer Portfolio-Wechsel das Tageskontingent
# nicht unbemerkt aufbraucht.
AUTO_FETCH_SOURCES = {"sec_edgar"}


def _index_anomaly_news(pipeline, analysis_data, only_sources=None):
    """Indiziert Nachrichten NUR für Einzeltitel-Anomalien: je Anomalie den verantwortlichen
    Ticker mit target_day = Anomaliedatum. Nur Nachricht im Fenster [Datum ± Fenster] landet
    im Vektorindex (siehe RAGPipeline.index_news_for_tickers).

    `only_sources`: Beschränkt den Abruf auf bestimmte Quellen (siehe AUTO_FETCH_SOURCES)."""
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
            coverage_window_days=pipeline.config.anomaly_window_days,
            only_sources=only_sources)


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
            # NUR kontingentfreie Quellen: der automatische Abruf läuft bei jedem
            # Portfolio-Wechsel und darf das Tageskontingent nicht aufbrauchen.
            _index_anomaly_news(pipeline, analysis_data,
                                only_sources=AUTO_FETCH_SOURCES)
    except Exception as e:
        logger.warning(f"Auto-Fetch fehlgeschlagen: {e}")
    finally:
        _rag_lock.release()

# Anzeigesprache der Nachrichtentabelle (Debug)
NEWS_TABLE_HEADERS = ['#', 'Ticker', 'Datum', 'Titel', 'Quelle', 'Indexiert']


# Kontextbudget für den GEPOOLTEN Prompt (alle Anomalien in einem Aufruf).
# Bemessung: ein Chunk belegt im Kontextblock im Mittel ~830 Zeichen (gemessen am realen
# Index), 25000 Token ≈ 100.000 Zeichen ≈ 120 Chunks — genug, damit auch bei ~100 Anomalien
# jede mindestens einen Treffer beisteuert. Der Prompt bleibt damit bei ~26.000 Token, also
# rund einem Fünftel des Kontextfensters.
# NICHT zu verwechseln mit dem Budget in rag/evaluation.py: dort wird je Anomalie ein
# eigener Aufruf gebaut, ein kleines Budget genügt dort.
POOLED_CONTEXT_TOKEN_BUDGET = 25000
RETRIEVAL_TOP_K = 3


def _collect_rag_context(rag_pipeline, analysis_data):
    """Ruft Nachrichten-Kontext ab — NUR für Einzeltitel-Anomalien (Ticker + Zeitfenster).

    Kein Fallback und KEINE allgemeinen Nachrichten. Gibt es für eine Anomalie im Fenster
    keine passende Nachricht, bleibt der Kontext insoweit leer.

    Reihum-Verteilung (Round Robin): Die Chunks werden NICHT anomalieweise aneinandergehängt,
    sondern rangweise — erst der beste Treffer jeder Anomalie, dann der zweitbeste, dann der
    dritte. Grund: Der Kontextblock wird am Budget abgeschnitten. Bei anomalieweiser Reihung
    verbrauchen die ersten Anomalien ihre drei Treffer, bevor spätere überhaupt einen
    bekommen — diese fielen dann vollständig aus dem Kontext und die RAG-Bedingung wäre für
    sie stillschweigend identisch mit der Naive-Bedingung. Reihum verliert stattdessen jede
    Anomalie zuerst ihren *drittbesten* Treffer; die Abdeckung bleibt gleichmäßig.

    Rückgabe: (kontext, chunks_im_kontext, anzahl_verworfen).
    """
    per_anomaly = [
        rag_pipeline.retrieve_for_anomaly(
            "reason for the stock price move, earnings, guidance or company news",
            b, top_k=RETRIEVAL_TOP_K)
        for b in _single_stock_anomalies(analysis_data)
    ]

    retrieved, seen = [], set()
    for rank in range(RETRIEVAL_TOP_K):
        for chunks in per_anomaly:
            if rank >= len(chunks):
                continue
            c = chunks[rank]
            link = c.get('metadata', {}).get('link', '')
            key = link or c.get('text', '')[:80]
            if key not in seen:
                seen.add(key)
                retrieved.append(c)

    context, used = rag_pipeline.format_context_for_llm(
        retrieved, max_tokens=POOLED_CONTEXT_TOKEN_BUDGET, return_used=True)
    return context, used, len(retrieved) - len(used)


def _run_rag_analysis(rag_pipeline, analysis_data):
    """Führt die RAG-Analyse aus und baut Ergebniskarte + Quellenliste.

    Gemeinsame Logik für den RAG-Tab UND den Vergleichs-Tab (ein Trigger-Button löst
    dort beide Analysen aus) — identischer Code, ein einziger Ort zum Pflegen.
    """
    if not rag_pipeline:
        return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

    try:
        # 1+2) NUR anomalie-bezogene Nachrichten indizieren und abrufen — unter dem
        # Store-Lock (kein paralleler Zugriff mit dem Auto-Fetch-Thread).
        with _rag_lock:
            _index_anomaly_news(rag_pipeline, analysis_data)
            context, retrieved, dropped = _collect_rag_context(rag_pipeline, analysis_data)

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
                    f"{len(retrieved)} abgerufene Nachrichten-Snippets als Kontext."
                    # Budget-Kürzung sichtbar machen: sonst würde eine Antwort mit
                    # unvollständigem Kontext wie eine mit vollständigem aussehen.
                    + (f" · {dropped} weitere(s) Snippet(s) wegen Kontextbudget nicht "
                       f"übergeben." if dropped else ""),
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


def _rag_prompt_component(rag_provider, analysis_data):
    """Baut die Debug-Anzeige des exakten RAG-Prompts (inkl. abgerufenem Nachrichten-
    Kontext). Gemeinsame Logik für den 'Prompt anzeigen'-Button im RAG-Tab UND im
    Vergleichs-Tab.
    """
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar.", color="warning")

    rag_pipeline = rag_provider.get()
    if not rag_pipeline:
        prompt = build_portfolio_prompt(analysis_data)
        return html.Pre(
            prompt + "\n\n[RAG nicht verfügbar — Nachrichten-Kontext fehlt]",
            style=PROMPT_DEBUG_STYLE)

    with _rag_lock:
        _index_anomaly_news(rag_pipeline, analysis_data)
        context, _, _ = _collect_rag_context(rag_pipeline, analysis_data)

    prompt = build_portfolio_prompt(analysis_data, news_context=context or None)
    return html.Pre(prompt, style=PROMPT_DEBUG_STYLE)


EVAL_METRIC_LABELS = {
    "faithfulness": "Faithfulness (Quellentreue)",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall (Ground Truth)",
}


def _fmt_metric(v):
    return "—" if v is None else f"{v:.2f}"


def _fmt_pct(v):
    return "—" if v is None else f"{v:.1f}%"


def _comparison_table(comparison):
    """Naive-LLM-vs-RAG-LLM-Vergleich (Spezifität, Zitierbarkeit) — siehe
    About_Thesis.md Forschungsfrage-Teilfragen 1-4 und rag/evaluation.py::comparison.
    Kein direktes RAGAS-Metrik-Objekt; separate Hakem-Klassifikation, siehe dort.
    """
    spec, cite, faith = (
        comparison["specificity_pct"], comparison["named_citation_pct"],
        comparison["rag_faithfulness_as_belegbarkeit"],
    )
    rows = [
        {
            "Metrik": "Konkrete Ursache genannt (Spezifität)",
            "Naive — Alle": _fmt_pct(spec["naive"]["all"]),
            "Naive — Vor Cutoff": _fmt_pct(spec["naive"]["pre_cutoff"]),
            "Naive — Nach Cutoff": _fmt_pct(spec["naive"]["post_cutoff"]),
            "RAG — Alle": _fmt_pct(spec["rag"]["all"]),
            "RAG — Vor Cutoff": _fmt_pct(spec["rag"]["pre_cutoff"]),
            "RAG — Nach Cutoff": _fmt_pct(spec["rag"]["post_cutoff"]),
        },
        {
            "Metrik": "  ↳ RAG nur bei gefundenem Kontext",
            "Naive — Alle": "—", "Naive — Vor Cutoff": "—", "Naive — Nach Cutoff": "—",
            "RAG — Alle": _fmt_pct(spec["rag"]["context_available_only"]),
            "RAG — Vor Cutoff": "—", "RAG — Nach Cutoff": "—",
        },
        {
            "Metrik": "Benannte Quelle (Zitierbarkeit)",
            "Naive — Alle": _fmt_pct(cite["naive"]["all"]),
            "Naive — Vor Cutoff": _fmt_pct(cite["naive"]["pre_cutoff"]),
            "Naive — Nach Cutoff": _fmt_pct(cite["naive"]["post_cutoff"]),
            "RAG — Alle": _fmt_pct(cite["rag"]["all"]),
            "RAG — Vor Cutoff": _fmt_pct(cite["rag"]["pre_cutoff"]),
            "RAG — Nach Cutoff": _fmt_pct(cite["rag"]["post_cutoff"]),
        },
        {
            "Metrik": "Faithfulness (Belegbarkeit-Proxy, nur RAG möglich)",
            "Naive — Alle": "—", "Naive — Vor Cutoff": "—", "Naive — Nach Cutoff": "—",
            "RAG — Alle": _fmt_metric(faith["all"]),
            "RAG — Vor Cutoff": _fmt_metric(faith["pre_cutoff"]),
            "RAG — Nach Cutoff": _fmt_metric(faith["post_cutoff"]),
        },
    ]
    return dbc.Table.from_dataframe(
        pd.DataFrame(rows), striped=True, bordered=True, hover=True, responsive=True, className="mt-2")


def _run_ragas_eval(rag_provider, analysis_data):
    """Führt die RAGAS-Evaluation (Faz 4) aus und rendert Tabelle + Aggregat.

    Referenzfreie Metriken je Einzeltitel-Anomalie (Faithfulness, Answer Relevancy,
    Context Precision); Context Recall zusätzlich NUR für Anomalien mit Eintrag in
    data/ground_truth.json (siehe rag/evaluation.py). Läuft unter demselben Lock wie
    die reguläre RAG-Analyse (kein paralleler FAISS-Zugriff).
    """
    if not os.environ.get("GROQ_API_KEY"):
        return dbc.Alert("GROQ_API_KEY nicht gesetzt.", color="warning")
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")
    if not _single_stock_anomalies(analysis_data):
        return dbc.Alert("Keine Einzeltitel-Anomalien — nichts zu bewerten.", color="info")

    rag_pipeline = rag_provider.get()
    if not rag_pipeline:
        return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")

    try:
        from rag.evaluation import run_ragas_evaluation, save_eval_results
    except ImportError:
        return dbc.Alert("ragas ist nicht installiert (pip install ragas==0.4.3).", color="warning")

    try:
        with _rag_lock:
            _index_anomaly_news(rag_pipeline, analysis_data)
            result = run_ragas_evaluation(rag_pipeline, analysis_data, os.environ.get("GROQ_API_KEY"))
        path = save_eval_results(result)

        rows = []
        for i, s in enumerate(result["samples"], 1):
            cutoff_label = "—" if s["post_cutoff"] is None else ("ja" if s["post_cutoff"] else "nein")
            if s["status"] != "ok":
                rows.append({
                    '#': i, 'Datum': s['date'], 'Ticker': s['ticker'],
                    'Faithfulness': "kein Kontext abgerufen", 'Answer Relevancy': "—",
                    'Context Precision': "—", 'Nach Cutoff?': cutoff_label,
                })
            else:
                rows.append({
                    '#': i, 'Datum': s['date'], 'Ticker': s['ticker'],
                    'Faithfulness': _fmt_metric(s['faithfulness']),
                    'Answer Relevancy': _fmt_metric(s['answer_relevancy']),
                    'Context Precision': _fmt_metric(s['context_precision']),
                    'Nach Cutoff?': cutoff_label,
                })
        table = dbc.Table.from_dataframe(
            pd.DataFrame(rows), striped=True, bordered=True, hover=True, responsive=True, className="mt-2")

        agg = result["aggregate"]
        agg_text = " · ".join(
            f"{EVAL_METRIC_LABELS[k]}: {_fmt_metric(v)}" for k, v in agg.items() if v is not None
        ) or "Keine Metriken berechnet (keine auswertbare Anomalie mit abgerufenem Kontext)."

        return html.Div([
            html.H5("RAGAS-Bewertung (RAG-interne Qualität)", className="mb-3", style={'color': '#f0f6fc'}),
            table,
            html.P(f"Mittelwerte: {agg_text}", className="mt-2", style={'color': '#f0f6fc'}),
            html.Hr(),
            html.H5("Naive-LLM vs. RAG-LLM — Ereignisbasierte Erklärung", className="mb-3 mt-4",
                    style={'color': '#f0f6fc'}),
            _comparison_table(result["comparison"]),
            html.Hr(),
            html.Small([
                f"Judge-Modell: {result['config']['judge_model']} (identisch zum Generator — "
                "Self-Preference-Bias als Limitation zu beachten). ",
                "Zitierbarkeit prüft nur die Formulierung (benannte vs. vage Quelle), "
                "NICHT die reale Existenz der Quelle (keine Websuche). ",
                f"{result['n_skipped']} Anomalie(n) ohne abgerufenen Kontext (RAG). ",
                f"Ergebnis gespeichert unter {path}.",
            ], className="text-muted"),
        ])
    except Exception as e:
        logger.error(f"Error in RAGAS evaluation: {e}")
        import traceback
        traceback.print_exc()
        return dbc.Alert([
            html.H5("Fehler bei der RAGAS-Evaluation", className="mb-2"),
            html.P(f"Fehlerdetails: {str(e)}"),
        ], color="danger")


def _anomaly_source_modal_body(ticker, date_str):
    """Baut den Modal-Inhalt für die 'Kaynak'-Spalte der Anomalietage-Tabelle.

    Liest AUSSCHLIESSLICH den Cache (NewsCache, kein RAG-Init, kein Netzabruf) und
    zeigt je Artikel den vollständigen gespeicherten Text — genau das, was bei der
    Indexierung tatsächlich gecacht wurde, nicht gekürzt wie in der Quellenliste
    der RAG-Analyse.
    """
    try:
        day = datetime.fromisoformat(str(date_str)[:10])
    except (ValueError, TypeError):
        return dbc.Alert("Ungültiges Datum.", color="warning")

    cache = NewsCache(DEFAULT_CONFIG.db_path)
    articles = cache.get_articles(ticker, day, DEFAULT_CONFIG.anomaly_window_days)
    if not articles:
        return dbc.Alert("Keine Quellen im Cache für dieses Fenster.", color="info")

    cards = []
    for a in articles:
        cards.append(dbc.Card([dbc.CardBody([
            dbc.Row([
                dbc.Col([html.H6(a.get('title') or '(ohne Titel)', style={'color': '#58a6ff'})], width=9),
                dbc.Col([
                    html.Small(a.get('source', 'Unknown'), className="text-muted d-block"),
                    html.Small((a.get('published') or '')[:10], className="text-muted"),
                ], width=3),
            ]),
            html.P(a.get('summary') or '(kein Inhalt gespeichert)',
                   style={'whiteSpace': 'pre-wrap', 'marginTop': '10px'}),
            html.A("📌 Link zur Quelle", href=a.get('link') or '#', target="_blank",
                   className="small", style={'color': '#79c0ff', 'textDecoration': 'underline'}),
        ])], className="card-custom", style={'marginBottom': '10px'}))
    return html.Div(cards)


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
        return _run_rag_analysis(rag_pipeline, analysis_data)

    @app.callback(
        Output('compare-rag-output', 'children'),
        Input('btn-compare-analyze', 'n_clicks'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def analyze_rag_compare(n_clicks, analysis_data):
        """Vergleichs-Tab, rechte Seite — identische Logik wie der RAG-Tab, ausgelöst
        vom gemeinsamen 'btn-compare-analyze'-Button (RAG lädt zuerst Kontext, daher
        i. d. R. einige Sekunden später fertig als die linke/Naive-Seite)."""
        if not n_clicks:
            return ""
        rag_pipeline = rag_provider.get()
        return _run_rag_analysis(rag_pipeline, analysis_data)

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
        return True, _rag_prompt_component(rag_provider, analysis_data)

    @app.callback(
        Output('collapse-compare-rag-prompt', 'is_open'),
        Output('compare-rag-prompt-container', 'children'),
        Input('btn-toggle-compare-rag-prompt', 'n_clicks'),
        State('collapse-compare-rag-prompt', 'is_open'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def toggle_compare_rag_prompt(n_clicks, is_open, analysis_data):
        """Vergleichs-Tab, rechte Seite — gleicher 'Prompt anzeigen' wie im RAG-Tab."""
        if is_open:
            return False, no_update
        return True, _rag_prompt_component(rag_provider, analysis_data)

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

    @app.callback(
        Output('rag-eval-output', 'children'),
        Input('btn-rag-evaluate', 'n_clicks'),
        State('analysis-data', 'data'),
        prevent_initial_call=True,
    )
    def evaluate_rag(n_clicks, analysis_data):
        if not n_clicks:
            return ""
        return _run_ragas_eval(rag_provider, analysis_data)

    @app.callback(
        Output('anomaly-source-modal', 'is_open'),
        Output('anomaly-source-modal-title', 'children'),
        Output('anomaly-source-modal-body', 'children'),
        Input({'type': 'anomaly-source-btn', 'ticker': ALL, 'date': ALL}, 'n_clicks'),
        prevent_initial_call=True,
    )
    def open_anomaly_source_modal(n_clicks_list):
        """Öffnet das Modal für die geklickte 'Kaynak'-Zelle in der Anomalietage-
        Tabelle. Liest ausschließlich den Cache — kein RAG-Trigger, kein Fetch."""
        if not any(n_clicks_list):
            return no_update, no_update, no_update
        triggered = ctx.triggered_id
        if not triggered:
            return no_update, no_update, no_update
        ticker, date_str = triggered['ticker'], triggered['date']
        return True, f"{ticker} — {date_str}", _anomaly_source_modal_body(ticker, date_str)

    @app.callback(
        Output('anomaly-source-modal', 'is_open', allow_duplicate=True),
        Input('anomaly-source-modal-close', 'n_clicks'),
        prevent_initial_call=True,
    )
    def close_anomaly_source_modal(n_clicks):
        return False
