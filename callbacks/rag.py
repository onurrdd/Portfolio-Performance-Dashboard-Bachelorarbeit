import os
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update, ALL, ctx

from rag.cache import NewsCache
from rag.config import DEFAULT_CONFIG
from rag import config as rag_config  # Sparmodus-Schalter, als Attribut gelesen

# Identischer Prompt-Builder UND identisches Modell wie im Naive-Tab — ein einziger
# Ort (GENERATOR_MODEL) verhindert ein stilles Auseinanderlaufen der beiden Modelle.
# make_llm_client/llm_api_key kapseln zudem die LLM-Anbindung (OpenRouter,
# siehe callbacks/naive_llm.py).
from callbacks.naive_llm import (build_portfolio_prompt, PROMPT_DEBUG_STYLE,
                                 GENERATOR_MODEL, GENERATION_MAX_TOKENS,
                                 llm_chat_with_retry, select_prompt_anomalies,
                                 single_stock_anomalies, prompt_anomaly_coverage_note,
                                 effective_generation_max_tokens,
                                 make_llm_client, llm_api_key, LLM_PROVIDER,
                                 single_anomaly_data, _anomaly_llm_answer,
                                 _anomaly_answer_heading)


def _anomalies_to_index(analysis_data):
    """Anomalien, für die Nachrichten geholt/indiziert werden. Im Sparmodus NUR die
    im Prompt gelistete Auswahl — sonst würde der (kontingentierte) Alpha-Vantage-
    Abruf über alle Anomalien laufen, obwohl das Retrieval später nur die Auswahl
    anfragt. Im Vollmodus unverändert: alle Einzeltitel-Anomalien."""
    if rag_config.SAVING_MODE:
        return select_prompt_anomalies(analysis_data)
    return single_stock_anomalies(analysis_data)

logger = logging.getLogger(__name__)

MODEL = GENERATOR_MODEL
# Antwortbudget kommt über effective_generation_max_tokens() aus callbacks/naive_llm.py —
# EIN Aufruf für beide Bedingungen, damit sie zwangsläufig denselben Wert benutzen (auch im
# Testmodus, siehe dort) und ein zweiter Literalwert nicht unbemerkt auseinanderlaufen kann.

# Serialisiert JEDEN Zugriff auf Vektorspeicher/Cache (FAISS ist nicht thread-sicher).
# Auto-Fetch (Hintergrund) und manuelle Analyse dürfen den Store nicht gleichzeitig anfassen.
_rag_lock = threading.Lock()


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
    for b in _anomalies_to_index(analysis_data):
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


# Kontextbudget je Anomalie-Aufruf. Anwendung und rag/evaluation.py bauen beide
# einen eigenen Aufruf PRO Anomalie (kein pooled Prompt mehr) — derselbe Wert wie
# dort (generate_answer: max_tokens=2000), damit Demonstration und Evaluation
# denselben Kontextumfang sehen.
PER_ANOMALY_CONTEXT_TOKEN_BUDGET = 2000
RETRIEVAL_TOP_K = 3
RETRIEVAL_QUERY = "reason for the stock price move, earnings, guidance or company news"


def _retrieve_for_selected(rag_pipeline, analysis_data):
    """Retrieval je im Prompt gelisteter Einzeltitel-Anomalie.

    Rückgabe: Liste von (anomalie, chunks) — chunks ist die für genau diese
    Anomalie (Ticker + Zeitfenster) abgerufene Trefferliste, ohne Fallback. Die
    Deduplizierung über Anomalien hinweg entfällt bewusst: jeder Aufruf ist
    eigenständig, ein Chunk darf für zwei verschiedene Ereignisse relevant sein.
    """
    return [
        (b, rag_pipeline.retrieve_for_anomaly(RETRIEVAL_QUERY, b, top_k=RETRIEVAL_TOP_K))
        for b in select_prompt_anomalies(analysis_data)
    ]


def _source_card(i, chunk):
    md = chunk.get('metadata', {})
    not_dated = md.get('date_filtered') is False
    return dbc.Card([dbc.CardBody([
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
    ])], className="card-custom", style={'marginBottom': '10px'})


def _run_rag_analysis(rag_pipeline, analysis_data):
    """RAG-Analyse: je Einzeltitel-Anomalie ein eigener Retrieval + LLM-Aufuf (kein
    pooled Prompt mehr) — dieselbe Ein-Ereignis-Mechanik wie rag/evaluation.py, damit
    Demonstration und Evaluation über denselben Pfad laufen.

    Gemeinsame Logik für den RAG-Tab UND den Vergleichs-Tab.
    """
    if not rag_pipeline:
        return dbc.Alert("RAG Pipeline nicht verfügbar oder nicht initialisiert", color="warning")
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")

    try:
        # Indizieren + Retrieval unter dem Store-Lock (kein paralleler FAISS-Zugriff
        # mit dem Auto-Fetch-Thread). Die LLM-Aufrufe danach brauchen den Lock nicht.
        with _rag_lock:
            _index_anomaly_news(rag_pipeline, analysis_data)
            per_anomaly = _retrieve_for_selected(rag_pipeline, analysis_data)

        if not per_anomaly:
            return dbc.Alert("Keine Einzeltitel-Anomalien zu erklären.", color="info")

        # Kontext je Anomalie formatieren; die LLM-Aufrufe laufen nebenläufig.
        prepared = []
        for b, chunks in per_anomaly:
            ctx, used = rag_pipeline.format_context_for_llm(
                chunks, max_tokens=PER_ANOMALY_CONTEXT_TOKEN_BUDGET, return_used=True)
            prepared.append((b, ctx, used))

        results = [None] * len(prepared)

        def _task(idx, b, ctx):
            try:
                results[idx] = _anomaly_llm_answer(analysis_data, b, news_context=ctx or None)
            except Exception as e:
                results[idx] = f"[Fehler: {e}]"

        with ThreadPoolExecutor(max_workers=min(len(prepared), 8)) as pool:
            for fut in [pool.submit(_task, i, b, ctx) for i, (b, ctx, _u) in enumerate(prepared)]:
                fut.result()

        components = [html.H5("RAG-gestützte Analyse — Erklärung je Anomalie",
                              className="mb-3", style={'color': '#f0f6fc'})]
        all_used = []
        for (b, _ctx, used), text in zip(prepared, results):
            all_used.extend(used)
            components.append(dbc.Card([dbc.CardBody([
                html.H6(_anomaly_answer_heading(b), className="mb-2", style={'color': '#58a6ff'}),
                html.Div(text, style={'whiteSpace': 'pre-wrap', 'lineHeight': '1.6',
                                      'fontSize': '0.95rem', 'color': '#f0f6fc'}),
                html.Small(
                    f"{len(used)} abgerufene Nachrichten-Snippet(s) als Kontext"
                    if used else "(no sources retrieved for this event)",
                    className="text-muted"),
            ])], className="card-custom", style={'marginBottom': '10px'}))

        components.append(html.Small(
            f"Erstellt mit {LLM_PROVIDER} ({MODEL}) · je Anomalie ein eigener Aufruf"
            + prompt_anomaly_coverage_note(analysis_data), className="text-muted"))

        if all_used:
            components.append(html.H5("📰 Verwendete Quellen", style={
                'color': '#79c0ff', 'marginTop': '20px', 'marginBottom': '10px'}))
            components.extend(_source_card(i, c) for i, c in enumerate(all_used, 1))

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
    """Debug-Anzeige der RAG-Prompts (inkl. abgerufenem Kontext) — je Anomalie ein
    Prompt, mit Trenner dazwischen. Gemeinsame Logik für RAG-Tab UND Vergleichs-Tab.
    """
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar.", color="warning")

    rag_pipeline = rag_provider.get()
    if not rag_pipeline:
        return html.Pre(
            build_portfolio_prompt(analysis_data) + "\n\n[RAG nicht verfügbar — Nachrichten-Kontext fehlt]",
            style=PROMPT_DEBUG_STYLE)

    with _rag_lock:
        _index_anomaly_news(rag_pipeline, analysis_data)
        per_anomaly = _retrieve_for_selected(rag_pipeline, analysis_data)

    if not per_anomaly:
        return html.Pre(build_portfolio_prompt(analysis_data), style=PROMPT_DEBUG_STYLE)

    parts = []
    for b, chunks in per_anomaly:
        ctx = rag_pipeline.format_context_for_llm(chunks, max_tokens=PER_ANOMALY_CONTEXT_TOKEN_BUDGET)
        parts.append(build_portfolio_prompt(single_anomaly_data(analysis_data, b),
                                            news_context=ctx or None))
    return html.Pre(("\n\n" + "=" * 70 + "\n\n").join(parts), style=PROMPT_DEBUG_STYLE)


EVAL_METRIC_LABELS = {
    "faithfulness": "Faithfulness (Quellentreue, RAG)",
    "answer_relevancy": "Answer Relevancy",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall (Ground Truth)",
    # Dieselbe Metrik wie "faithfulness" (Faithfulness gegen dieselbe Quellenbasis),
    # nur für die Naive-Antwort erhoben — Name betont den Vergleich, nicht eine
    # andersartige Metrik (siehe rag/evaluation.py, naive_dataset).
    "source_support_naive": "Faithfulness (Naive, gegen dieselbe Quellenbasis)",
    "answer_relevancy_naive": "Answer Relevancy (Naive)",
}


def _fmt_metric(v):
    return "—" if v is None else f"{v:.2f}"


def _fmt_pct(v):
    return "—" if v is None else f"{v:.1f}%"


def _group_size_note(group_sizes):
    """Bezugsmengen der Cutoff-Gruppen als Fliesstext. `total` ist die Zahl der
    Anomalien in der Gruppe, `evaluable` die Teilmenge mit abgerufenem Kontext —
    nur diese geht in die Quellendeckung ein."""
    if not group_sizes:
        return ""
    pre, post = group_sizes.get("pre_cutoff", {}), group_sizes.get("post_cutoff", {})
    unknown = group_sizes.get("unknown", {})
    text = (f"Bezugsmengen — vor Cutoff: n={pre.get('total', 0)} "
            f"(auswertbar {pre.get('evaluable', 0)}), "
            f"nach Cutoff: n={post.get('total', 0)} "
            f"(auswertbar {post.get('evaluable', 0)})")
    if unknown.get("total"):
        text += f", ohne Datumszuordnung: n={unknown['total']}"
    return text + ". "


def _comparison_table(comparison):
    """Naive-LLM-vs-RAG-LLM-Vergleich (Spezifität, Zitierbarkeit) — siehe
    About_Thesis.md Forschungsfrage-Teilfragen 1-4 und rag/evaluation.py::comparison.
    Kein direktes RAGAS-Metrik-Objekt; separate Hakem-Klassifikation, siehe dort.
    """
    spec, cite = comparison["specificity_pct"], comparison["named_citation_pct"]
    # Belegbarkeit wird für BEIDE Bedingungen an derselben Quellenbasis erhoben
    # (siehe rag/evaluation.py, Durchgang 3b); die Nicht-Deckungsquote ist die
    # Operationalisierung der in der Forschungsfrage genannten Halluzinationsrate.
    support, unsupported = (
        comparison["source_support"], comparison["unsupported_claim_rate"],
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
            "Metrik": "Quellendeckung (Anteil belegter Aussagen)",
            "Naive — Alle": _fmt_metric(support["naive"]["all"]),
            "Naive — Vor Cutoff": _fmt_metric(support["naive"]["pre_cutoff"]),
            "Naive — Nach Cutoff": _fmt_metric(support["naive"]["post_cutoff"]),
            "RAG — Alle": _fmt_metric(support["rag"]["all"]),
            "RAG — Vor Cutoff": _fmt_metric(support["rag"]["pre_cutoff"]),
            "RAG — Nach Cutoff": _fmt_metric(support["rag"]["post_cutoff"]),
        },
        {
            "Metrik": "  ↳ Nicht belegte Aussagen (Halluzinationsrate)",
            "Naive — Alle": _fmt_metric(unsupported["naive"]["all"]),
            "Naive — Vor Cutoff": _fmt_metric(unsupported["naive"]["pre_cutoff"]),
            "Naive — Nach Cutoff": _fmt_metric(unsupported["naive"]["post_cutoff"]),
            "RAG — Alle": _fmt_metric(unsupported["rag"]["all"]),
            "RAG — Vor Cutoff": _fmt_metric(unsupported["rag"]["pre_cutoff"]),
            "RAG — Nach Cutoff": _fmt_metric(unsupported["rag"]["post_cutoff"]),
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
    if not llm_api_key():
        return dbc.Alert("LLM-API-Schlüssel nicht gesetzt (siehe LLM_PROVIDER).", color="warning")
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")
    if not single_stock_anomalies(analysis_data):
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
            result = run_ragas_evaluation(rag_pipeline, analysis_data, llm_api_key())
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
            # Fällt auf den Schlüsselnamen zurück: eine neu hinzugekommene Metrik
            # ohne Beschriftung soll die Anzeige eines fertigen Laufs nicht verwerfen.
            f"{EVAL_METRIC_LABELS.get(k, k)}: {_fmt_metric(v)}"
            for k, v in agg.items() if v is not None
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
                "Quellendeckung misst beide Bedingungen an derselben abgerufenen "
                "Quellenbasis; die Naive-Bedingung hat diesen Kontext nie gesehen. "
                "Nicht belegt heißt nicht falsch — eine Aussage kann zutreffen und "
                "trotzdem nicht in den abgerufenen Dokumenten stehen. ",
                f"{result['n_skipped']} Anomalie(n) ohne abgerufenen Kontext (RAG). ",
                # Bezugsmengen der Cutoff-Gruppen: ohne sie ist eine Quote nicht
                # deutbar (siehe rag/evaluation.py::group_sizes).
                _group_size_note(result.get("group_sizes")),
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
    articles = cache.get_articles(ticker, day, DEFAULT_CONFIG.anomaly_window_days,
                                  window_days_after=DEFAULT_CONFIG.anomaly_window_days_after)
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
        if not analysis_data or not single_stock_anomalies(analysis_data):
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
        Output('rag-reindex-status', 'children'),
        Input('btn-rag-reindex', 'n_clicks'),
        prevent_initial_call=True,
    )
    def reindex_from_cache(n_clicks):
        """Segmentiert die gecachten Artikel mit dem AKTUELLEN Chunker neu.

        Ohne diesen Schritt bleibt eine Änderung an rag/chunker.py wirkungslos: der
        Index wächst nur um neue Artikel, die vorhandenen Chunks bleiben unberührt.
        Läuft im Prozess der App — der Effekt ist ohne Neustart im nächsten
        'Prompt anzeigen' sichtbar."""
        rag_pipeline = rag_provider.get()
        if not rag_pipeline:
            return html.Span("RAG nicht verfügbar", className="text-danger")
        try:
            # Unter _rag_lock: der Neuaufbau leert den Index kurzzeitig
            # (vectorstore.reset()). Ein gleichzeitiges Retrieval würde sonst auf
            # einem halb gefüllten Index suchen und stillschweigend zu wenig finden.
            with _rag_lock:
                stats = rag_pipeline.reindex_from_cache()
        except Exception as e:
            logger.exception("Reindex fehlgeschlagen")
            return html.Span(f"Fehler: {e}", className="text-danger")
        scope = stats.get("scope")
        note = (f" — Sparmodus: nur {', '.join(scope)}; vor einem vollen Lauf "
                f"ohne Sparmodus erneut aufbauen" if scope else " — volles Portfolio")
        return html.Span(
            f"{stats['articles']} Artikel neu segmentiert: "
            f"{stats['chunks_before']} → {stats['chunks_after']} Chunks{note}",
            className="text-success")

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
