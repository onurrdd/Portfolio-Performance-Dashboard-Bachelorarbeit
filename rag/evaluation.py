"""
RAGAS-Evaluation (Faz 4) — referenzfreie Qualitätsmessung der RAG-Pipeline.

Bewusst OHNE Ground Truth (siehe implementierung_schritte.md, Faz 4 — Hindsight-Bias-
Argument): die Standard-Metriken sind referenzfrei — Faithfulness, Answer Relevancy,
Context Precision (without reference). Judge-LLM = identisches Modell wie der
Generator (GENERATOR_MODEL); Self-Preference-Bias ist eine bewusst in Kauf genommene
Limitation (siehe Thesis).

Anomalie-Sözleşmesi (Kontrakt mit utils/anomaly.py — für Robustheit gegenüber
Änderungen an der Anomalie-Erkennung): aus jedem Anomalie-Dict werden NUR die
Schlüssel `date`, `responsible_ticker`, `ticker_own_return_pct`, `concentration`
gelesen. Solange diese vier Schlüssel bestehen bleiben, läuft die Evaluation
unverändert weiter — unabhängig davon, wie sich Fenstergröße, Schwellenwert oder
sonstige Details des Erkennungsalgorithmus ändern.

Kaynak bağımsızlığı (Robustheit gegenüber neuen News-Quellen): diese Datei kennt
keinen einzigen Quellennamen (SEC EDGAR, Alpha Vantage, ...) — nur die Pipeline-
Schnittstelle (`retrieve_for_anomaly` / `format_context_for_llm`). Eine neue Quelle
in news/base.py::get_sources() wirkt sich automatisch aus, ohne dass dieses Modul
angefasst werden muss. Die aktive Quellenliste wird lediglich zur Nachvollziehbarkeit
im Config-Snapshot mitgeschrieben (siehe run_ragas_evaluation).

Optionaler Ground-Truth-Hook (siehe GROUND_TRUTH_PATH): existiert `data/ground_truth.json`
NICHT, ändert sich am obigen Verhalten nichts. Existiert die Datei, wird für die
passenden Anomalien zusätzlich `LLMContextRecall` berechnet (referenzbasiert) — der
Rest der Stichprobe bleibt referenzfrei. So lässt sich Ground Truth später (falls
zeitlich möglich) inkrementell nachrüsten, ohne den Code zu ändern.

Naive-LLM-vs-RAG-LLM-Vergleich (siehe classify_explanation, comparison-Block in
run_ragas_evaluation): zusätzlich zu den RAG-internen RAGAS-Metriken wird für JEDE
Einzeltitel-Anomalie — unabhängig davon, ob RAG Kontext gefunden hat — sowohl eine
Naive- als auch eine RAG-Antwort erzeugt und vom Hakem-LLM symmetrisch nach Spezifität
und Zitierbarkeit klassifiziert (siehe About_Thesis.md Forschungsfrage-Teilfragen 1+2).
Das ist bewusst KEIN RAGAS-Metrik-Objekt (Faithfulness/Context Precision bleiben
unverändert RAG-intern) — die Klassifikation ist ein eigenständiger, ebenfalls
referenzfreier Hakem-Aufruf. Grenze: der Hakem kann die reale Existenz eines genannten
Quellennamens nicht verifizieren (keine Websuche) — klassifiziert wird nur die
Konkretheit der FORMULIERUNG, nicht ihr Wahrheitsgehalt.
"""
import os
import json
import logging
from datetime import datetime

from callbacks.naive_llm import GENERATOR_MODEL, ANOMALY_PROMPT_SAFE_MODE, build_portfolio_prompt

logger = logging.getLogger(__name__)

# Judge = Generator (bewusst identisch) — siehe Docstring oben.
JUDGE_MODEL = GENERATOR_MODEL
GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"

# Wissensstand des Generator-Modells (openai/gpt-oss-120b), laut offiziellem Model
# Card ca. Juni 2024 (Quelle: OpenAI gpt-oss Model Card, 05.08.2025; einzelne
# Sekundärquellen nennen Mai/Juli 2024 — Datum ist daher als ungefähr zu behandeln
# und im Thesistext entsprechend zu relativieren). Dient nur der Vor/Nach-Cutoff-
# Etikettierung der Stichprobe, nicht als exakte Wahrheitsgrenze.
GENERATOR_KNOWLEDGE_CUTOFF = "2024-06-01"

TOP_K = 3  # identisch zur Produktions-Retrieval (siehe callbacks/rag.py::_collect_rag_context)
RETRIEVAL_QUERY = "reason for the stock price move, earnings, guidance or company news"  # dito

GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_DIR = "data"


def _single_stock_anomalies(analysis_data):
    """Lokale Kopie des Filters aus callbacks/rag.py::_single_stock_anomalies.

    Bewusst dupliziert statt importiert: vermeidet einen Importzyklus
    (callbacks.rag importiert bereits aus rag.evaluation) und hält die Anomalie-
    Sözleşmesi dieses Moduls (siehe Modul-Docstring) an einer einzigen, expliziten
    Stelle sichtbar.
    """
    return [
        b for b in (analysis_data or {}).get('active_return_breaks', [])
        if b.get('concentration') == 'Hisseye özgü' and b.get('responsible_ticker')
    ]


def _load_ground_truth():
    """Lädt data/ground_truth.json, falls vorhanden. Format:
    {"YYYY-MM-DD|TICKER": "Referenzantwort", ...}. Fehlt die Datei, {} (kein Effekt)."""
    if not os.path.exists(GROUND_TRUTH_PATH):
        return {}
    try:
        with open(GROUND_TRUTH_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning(f"ground_truth.json konnte nicht gelesen werden: {e}")
        return {}


def build_eval_samples(rag_pipeline, analysis_data):
    """Baut eine RAGAS-Stichprobe PRO Anomalie (Frage/Kontexte/Status).

    Nur Einzeltitel-Anomalien (siehe _single_stock_anomalies) — identisch zur
    Produktionslogik. Kontext-Retrieval ist eins zu eins die Produktionsfunktion
    (gleiche Query, gleiches top_k), damit die Evaluation misst, was der Nutzer
    tatsächlich als Kontext bekommt.
    """
    ground_truth = _load_ground_truth()
    samples = []
    for b in _single_stock_anomalies(analysis_data):
        ticker = b.get('responsible_ticker')
        date_str = b.get('date')
        own_return = b.get('ticker_own_return_pct', 0) or 0
        direction = "rise" if own_return >= 0 else "fall"
        question = f"Why did {ticker} {direction} {own_return:+.2f}% on {date_str}?"

        chunks = rag_pipeline.retrieve_for_anomaly(RETRIEVAL_QUERY, b, top_k=TOP_K)
        contexts = [c.get("text", "") for c in chunks]

        post_cutoff = None
        try:
            post_cutoff = str(date_str)[:10] > GENERATOR_KNOWLEDGE_CUTOFF
        except TypeError:
            pass

        samples.append({
            "anomaly": b,
            "date": date_str,
            "ticker": ticker,
            "question": question,
            "chunks": chunks,
            "contexts": contexts,
            "status": "ok" if contexts else "no_context",
            "post_cutoff": post_cutoff,
            "reference": ground_truth.get(f"{date_str}|{ticker}"),
        })
    return samples


def generate_answer(rag_pipeline, analysis_data, sample):
    """Erzeugt die LLM-Antwort für EINE Anomalie — mit dem IDENTISCHEN Prompt-Template
    wie Naive-LLM/RAG-LLM (build_portfolio_prompt), aber auf diese eine Anomalie
    beschränkt (active_return_breaks = [sample['anomaly']]), damit Faithfulness die
    Antwort gegen exakt die dafür abgerufenen Kontexte prüfen kann."""
    from groq import Groq

    single_data = {**(analysis_data or {}), "active_return_breaks": [sample["anomaly"]]}
    news_context = rag_pipeline.format_context_for_llm(sample["chunks"], max_tokens=2000)
    prompt = build_portfolio_prompt(single_data, news_context=news_context or None)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GENERATOR_MODEL,
        max_tokens=4096,
    )
    return response.choices[0].message.content


def generate_naive_answer(analysis_data, sample):
    """Symmetrisch zu generate_answer, aber OHNE abgerufenen Kontext — spiegelt exakt
    den Naive-LLM-Tab (build_portfolio_prompt ohne news_context). Wird für JEDE
    Anomalie erzeugt, auch wenn RAG keinen Kontext gefunden hat (status='no_context'):
    der Vergleich braucht in diesem Fall gerade die Gegenüberstellung."""
    from groq import Groq

    single_data = {**(analysis_data or {}), "active_return_breaks": [sample["anomaly"]]}
    prompt = build_portfolio_prompt(single_data)

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model=GENERATOR_MODEL,
        max_tokens=4096,
    )
    return response.choices[0].message.content


SPECIFICITY_CATEGORIES = ("concrete_event", "generic", "no_cause_given")
CITATION_CATEGORIES = ("named_source", "vague_reference", "no_citation")

_CLASSIFY_PROMPT = """You are evaluating an AI-generated explanation for a stock price move.

Question: {question}

Answer to evaluate:
\"\"\"{answer}\"\"\"

Classify the answer on two dimensions and respond with ONLY a JSON object, no other text:

1. "specificity": one of
   - "concrete_event": names a specific, dated event (earnings report, product launch, executive change, regulatory/legal news, M&A, analyst rating action, or a similarly specific cause) tied to this exact ticker and date.
   - "generic": resorts to vague, non-specific explanations (e.g. "market volatility", "investor sentiment", general macroeconomic conditions) without naming a specific event.
   - "no_cause_given": explicitly states it does not know or cannot determine the cause.

2. "citation_type": one of
   - "named_source": cites a specific, checkable source (e.g. a named news outlet, a specific SEC filing type, a specific earnings call).
   - "vague_reference": refers to a source only vaguely (e.g. "reports suggest", "it was reported") without naming anything checkable.
   - "no_citation": states a cause with no source language at all.

Respond with exactly this JSON shape: {{"specificity": "...", "citation_type": "..."}}"""


def classify_explanation(question, answer, api_key):
    """Symmetrische Hakem-Klassifikation EINER Antwort (Naive ODER RAG) — Spezifität
    und Zitierbarkeit, siehe About_Thesis.md Forschungsfrage-Teilfragen 1 (Spezifität)
    und 2 (Belegbarkeit, Naive-Seite). Absichtlich EIN strukturierter Aufruf statt
    RAGAS-AspectCritic (siehe implementierung_schritte.md, Faz 4) — kategorial statt
    binär, kein zusätzliches Dataset/Metrik-Objekt nötig.

    Grenze: kann die reale Existenz eines genannten Quellennamens NICHT verifizieren
    (keine Websuche, Projektentscheidung) — klassifiziert nur, wie konkret/nachprüfbar
    die Antwort FORMULIERT ist.

    Fehlertolerant (Timeout/JSON-Parse-Fehler): liefert {"specificity": None,
    "citation_type": None} statt die gesamte Auswertung abzubrechen.
    """
    from groq import Groq

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": _CLASSIFY_PROMPT.format(question=question, answer=answer)}],
            model=JUDGE_MODEL,
            max_tokens=1024,
            reasoning_effort="low",
            response_format={"type": "json_object"},
        )
        parsed = json.loads(response.choices[0].message.content)
        specificity = parsed.get("specificity")
        citation_type = parsed.get("citation_type")
        return {
            "specificity": specificity if specificity in SPECIFICITY_CATEGORIES else None,
            "citation_type": citation_type if citation_type in CITATION_CATEGORIES else None,
        }
    except Exception as e:
        logger.warning(f"classify_explanation fehlgeschlagen: {e}")
        return {"specificity": None, "citation_type": None}


def _build_local_embeddings(embedder):
    """Wrappt den bereits geladenen EmbeddingGenerator (all-MiniLM-L6-v2) für RAGAS'
    ResponseRelevancy — verhindert, dass ein zweites Embedding-Modell separat geladen
    wird. Muss von langchain_core.embeddings.Embeddings erben, weil ragas.evaluate()
    über isinstance() prüft, ob eine Einbettung gewrappt werden muss (siehe
    ragas/evaluation.py); die Klasse wird hier (nicht auf Modulebene) definiert, damit
    der langchain_core-Import lazy bleibt."""
    from langchain_core.embeddings import Embeddings

    class LocalEmbeddings(Embeddings):
        def embed_query(self, text):
            return embedder.embed_text(text)

        def embed_documents(self, texts):
            return embedder.embed_text(list(texts))

    return LocalEmbeddings()


def run_ragas_evaluation(rag_pipeline, analysis_data, api_key):
    """Führt die vollständige Evaluation aus — zwei unabhängige Schichten:

    1) RAG-interne RAGAS-Metriken (Faithfulness, Answer Relevancy, Context Precision,
       optional Context Recall) — NUR für Anomalien mit abgerufenem Kontext
       (status='ok'), unverändert gegenüber der ursprünglichen Version.
    2) Naive-LLM-vs-RAG-LLM-Vergleich (Spezifität, Zitierbarkeit; siehe
       classify_explanation) — für JEDE Einzeltitel-Anomalie, unabhängig vom Kontext-
       Status, weil Naive-LLM ohnehin nie Kontext braucht und der Fall "RAG fand
       keinen Kontext" selbst ein Vergleichsergebnis ist (RAG degradiert dann zu
       Naive).

    Rückgabe: {"samples": [...], "aggregate": {...}, "comparison": {...},
               "config": {...}, "n_skipped": int, "timestamp": iso-str}
    """
    import numpy as np
    from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
    from langchain_openai import ChatOpenAI

    samples = build_eval_samples(rag_pipeline, analysis_data)
    evaluable = [s for s in samples if s["status"] == "ok"]
    n_skipped = len(samples) - len(evaluable)

    # 1) Antworten erzeugen — RAG- UND Naive-Antwort für ALLE Anomalien (sequenziell,
    # Groq-Free-Tier-freundlich). RAG-Antwort ist bei status='no_context' identisch
    # zur Naive-Antwort im Prompt-Aufbau (news_context=None in beiden Fällen) — der
    # Vergleich zeigt dann konsequenterweise Konvergenz statt eines Fehlers.
    for s in samples:
        s["rag_answer"] = generate_answer(rag_pipeline, analysis_data, s)
        s["naive_answer"] = generate_naive_answer(analysis_data, s)

    # 2) Symmetrische Hakem-Klassifikation beider Antworten (siehe classify_explanation).
    # Eigenständig fehlertolerant — ein einzelner Klassifikationsfehler bricht die
    # gesamte Auswertung nicht ab.
    for s in samples:
        s["rag_classification"] = classify_explanation(s["question"], s["rag_answer"], api_key)
        s["naive_classification"] = classify_explanation(s["question"], s["naive_answer"], api_key)

    # 3) RAG-interne RAGAS-Metriken — NUR für Anomalien mit Kontext.
    if evaluable:
        has_any_reference = any(s["reference"] for s in evaluable)
        ragas_samples = [
            SingleTurnSample(
                user_input=s["question"],
                retrieved_contexts=s["contexts"],
                response=s["rag_answer"],
                reference=s["reference"],
            )
            for s in evaluable
        ]
        dataset = EvaluationDataset(samples=ragas_samples)

        # reasoning_effort="low": openai/gpt-oss-120b ist ein Reasoning-Modell und
        # verbraucht sonst einen variablen, teils großen Anteil von max_tokens für
        # einen internen Denkschritt VOR der eigentlichen Ausgabe. Bei Faithfulness
        # (zwei verkettete, längere strukturierte LLM-Aufrufe: Aussagenzerlegung +
        # Verifikation) führte das ohne diese Option zu abgeschnittenen Antworten
        # (LLMDidNotFinishException) und damit zu NaN. "low" hält den Denkschritt kurz
        # und lässt genug Tokens für die eigentliche strukturierte Ausgabe übrig.
        raw_llm = ChatOpenAI(
            model=JUDGE_MODEL, base_url=GROQ_OPENAI_BASE_URL,
            api_key=api_key, temperature=0, max_tokens=4096,
            model_kwargs={"reasoning_effort": "low"},
        )
        # max_workers=1 + Retries: Groq-Free-Tier-Rate-Limits (429) über ~9-10 Judge-
        # Aufrufe je Anomalie. bypass_n=True: openai/gpt-oss-120b (Reasoning-Modell,
        # via Groqs OpenAI-kompatiblem Endpoint) lehnt n>1 ab ("'n' : number must be
        # at most 1"); ResponseRelevancy würde sonst mit n=strictness aufrufen und
        # scheitern — bypass_n erzwingt stattdessen sequenzielle Einzel-Aufrufe.
        run_config = RunConfig(timeout=120, max_retries=8, max_wait=60, max_workers=1)
        judge = LangchainLLMWrapper(raw_llm, run_config=run_config, bypass_n=True)

        metrics = [Faithfulness(), ResponseRelevancy(), LLMContextPrecisionWithoutReference()]
        if has_any_reference:
            metrics.append(LLMContextRecall())

        embeddings = _build_local_embeddings(rag_pipeline.embedder)

        eval_result = evaluate(
            dataset, metrics=metrics, llm=judge, embeddings=embeddings,
            run_config=run_config, raise_exceptions=False, show_progress=False,
        )
        for s, score in zip(evaluable, eval_result.scores):
            s["_ragas_score"] = score

    # 4) Ergebniszeilen — EINHEITLICH für alle Samples (RAGAS-Metriken bleiben None
    # außerhalb von 'evaluable', Vergleichsfelder sind für alle Samples befüllt).
    result_samples = []
    for s in samples:
        score = s.get("_ragas_score", {})
        result_samples.append({
            "date": s["date"], "ticker": s["ticker"], "question": s["question"],
            "status": s["status"], "n_contexts": len(s["contexts"]),
            "post_cutoff": s["post_cutoff"], "has_reference": s["reference"] is not None,
            "faithfulness": _clean(score.get("faithfulness")),
            "answer_relevancy": _clean(score.get("answer_relevancy")),
            "context_precision": _clean(score.get("llm_context_precision_without_reference")),
            "context_recall": _clean(score.get("context_recall")),
            "rag_answer": s["rag_answer"],
            "naive_answer": s["naive_answer"],
            "specificity_rag": s["rag_classification"]["specificity"],
            "citation_rag": s["rag_classification"]["citation_type"],
            "specificity_naive": s["naive_classification"]["specificity"],
            "citation_naive": s["naive_classification"]["citation_type"],
        })

    def _mean(key, cutoff_filter=None):
        vals = [
            r[key] for r in result_samples
            if r.get(key) is not None and (cutoff_filter is None or r["post_cutoff"] == cutoff_filter)
        ]
        return float(np.mean(vals)) if vals else None

    def _category_pct(field, target_value, cutoff_filter=None, context_only=False):
        """% Anteil von rows[field] == target_value, unter nicht-None Werten."""
        rows = [r for r in result_samples if r.get(field) is not None]
        if cutoff_filter is not None:
            rows = [r for r in rows if r["post_cutoff"] == cutoff_filter]
        if context_only:
            rows = [r for r in rows if r["status"] == "ok"]
        if not rows:
            return None
        return round(100 * sum(1 for r in rows if r[field] == target_value) / len(rows), 1)

    aggregate = {
        "faithfulness": _mean("faithfulness"),
        "answer_relevancy": _mean("answer_relevancy"),
        "context_precision": _mean("context_precision"),
        "context_recall": _mean("context_recall"),
    }

    # Naive-LLM-vs-RAG-LLM-Vergleich (siehe About_Thesis.md Forschungsfrage-Teilfragen
    # 1-4): Anteil konkreter Ursachenbenennung (Spezifität) und benannter Quellen
    # (Zitierbarkeit, Belegbarkeit-Proxy für die Naive-Seite), je Bedingung und
    # Cutoff-Gruppe. RAGs Belegbarkeit wird NICHT neu berechnet, sondern über die
    # bereits vorhandene Faithfulness wiederverwendet (keine Doppelberechnung).
    comparison = {
        "specificity_pct": {
            "naive": {
                "all": _category_pct("specificity_naive", "concrete_event"),
                "pre_cutoff": _category_pct("specificity_naive", "concrete_event", cutoff_filter=False),
                "post_cutoff": _category_pct("specificity_naive", "concrete_event", cutoff_filter=True),
            },
            "rag": {
                "all": _category_pct("specificity_rag", "concrete_event"),
                "pre_cutoff": _category_pct("specificity_rag", "concrete_event", cutoff_filter=False),
                "post_cutoff": _category_pct("specificity_rag", "concrete_event", cutoff_filter=True),
                "context_available_only": _category_pct("specificity_rag", "concrete_event", context_only=True),
            },
        },
        "named_citation_pct": {
            "naive": {
                "all": _category_pct("citation_naive", "named_source"),
                "pre_cutoff": _category_pct("citation_naive", "named_source", cutoff_filter=False),
                "post_cutoff": _category_pct("citation_naive", "named_source", cutoff_filter=True),
            },
            "rag": {
                "all": _category_pct("citation_rag", "named_source"),
                "pre_cutoff": _category_pct("citation_rag", "named_source", cutoff_filter=False),
                "post_cutoff": _category_pct("citation_rag", "named_source", cutoff_filter=True),
            },
        },
        "rag_faithfulness_as_belegbarkeit": {
            "all": aggregate["faithfulness"],
            "pre_cutoff": _mean("faithfulness", cutoff_filter=False),
            "post_cutoff": _mean("faithfulness", cutoff_filter=True),
        },
    }

    config = {
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "embedding_model": rag_pipeline.embedder.model_name,
        "top_k": TOP_K,
        "metrics": ["faithfulness", "answer_relevancy", "context_precision"]
                   + (["context_recall"] if any(r.get("has_reference") for r in result_samples) else []),
        "ragas_version": _ragas_version(),
        "anomaly_prompt_safe_mode": ANOMALY_PROMPT_SAFE_MODE,
        "active_sources": [getattr(s, "name", "?") for s in rag_pipeline.sources],
        "generator_knowledge_cutoff": GENERATOR_KNOWLEDGE_CUTOFF,
        "specificity_categories": list(SPECIFICITY_CATEGORIES),
        "citation_categories": list(CITATION_CATEGORIES),
        "citation_verification_limitation": (
            "Der Hakem prueft nur die Formulierung (konkreter Name vs. vage Angabe), "
            "NICHT die reale Existenz einer genannten Quelle (keine Websuche)."
        ),
    }

    return {
        "samples": sorted(result_samples, key=lambda r: (r["date"] or "", r["ticker"] or "")),
        "aggregate": aggregate,
        "comparison": comparison,
        "config": config,
        "n_skipped": n_skipped,
        "timestamp": datetime.now().isoformat(),
    }


def _clean(value):
    """NaN -> None (JSON-serialisierbar, in der UI als "—" darstellbar)."""
    import math
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _ragas_version():
    try:
        import ragas
        return ragas.__version__
    except Exception:
        return "unknown"


def save_eval_results(result):
    """Speichert das Evaluationsergebnis als Zeitstempel-JSON unter data/."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(RESULTS_DIR, f"ragas_eval_{ts}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return path
