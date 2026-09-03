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
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from callbacks.naive_llm import (GENERATOR_MODEL, ANOMALY_PROMPT_SAFE_MODE,
                                 build_portfolio_prompt, llm_chat_with_retry,
                                 make_llm_client, llm_base_url, llm_api_key,
                                 LLM_PROVIDER, GENERATOR_KNOWLEDGE_CUTOFF,
                                 select_prompt_anomalies)
from rag import config as rag_config  # Sparmodus-Schalter, als Attribut gelesen

logger = logging.getLogger(__name__)

# Judge = Generator (bewusst identisch) — siehe Docstring oben.
JUDGE_MODEL = GENERATOR_MODEL
# Anbieter-Endpunkt/-Schlüssel kommen zentral aus callbacks/naive_llm.py
# (OpenRouter, OpenAI-kompatible Schnittstelle).
#
# GENERATOR_KNOWLEDGE_CUTOFF ist eine Eigenschaft des Generator-Modells und wohnt
# daher in callbacks/naive_llm.py; hier nur importiert (Vor/Nach-Cutoff-Etikett).

TOP_K = 3  # identisch zur Anwendung (callbacks/rag.py::RETRIEVAL_TOP_K)
RETRIEVAL_QUERY = "reason for the stock price move, earnings, guidance or company news"  # dito

GROUND_TRUTH_PATH = "data/ground_truth.json"
RESULTS_DIR = "data"

# Zahl der gleichzeitig laufenden LLM-Aufrufe. Je Anomalie fallen rund vierzehn
# Aufrufe an (zwei Antworten, zwei Klassifikationen, zehn RAGAS-Aufrufe); streng
# nacheinander abgearbeitet dauert ein Lauf ueber ein ganzes Portfolio Stunden,
# ohne dass die Rechenlast dies erforderte — die Zeit vergeht in der Wartezeit auf
# den Anbieter, nicht in lokaler Rechnung. Der Wert ist ueber die Umgebung
# einstellbar, damit ein Anbieter mit engem Anfragelimit ohne Codeaenderung wieder
# auf sequenzielle Abarbeitung (1) zurueckgestellt werden kann.
EVAL_MAX_WORKERS = int(os.environ.get("EVAL_MAX_WORKERS", "8"))


def _run_parallel(tasks):
    """Fuehrt eine Liste argumentloser Aufrufe nebenlaeufig aus und wartet auf alle.

    Die Aufgaben schreiben ihr Ergebnis selbst in das jeweilige Stichprobenobjekt;
    die Reihenfolge der Stichprobe bleibt dadurch unabhaengig davon, in welcher
    Reihenfolge die Antworten eintreffen. Faellt ein Aufruf mit einer Ausnahme aus,
    wird sie beim Einsammeln erneut ausgeloest — wie bei sequenzieller Abarbeitung.
    """
    if EVAL_MAX_WORKERS <= 1:
        for task in tasks:
            task()
        return
    with ThreadPoolExecutor(max_workers=EVAL_MAX_WORKERS) as pool:
        for future in [pool.submit(t) for t in tasks]:
            future.result()


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


def _eval_anomalies(analysis_data):
    """Anomalien, die in die Evaluation eingehen. Im Sparmodus (rag_config.SAVING_MODE)
    exakt die im Prompt gelistete Auswahl (select_prompt_anomalies) — dieselbe kleine,
    über den Cutoff ausgewogene Menge wie in den LLM-Tabs, damit ein RAGAS-Lauf
    denselben Bruchteil des Kontingents kostet. Im Vollmodus alle Einzeltitel-
    Anomalien (je Anomalie ein eigener Aufruf, siehe Modul-Docstring)."""
    if rag_config.SAVING_MODE:
        return select_prompt_anomalies(analysis_data)
    return _single_stock_anomalies(analysis_data)


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

    Nur Einzeltitel-Anomalien (siehe _eval_anomalies) — identisch zur
    Produktionslogik, im Sparmodus auf die im Prompt gelistete Auswahl beschränkt.
    Kontext-Retrieval ist eins zu eins die Produktionsfunktion (gleiche Query,
    gleiches top_k), damit die Evaluation misst, was der Nutzer tatsächlich als
    Kontext bekommt.
    """
    ground_truth = _load_ground_truth()
    samples = []
    for b in _eval_anomalies(analysis_data):
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
    single_data = {**(analysis_data or {}), "active_return_breaks": [sample["anomaly"]]}
    news_context = rag_pipeline.format_context_for_llm(sample["chunks"], max_tokens=2000)
    prompt = build_portfolio_prompt(single_data, news_context=news_context or None)

    client = make_llm_client()
    response = llm_chat_with_retry(
        client,
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
    single_data = {**(analysis_data or {}), "active_return_breaks": [sample["anomaly"]]}
    prompt = build_portfolio_prompt(single_data)

    client = make_llm_client()
    response = llm_chat_with_retry(
        client,
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

    `api_key` wird nicht mehr direkt verwendet (Provider-Auswahl läuft zentral über
    make_llm_client/LLM_PROVIDER) — der Parameter bleibt aus Kompatibilität zur
    bestehenden Aufrufsignatur erhalten.
    """
    try:
        client = make_llm_client()
        response = llm_chat_with_retry(
            client,
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
       classify_explanation) — für JEDE Anomalie der Stichprobe (im Sparmodus die
       Auswahl, sonst alle Einzeltitel-Anomalien), unabhängig vom Kontext-Status,
       weil Naive-LLM ohnehin nie Kontext braucht und der Fall "RAG fand keinen
       Kontext" selbst ein Vergleichsergebnis ist (RAG degradiert dann zu Naive).

    Rückgabe: {"samples": [...], "aggregate": {...}, "comparison": {...},
               "group_sizes": {...}, "config": {...}, "n_skipped": int,
               "timestamp": iso-str}
    """
    import numpy as np
    from ragas import evaluate, EvaluationDataset, SingleTurnSample, RunConfig
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithoutReference, LLMContextRecall
    from langchain_openai import ChatOpenAI

    samples = build_eval_samples(rag_pipeline, analysis_data)
    evaluable = [s for s in samples if s["status"] == "ok"]
    n_skipped = len(samples) - len(evaluable)

    # 1) Antworten erzeugen — RAG- UND Naive-Antwort für ALLE Anomalien. Die Aufrufe
    # sind voneinander unabhängig und laufen deshalb nebenläufig (EVAL_MAX_WORKERS).
    # RAG-Antwort ist bei status='no_context' identisch zur Naive-Antwort im
    # Prompt-Aufbau (news_context=None in beiden Fällen) — der Vergleich zeigt dann
    # konsequenterweise Konvergenz statt eines Fehlers.
    def _answer_tasks():
        for s in samples:
            yield lambda s=s: s.__setitem__(
                "rag_answer", generate_answer(rag_pipeline, analysis_data, s))
            yield lambda s=s: s.__setitem__(
                "naive_answer", generate_naive_answer(analysis_data, s))

    _run_parallel(list(_answer_tasks()))

    # 2) Symmetrische Hakem-Klassifikation beider Antworten (siehe classify_explanation).
    # Eigenständig fehlertolerant — ein einzelner Klassifikationsfehler bricht die
    # gesamte Auswertung nicht ab. Setzt Schritt 1 voraus und läuft deshalb erst
    # danach, in sich aber ebenfalls nebenläufig.
    def _classification_tasks():
        for s in samples:
            yield lambda s=s: s.__setitem__(
                "rag_classification",
                classify_explanation(s["question"], s["rag_answer"], api_key))
            yield lambda s=s: s.__setitem__(
                "naive_classification",
                classify_explanation(s["question"], s["naive_answer"], api_key))

    _run_parallel(list(_classification_tasks()))

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
            model=JUDGE_MODEL, base_url=llm_base_url(),
            api_key=llm_api_key(), temperature=0, max_tokens=4096,
            model_kwargs={"reasoning_effort": "low"},
        )
        # max_workers: rund zehn Judge-Aufrufe je Anomalie, die voneinander
        # unabhängig sind; sie laufen deshalb nebenläufig (siehe EVAL_MAX_WORKERS).
        # Die Wiederholungen (max_retries) fangen vereinzelte Anfragebegrenzungen des
        # Anbieters ab. bypass_n=True: openai/gpt-oss-120b (Reasoning-Modell, über
        # einen OpenAI-kompatiblen Endpunkt) lehnt n>1 ab ("'n' : number must be
        # at most 1"); ResponseRelevancy würde sonst mit n=strictness aufrufen und
        # scheitern — bypass_n erzwingt stattdessen einzelne Aufrufe.
        # timeout/max_retries/max_wait: mit reasoning_effort="low" beendet ein
        # einzelner Judge-Aufruf ueber OpenRouter (bezahltes Kontingent, kein
        # Minutenlimit) i. d. R. in unter einer Minute. Die frueher grosszuegigen
        # Werte (timeout=300, max_retries=8, max_wait=60) stammten aus einer Phase
        # mit strengem Minutenlimit und zogen einen einzelnen fehlerhaften Fall
        # minutenlang in die Laenge. Schlaegt ein Aufruf trotz der drei Versuche
        # fehl, liefert raise_exceptions=False fuer diese Metrik NaN, ohne den Lauf
        # abzubrechen.
        run_config = RunConfig(timeout=120, max_retries=3, max_wait=20,
                               max_workers=EVAL_MAX_WORKERS)
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

        # 3b) ZWEITER Durchgang: dieselbe Quellenbasis, aber die NAIVE Antwort als
        # Response. Damit wird die Belegbarkeit beider Bedingungen an derselben
        # Messlatte erhoben — genau das, was die Forschungsfrage zur Belegbarkeit
        # verlangt ("ob die angeführte Information in dieser Quelle tatsächlich
        # enthalten ist", einschließlich der Naive-Seite). Ohne diesen Durchgang
        # bliebe die Naive-Bedingung allein über die FORM ihrer Quellenangabe
        # beurteilt (siehe classify_explanation) — eine erfundene, aber konkret
        # formulierte Quelle wäre von einer echten nicht zu unterscheiden.
        #
        # Faithfulness = Quellendeckung der Naive-Antwort (bewusst NICHT "Faithfulness"
        # genannt: das Modell hat diesen Kontext nie gesehen, kann ihm also nicht
        # "treu" sein — siehe SOURCE_SUPPORT_* und implementierung_schritte.md).
        # ResponseRelevancy = Passung der Naive-Antwort zur Frage, symmetrisch zur
        # RAG-Seite erhoben, damit der Vergleich auch die Antwortfokussierung erfasst
        # (weicht die Antwort aus, bleibt sie unvollständig). ContextPrecision bleibt
        # weg — es bewertet das Retrieval, das es in der Naive-Bedingung nicht gibt.
        naive_dataset = EvaluationDataset(samples=[
            SingleTurnSample(
                user_input=s["question"],
                retrieved_contexts=s["contexts"],
                response=s["naive_answer"],
            )
            for s in evaluable
        ])
        naive_result = evaluate(
            naive_dataset, metrics=[Faithfulness(), ResponseRelevancy()],
            llm=judge, embeddings=embeddings,
            run_config=run_config, raise_exceptions=False, show_progress=False,
        )
        for s, score in zip(evaluable, naive_result.scores):
            s["_naive_ragas_score"] = score

    # 4) Ergebniszeilen — EINHEITLICH für alle Samples (RAGAS-Metriken bleiben None
    # außerhalb von 'evaluable', Vergleichsfelder sind für alle Samples befüllt).
    result_samples = []
    for s in samples:
        score = s.get("_ragas_score", {})
        naive_score = s.get("_naive_ragas_score", {})
        result_samples.append({
            "date": s["date"], "ticker": s["ticker"], "question": s["question"],
            "status": s["status"], "n_contexts": len(s["contexts"]),
            "post_cutoff": s["post_cutoff"], "has_reference": s["reference"] is not None,
            "faithfulness": _clean(score.get("faithfulness")),
            # Quellendeckung der Naive-Antwort gegen DIESELBE Quellenbasis
            # (siehe Durchgang 3b). Eigener Name, weil die Naive-Bedingung den
            # Kontext nie gesehen hat.
            "source_support_naive": _clean(naive_score.get("faithfulness")),
            "answer_relevancy": _clean(score.get("answer_relevancy")),
            "answer_relevancy_naive": _clean(naive_score.get("answer_relevancy")),
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
        # Naive-Bedingung an derselben Quellenbasis (siehe Durchgang 3b).
        "source_support_naive": _mean("source_support_naive"),
        "answer_relevancy_naive": _mean("answer_relevancy_naive"),
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
        # Antwortfokussierung (Response Relevancy) SYMMETRISCH je Bedingung: misst,
        # ob die Antwort die Frage trifft oder ausweicht/unvollständig bleibt —
        # keine Halluzinationsaussage (dafür source_support), sondern Antwortqualität.
        "answer_relevancy": {
            "rag": {
                "all": aggregate["answer_relevancy"],
                "pre_cutoff": _mean("answer_relevancy", cutoff_filter=False),
                "post_cutoff": _mean("answer_relevancy", cutoff_filter=True),
            },
            "naive": {
                "all": _mean("answer_relevancy_naive"),
                "pre_cutoff": _mean("answer_relevancy_naive", cutoff_filter=False),
                "post_cutoff": _mean("answer_relevancy_naive", cutoff_filter=True),
            },
        },
        # Belegbarkeit SYMMETRISCH: beide Bedingungen an derselben Quellenbasis
        # gemessen (siehe Durchgang 3b). Für die RAG-Bedingung ist das die bereits
        # berechnete Faithfulness — nicht neu erhoben, sondern wiederverwendet.
        "source_support": {
            "rag": {
                "all": aggregate["faithfulness"],
                "pre_cutoff": _mean("faithfulness", cutoff_filter=False),
                "post_cutoff": _mean("faithfulness", cutoff_filter=True),
            },
            "naive": {
                "all": _mean("source_support_naive"),
                "pre_cutoff": _mean("source_support_naive", cutoff_filter=False),
                "post_cutoff": _mean("source_support_naive", cutoff_filter=True),
            },
        },
        # Operationalisierung der in der Forschungsfrage genannten
        # "Halluzinationsrate": Anteil der NICHT durch die Quellenbasis gedeckten
        # Aussagen. Ausdrücklich KEINE Falschheitsaussage — eine Aussage kann
        # zutreffen und trotzdem nicht in den abgerufenen Dokumenten stehen
        # (siehe implementierung_schritte.md, Grenzen der Quellendeckung).
        "unsupported_claim_rate": {
            "rag": {
                "all": _complement(aggregate["faithfulness"]),
                "pre_cutoff": _complement(_mean("faithfulness", cutoff_filter=False)),
                "post_cutoff": _complement(_mean("faithfulness", cutoff_filter=True)),
            },
            "naive": {
                "all": _complement(_mean("source_support_naive")),
                "pre_cutoff": _complement(_mean("source_support_naive", cutoff_filter=False)),
                "post_cutoff": _complement(_mean("source_support_naive", cutoff_filter=True)),
            },
        },
    }

    config = {
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        # Anbieter mitschreiben: Modell und Parameter allein bestimmen die
        # Vergleichbarkeit zwar vollstaendig, aber bei einem Anbieterwechsel muss
        # nachvollziehbar bleiben, unter welcher Infrastruktur eine Messreihe
        # entstanden ist (siehe callbacks/naive_llm.py::LLM_PROVIDER).
        "llm_provider": LLM_PROVIDER,
        "eval_max_workers": EVAL_MAX_WORKERS,
        "embedding_model": rag_pipeline.embedder.model_name,
        "top_k": TOP_K,
        "metrics": ["faithfulness", "answer_relevancy", "context_precision"]
                   + (["context_recall"] if any(r.get("has_reference") for r in result_samples) else []),
        "ragas_version": _ragas_version(),
        "anomaly_prompt_safe_mode": ANOMALY_PROMPT_SAFE_MODE,
        "active_sources": [getattr(s, "name", "?") for s in rag_pipeline.sources],
        "generator_knowledge_cutoff": GENERATOR_KNOWLEDGE_CUTOFF,
        # Sparmodus mitschreiben: ein Ergebnisobjekt aus einem Sparmodus-Lauf
        # (kleine, ausgewogene Auswahl + Alpha Vantage aktiv) darf nicht mit den
        # für die Thesis zählenden Zahlen aus dem Vollmodus verwechselt werden.
        "saving_mode": bool(rag_config.SAVING_MODE),
        "saving_mode_anomaly_keys": [f"{s['date']}|{s['ticker']}" for s in samples],
        "specificity_categories": list(SPECIFICITY_CATEGORIES),
        "citation_categories": list(CITATION_CATEGORIES),
        "citation_verification_limitation": (
            "Der Hakem prueft nur die Formulierung (konkreter Name vs. vage Angabe), "
            "NICHT die reale Existenz einer genannten Quelle (keine Websuche)."
        ),
    }

    # Gruppengroessen: Eine Quote ist nur zusammen mit ihrer Bezugsmenge deutbar —
    # 100 % aus einer einzigen Anomalie und 100 % aus vierzig Anomalien sind im
    # Ergebnisobjekt sonst nicht zu unterscheiden. Zusaetzlich zur Gesamtzahl wird
    # je Gruppe die Zahl der AUSWERTBAREN Faelle gefuehrt (mit abgerufenem
    # Kontext), denn nur diese gehen in die Quellendeckung ein.
    def _group_size(cutoff_value):
        rows = [r for r in result_samples if r["post_cutoff"] is cutoff_value]
        return {
            "total": len(rows),
            "evaluable": sum(1 for r in rows if r["status"] == "ok"),
        }

    group_sizes = {
        "pre_cutoff": _group_size(False),
        "post_cutoff": _group_size(True),
        # Anomalien mit unlesbarem Datum: weder der einen noch der anderen Gruppe
        # zuzuordnen, daher separat ausgewiesen statt stillschweigend verteilt.
        "unknown": _group_size(None),
    }

    return {
        "samples": sorted(result_samples, key=lambda r: (r["date"] or "", r["ticker"] or "")),
        "aggregate": aggregate,
        "comparison": comparison,
        "group_sizes": group_sizes,
        "config": config,
        "n_skipped": n_skipped,
        "timestamp": datetime.now().isoformat(),
    }


def _complement(value):
    """1 - value, mit None-Durchreichung. Wandelt einen Deckungsgrad in die
    zugehörige Nicht-Deckungsquote um (siehe comparison.unsupported_claim_rate)."""
    return None if value is None else 1.0 - float(value)


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
