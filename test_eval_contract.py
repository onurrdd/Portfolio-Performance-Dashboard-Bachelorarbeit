#!/usr/bin/env python
"""
Schnittstellentest der Evaluationsschicht — OHNE LLM-Aufrufe.

Die Evaluation (rag/evaluation.py) ist bewusst nicht in die RAG-Pipeline
eingebaut, sondern greift von aussen auf sie zu. Damit bleibt sie von
Aenderungen am Retrieval unberuehrt, solange die benutzte Schnittstelle
stabil bleibt. Genau diese Annahme prueft dieser Test: er belegt, dass die
Beruehrungspunkte zwischen Evaluation, Pipeline, Anomalieerkennung und
Prompt-Erzeugung vorhanden sind und die erwartete Form liefern.

Geprueft werden vier Vertraege:
  [A] Import-Vertrag  — was rag/evaluation.py aus callbacks/naive_llm.py bezieht
  [B] Anomalie-Felder — die vier Schluessel, die die Evaluation je Anomalie liest
  [C] Pipeline        — die fuenf Beruehrungspunkte am RAGPipeline-Objekt
  [D] Ergebnisform    — die Schluessel der gespeicherten Auswertung

Der Test kostet nichts: er ruft weder das Generator- noch das Hakem-Modell auf
und laedt keine Quellen aus dem Netz. Ein Fehlschlag bedeutet, dass ein
Eingriff in die RAG-Schicht die Auswertung stillschweigend unbrauchbar
gemacht haette.

Nutzung: python test_eval_contract.py
"""
import sys
import os
import glob
import json
import inspect

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

from dotenv import load_dotenv
load_dotenv()

import numpy as np

SEP = "=" * 64
print(SEP)
print("Schnittstellentest der Evaluationsschicht (LLM-frei)")
print(SEP)


def fail(msg):
    print("FAIL " + msg)
    sys.exit(1)


# --- [A] Import-Vertrag ----------------------------------------------------
print()
print("[A] Import-Vertrag Evaluation <- Naive-LLM ...")
try:
    import callbacks.naive_llm as naive_llm
    from callbacks.naive_llm import (GENERATOR_MODEL, ANOMALY_PROMPT_SAFE_MODE,
                                     build_portfolio_prompt, llm_chat_with_retry,
                                     make_llm_client, llm_base_url, llm_api_key,
                                     LLM_PROVIDER)
    from rag.evaluation import (build_eval_samples, run_ragas_evaluation,
                                save_eval_results, GENERATOR_KNOWLEDGE_CUTOFF,
                                TOP_K, RETRIEVAL_QUERY)
    from rag import config as rag_config
    from rag.pipeline import RAGPipeline
except Exception as e:
    fail("Import: " + str(e))

if not isinstance(GENERATOR_MODEL, str) or not GENERATOR_MODEL:
    fail("GENERATOR_MODEL ist kein nichtleerer String")
if LLM_PROVIDER != "openrouter":
    fail("Unbekannter LLM_PROVIDER: " + str(LLM_PROVIDER))
if not isinstance(llm_base_url(), str) or "://" not in llm_base_url():
    fail("llm_base_url() liefert keine URL")
if len(str(GENERATOR_KNOWLEDGE_CUTOFF)) != 10:
    fail("GENERATOR_KNOWLEDGE_CUTOFF ist kein ISO-Datum")
print("OK  alle Symbole vorhanden; Anbieter=" + LLM_PROVIDER
      + ", Modell=" + GENERATOR_MODEL)
print("OK  Wissensschnitt=" + GENERATOR_KNOWLEDGE_CUTOFF
      + ", top_k=" + str(TOP_K))

# Der Prompt-Bau muss beide Bedingungen bedienen: mit und ohne abgerufenen
# Kontext. Faellt eine der beiden Formen weg, ist der Vergleich nicht mehr
# symmetrisch — beide Antworten muessen aus DEMSELBEN Template stammen.
# rag_config.SAVING_MODE wird nur zur Laufzeit abgeschaltet (die Datei bleibt
# unveraendert), damit die Sparmodus-Auswahl die Beispielanomalie nicht verwirft.
rag_config.SAVING_MODE = False

FAKE_ANOMALY = {
    "date": "2024-01-25",
    "responsible_ticker": "TSLA",
    "ticker_own_return_pct": -12.13,
    "concentration": "Hisseye özgü",
}
FAKE_DATA = {
    "positions": [],
    "metrics": {"total_return": 0.1, "sortino": 0.5},
    "rolling_sharpe": {"current": 0.4},
    "benchmark": {"total_return": 0.08, "sortino": 0.6, "sharpe_current": 0.5},
    "active_return_breaks": [FAKE_ANOMALY],
}
MARKER = "CONTEXT-MARKER-XYZ"
try:
    p_naive = build_portfolio_prompt(FAKE_DATA)
    p_rag = build_portfolio_prompt(FAKE_DATA, news_context=MARKER)
except Exception as e:
    fail("build_portfolio_prompt: " + str(e))
if not isinstance(p_naive, str) or not p_naive:
    fail("build_portfolio_prompt ohne Kontext liefert keinen Text")
if MARKER not in p_rag:
    fail("uebergebener Kontext erscheint nicht im Prompt")
if MARKER in p_naive:
    fail("Naive-Prompt enthaelt Kontext, obwohl keiner uebergeben wurde")
if "TSLA" not in p_naive:
    fail("Anomalie-Ticker erscheint nicht im Prompt")
print("OK  build_portfolio_prompt bedient beide Bedingungen "
      "(mit/ohne Kontext); Ticker erreicht den Prompt")
print("OK  ANOMALY_PROMPT_SAFE_MODE=" + str(ANOMALY_PROMPT_SAFE_MODE)
      + ", Client-/Retry-Funktionen aufrufbar="
      + str(callable(llm_chat_with_retry) and callable(make_llm_client)
            and callable(llm_api_key)))


# --- [B] Anomalie-Felder ---------------------------------------------------
print()
print("[B] Anomalie-Felder aus der Anomalieerkennung ...")
# Die Anomalieerkennung gehoert zum unveraenderlichen Kern der Anwendung; sie
# wird hier nicht ausgefuehrt (das erforderte Kursdaten aus dem Netz), sondern
# es wird geprueft, dass sie die vier von der Evaluation gelesenen Schluessel
# ueberhaupt erzeugt.
try:
    import utils.anomaly as anomaly_mod
    src = inspect.getsource(anomaly_mod)
except Exception as e:
    fail("utils.anomaly nicht lesbar: " + str(e))

REQUIRED_FIELDS = ("date", "responsible_ticker", "ticker_own_return_pct",
                   "concentration")
missing = [f for f in REQUIRED_FIELDS
           if ("'%s'" % f) not in src and ('"%s"' % f) not in src]
if missing:
    fail("Anomalieerkennung erzeugt diese Felder nicht mehr: " + str(missing))
# Die Evaluation waehlt ausschliesslich einzeltitelgetriebene Anomalien aus;
# der Filterwert muss zeichengleich sein, sonst bleibt die Stichprobe leer.
if "Hisseye özgü" not in src:
    fail("Konzentrationslabel 'Hisseye özgü' kommt in utils/anomaly.py nicht vor")
print("OK  vier Felder vorhanden: " + ", ".join(REQUIRED_FIELDS))
print("OK  Filterwert 'Hisseye özgü' stimmt mit der Anomalieerkennung ueberein")

# Die Stichprobenauswahl selbst laesst sich ohne Kursdaten pruefen: ein
# einzelner, korrekt etikettierter Datensatz muss durchkommen.
from rag.evaluation import _single_stock_anomalies
picked = _single_stock_anomalies(FAKE_DATA)
if len(picked) != 1:
    fail("Auswahl einzeltitelgetriebener Anomalien liefert "
         + str(len(picked)) + " statt 1")
dropped = _single_stock_anomalies(
    {"active_return_breaks": [{**FAKE_ANOMALY, "concentration": "Dağılmış"}]})
if dropped:
    fail("breit gestreute Anomalie wird nicht ausgefiltert")
print("OK  Auswahl nimmt einzeltitelgetriebene Anomalien und verwirft gestreute")


# --- [C] Pipeline-Beruehrungspunkte ---------------------------------------
print()
print("[C] Beruehrungspunkte am RAGPipeline-Objekt ...")
try:
    pipeline = RAGPipeline()
except Exception as e:
    fail("RAGPipeline() nicht konstruierbar: " + str(e))

# (1) retrieve_for_anomaly — Signatur
sig = inspect.signature(pipeline.retrieve_for_anomaly)
for param in ("query", "anomaly", "top_k"):
    if param not in sig.parameters:
        fail("retrieve_for_anomaly ohne Parameter " + param)
print("OK  (1) retrieve_for_anomaly" + str(sig))

# (2) format_context_for_llm — Signatur
sig2 = inspect.signature(pipeline.format_context_for_llm)
for param in ("retrieved_chunks", "max_tokens"):
    if param not in sig2.parameters:
        fail("format_context_for_llm ohne Parameter " + param)
print("OK  (2) format_context_for_llm" + str(sig2))

# (3) embedder — Modellname und Vektorform. RAGAS' ResponseRelevancy wickelt
#     genau dieses Objekt ein, statt ein zweites Modell zu laden; dafuer muessen
#     Einzel- und Stapelaufruf formgleich antworten.
if not isinstance(getattr(pipeline.embedder, "model_name", None), str):
    fail("embedder.model_name fehlt oder ist kein String")
dim = pipeline.embedder.embedding_dim
v_one = np.asarray(pipeline.embedder.embed_text("Testsatz zur Einbettung"))
v_many = np.asarray(pipeline.embedder.embed_text(["Satz eins", "Satz zwei"]))
if v_one.ndim != 1 or v_one.shape[0] != dim:
    fail("embed_text(str) liefert Form " + str(v_one.shape)
         + ", erwartet (" + str(dim) + ",)")
if v_many.ndim != 2 or v_many.shape != (2, dim):
    fail("embed_text(list) liefert Form " + str(v_many.shape)
         + ", erwartet (2, " + str(dim) + ")")
print("OK  (3) embedder=" + pipeline.embedder.model_name
      + ", Dimension=" + str(dim) + ", Einzel- und Stapelaufruf formgleich")

# (4) sources — die Anbieterliste wird in die Ergebnisdatei geschrieben
names = [getattr(s, "name", None) for s in pipeline.sources]
if not names or any(n is None for n in names):
    fail("mindestens eine Quelle ohne Attribut name")
print("OK  (4) " + str(len(names)) + " Quelle(n): " + ", ".join(names))

# (5) Retrieval und Formatierung im Zusammenspiel — auf dem bestehenden Index,
#     ohne Netzzugriff. Ein leeres Ergebnis ist zulaessig (das Retrieval kennt
#     keinen Fallback), nur die Form muss stimmen.
chunks = pipeline.retrieve_for_anomaly(RETRIEVAL_QUERY, FAKE_ANOMALY, top_k=TOP_K)
if not isinstance(chunks, list):
    fail("retrieve_for_anomaly liefert keine Liste")
for c in chunks:
    if "text" not in c:
        fail("abgerufener Chunk ohne Feld text")
ctx = pipeline.format_context_for_llm(chunks, max_tokens=2000)
if not isinstance(ctx, str):
    fail("format_context_for_llm liefert keinen String")
if chunks and not ctx.strip():
    fail("Treffer vorhanden, aber formatierter Kontext ist leer")
print("OK  (5) Retrieval liefert " + str(len(chunks))
      + " Chunk(s) mit Textfeld; Formatierung ergibt "
      + str(len(ctx)) + " Zeichen")


# --- [D] Ergebnisform ------------------------------------------------------
print()
print("[D] Form der gespeicherten Auswertung ...")
files = sorted(glob.glob(os.path.join("data", "ragas_eval_*.json")))
if not files:
    print("SKIP  keine Auswertungsdatei in data/ vorhanden — "
          "Formpruefung uebersprungen")
else:
    path = files[-1]
    with open(path, "r", encoding="utf-8") as fh:
        res = json.load(fh)
    for key in ("samples", "aggregate", "comparison", "config", "n_skipped",
                "group_sizes"):
        if key not in res:
            fail("Ergebnisdatei ohne Block " + key)
    for key in ("source_support", "unsupported_claim_rate", "specificity_pct",
                "named_citation_pct"):
        if key not in res["comparison"]:
            fail("Vergleichsblock ohne Kennzahl " + key)
    for cond in ("rag", "naive"):
        if cond not in res["comparison"]["source_support"]:
            fail("Quellendeckung ohne Bedingung " + cond)
        for grp in ("all", "pre_cutoff", "post_cutoff"):
            if grp not in res["comparison"]["source_support"][cond]:
                fail("Quellendeckung " + cond + " ohne Gruppe " + grp)
    for grp in ("pre_cutoff", "post_cutoff", "unknown"):
        if grp not in res["group_sizes"]:
            fail("Gruppengroessen ohne " + grp)
        if "evaluable" not in res["group_sizes"][grp]:
            fail("Gruppe " + grp + " ohne Zahl der auswertbaren Faelle")
    if "llm_provider" not in res["config"]:
        fail("Konfigurationsabzug ohne llm_provider")
    if res["samples"]:
        s0 = res["samples"][0]
        for key in ("date", "ticker", "status", "post_cutoff",
                    "source_support_naive", "faithfulness",
                    "specificity_rag", "specificity_naive",
                    "citation_rag", "citation_naive"):
            if key not in s0:
                fail("Einzelergebnis ohne Feld " + key)
    print("OK  " + os.path.basename(path) + ": alle Bloecke vorhanden "
          "(symmetrische Quellendeckung, Gruppengroessen, Wissensschnitt)")

print()
print(SEP)
print("ALLE TESTS BESTANDEN")
print(SEP)
