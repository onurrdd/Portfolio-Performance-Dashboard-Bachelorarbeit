import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Optional
import pandas as pd
from groq import Groq, RateLimitError as GroqRateLimitError
try:
    from openai import RateLimitError as OpenAIRateLimitError, APITimeoutError as OpenAIAPITimeoutError
except ImportError:
    OpenAIRateLimitError = GroqRateLimitError  # Platzhalter, falls openai-Paket fehlt
    OpenAIAPITimeoutError = GroqRateLimitError
import dash_bootstrap_components as dbc
from dash import Input, Output, State, html, no_update

# Leichtgewichtig (nur sqlite3, keine Torch/Embedding-Importe) — unabhängig vom lazy
# RAGProvider, daher hier gefahrlos verwendbar, ohne die schwere Pipeline zu laden.
from rag.cache import NewsCache
from rag.config import DEFAULT_CONFIG
# Als Modul importiert (nicht die einzelnen Namen): der Sparmodus-Schalter wird in
# eval_probe.py / test_eval_contract.py zur Laufzeit umgesetzt und muss über das
# Attribut gelesen werden, damit die Umsetzung greift (siehe rag/config.py).
from rag import config as rag_config

# --- LLM-Anbieter (austauschbar) ---
# Groq und OpenRouter bedienen denselben offenen Modellnamen (openai/gpt-oss-120b)
# über dieselbe OpenAI-kompatible REST-Schnittstelle; nur Endpunkt und Schlüssel
# unterscheiden sich. Die Umschaltung erfolgt zentral über eine Umgebungsvariable,
# damit kein Aufrufer (Naive-LLM, RAG-LLM, Evaluation) angepasst werden muss, wenn
# das Kontingent eines Anbieters erschöpft ist — betrifft NICHT die AI-Risk-Analyse
# (callbacks/ai_analysis.py), die unverändert an Groq gebunden bleibt.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "groq").lower()
_PROVIDER_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}
_PROVIDER_API_KEY_ENV = {
    "groq": "GROQ_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def llm_base_url() -> str:
    return _PROVIDER_BASE_URLS[LLM_PROVIDER]


def llm_api_key() -> str:
    return os.environ.get(_PROVIDER_API_KEY_ENV[LLM_PROVIDER], "")


# Hartes Request-Timeout (Sekunden) für jeden LLM-Aufruf. Ein Reasoning-Modell
# braucht für die volle Antwort mitunter mehrere Minuten; darüber hinaus deutet
# Stillstand auf eine hängende Verbindung hin und der Aufruf soll mit Fehler
# abbrechen, statt einen Lauf unbegrenzt zu blockieren.
_REQUEST_TIMEOUT = 300.0


def make_llm_client():
    """Liefert einen Chat-Completions-Client für den aktiven Anbieter (LLM_PROVIDER).

    Mit hartem Request-Timeout: OpenRouter kann eine Antwort beliebig lange
    verzögern; ohne Timeout blockiert der Aufruf unbegrenzt und ein
    Evaluationslauf hängt statt mit einem Fehler abzubrechen (den
    llm_chat_with_retry als Nicht-429 durchreicht)."""
    if LLM_PROVIDER == "groq":
        return Groq(api_key=llm_api_key(), timeout=_REQUEST_TIMEOUT)
    from openai import OpenAI
    return OpenAI(base_url=llm_base_url(), api_key=llm_api_key(), timeout=_REQUEST_TIMEOUT)

# Sprache der LLM-ANTWORT: "de", "en" oder "tr".
# Die Prompts selbst sind IMMER Englisch (Projektregel, siehe CLAUDE.md).
RESPONSE_LANGUAGE = "tr"

# Generierungsmodell für Naive-LLM UND RAG-LLM (identisch, siehe callbacks/rag.py::MODEL).
# Ursprünglich llama-3.3-70b-versatile; von Groq zwischenzeitlich aus dem Angebot
# entfernt (API liefert 404 "model does not exist"). Ersatz: openai/gpt-oss-120b
# (ein Reasoning-Modell — benötigt ausreichend max_tokens, da vor der Antwort ein
# interner Denkschritt Tokens verbraucht; 4096 hat sich als ausreichend erwiesen).
GENERATOR_MODEL = "openai/gpt-oss-120b"

# Wissensstand des Generator-Modells (openai/gpt-oss-120b), laut offizieller Model
# Card ca. Juni 2024 (Quelle: OpenAI gpt-oss Model Card, 05.08.2025; einzelne
# Sekundärquellen nennen Mai/Juli 2024 — Datum daher als ungefähr behandeln und im
# Thesistext entsprechend relativieren). Dient der Vor/Nach-Cutoff-Etikettierung der
# Anomalien, nicht als exakte Wahrheitsgrenze. Eigenschaft des Modells, daher hier
# neben GENERATOR_MODEL; rag/evaluation.py importiert von hier.
GENERATOR_KNOWLEDGE_CUTOFF = "2024-06-01"

# Antwortbudget für BEIDE Bedingungen (Naive-LLM und RAG-LLM). Modell UND Parameter
# müssen über beide Bedingungen identisch sein — sonst wäre ein beobachteter
# Unterschied nicht mehr allein dem Retrieval zuzuschreiben. Der Wert ist durch das
# Minuten-Token-Limit des Anbieters (Groq Free-Tier, 8000 TPM: prompt_tokens +
# max_tokens zusammen) nach oben begrenzt; eine unabgeschnittene Antwort wird daher
# über die Anzahl der im Prompt gelisteten Ereignisse gesteuert
# (PROMPT_ANOMALY_CAP_PER_TICKER), nicht über ein großes Budget.
# effective_generation_max_tokens() (unten) ist der tatsächlich verwendete Wert — dort
# fließt auch der Sparmodus (rag_config.SAVING_MODE) ein; callbacks/rag.py ruft dieselbe
# Funktion auf, damit beide Bedingungen stets denselben Wert benutzen.
GENERATION_MAX_TOKENS = 4096

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

# Kleineres Antwortbudget für den Sparmodus (rag_config.SAVING_MODE): dort stehen nur
# wenige Ereignisse im Prompt, entsprechend weniger Antworttext ist nötig, und ein
# einzelner Aufruf reserviert so wenig wie möglich vom Tages-Token-Kontingent (TPD).
# Nur das LLM-Budget wohnt hier; der Modus-Schalter selbst steht in rag/config.py.
SAVING_MODE_MAX_TOKENS = 1536

NO_ANOMALY_PLACEHOLDER = "(No single-stock anomaly events were detected for this portfolio.)"

PROMPT_ANOMALY_SAFETY_INSTRUCTION = """
Answer ONLY if you have specific, verifiable knowledge of a real, documented reason (e.g. an earnings report, guidance change, product announcement, analyst rating action, regulatory or legal news, M&A news, or a specific macroeconomic event) known to have affected this exact ticker on this exact date. If you are not confident about the specific cause for a given event, explicitly write "No reliable specific cause known for this date" instead of guessing. Do not invent a plausible-sounding explanation, and do not substitute general market or sector commentary for a specific reason you are not sure about."""

# Anzeigesprache der Anomalie-Tabelle (unabhängig von RESPONSE_LANGUAGE): "tr" oder "de"
TABLE_LANGUAGE = "de"

TABLE_HEADERS = {
    "tr": ['#', 'Tarih', 'Gerçekleşen', 'Beklenen', 'Sürpriz', 'Sürpriz (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Sorumlu Ticker', 'Hisse Getirisi',
           'Hisse Sürprizi', 'Hisse (MAD-z)', 'Uyarı', 'Kaynak'],
    "de": ['#', 'Datum', 'Tatsächlich', 'Erwartet', 'Überraschung', 'Überraschung (MAD-z)',
           'Benchmark (MAD-z)', 'β', 'Verantwortlicher Ticker', 'Titel-Rendite',
           'Titel-Residuum', 'Titel (MAD-z)', 'Warnung', 'Quelle'],
}

CONCENTRATION_LABELS = {
    "tr": {"Hisseye özgü": "Hisseye özgü", "Dağılmış": "Dağılmış"},
    "de": {"Hisseye özgü": "Einzeltitel", "Dağılmış": "Verteilt"},
}

TABLE_TEXT = {
    "tr": {"no_breaks": "Anomali günü tespit edilmedi.", "not_attributable": "— (atfedilemez)", "dash": "—",
           "source_count": "{n} kaynak"},
    "de": {"no_breaks": "Kein Anomalietag festgestellt.", "not_attributable": "— (nicht zuordenbar)", "dash": "—",
           "source_count": "{n} Quelle(n)"},
}


# Obergrenze der im gepoolten Prompt gelisteten Ereignisse, je verantwortlichem Titel.
# Der Prompt verlangt eine eigene Erklärung je Ereignis, die benötigte Antwortlänge wächst
# also linear mit der Ereigniszahl, während das Antwortbudget durch das Minutenlimit des
# Anbieters hart begrenzt ist. Die Auswahl je Titel (statt global) verhindert, dass eine
# dominante Position die Liste monopolisiert. Auswahlkriterium ist die standardisierte
# Abweichung |surprise_mad_z| — dasselbe Maß, das die Anomalie überhaupt auslöst.
# Gilt NUR für den gepoolten Prompt der Anwendung; die Evaluation (rag/evaluation.py)
# baut je Anomalie einen eigenen Aufruf und wertet daher ALLE Anomalien aus.
PROMPT_ANOMALY_CAP_PER_TICKER = 10


def single_stock_anomalies(analysis_data):
    """Einzeltitel-Anomalien (concentration == 'Hisseye özgü' mit verantwortlichem Ticker)."""
    return [
        b for b in (analysis_data or {}).get('active_return_breaks', [])
        if b.get('concentration') == 'Hisseye özgü' and b.get('responsible_ticker')
    ]


def _anomaly_key(b):
    """Stabiler Schlüssel 'YYYY-MM-DD|TICKER' einer Anomalie (für SAVING_MODE_PINNED
    und den Hinweistext)."""
    return f"{str(b.get('date'))[:10]}|{b.get('responsible_ticker')}"


def _tickers_with_cached_source(single_stock):
    """Teilmenge der Anomalie-Schlüssel, für die bereits Quellen im Anomaliefenster
    im Cache liegen (dieselbe Abfrage wie die 'Kaynak'-Spalte der Anomalietage-
    Tabelle). Ein Cache-Zugriff für die ganze Liste, nicht einer pro Anomalie."""
    cache = NewsCache(DEFAULT_CONFIG.db_path)
    window = DEFAULT_CONFIG.anomaly_window_days
    window_after = DEFAULT_CONFIG.anomaly_window_days_after
    have = set()
    for b in single_stock:
        ticker = b.get('responsible_ticker')
        date_str = b.get('date')
        if not (ticker and date_str):
            continue
        try:
            day = datetime.fromisoformat(str(date_str)[:10])
        except ValueError:
            continue
        if cache.get_articles(ticker, day, window, window_days_after=window_after):
            have.add(_anomaly_key(b))
    return have


def select_saving_mode_anomalies(single_stock):
    """Sparmodus-Auswahl (rag_config.SAVING_MODE): eine kleine, über den Knowledge-
    Cutoff ausgewogene Teilmenge der Einzeltitel-Anomalien.

    SAVING_MODE_PINNED gesetzt → genau diese Schlüssel (chronologisch). Sonst
    automatisch: Anomalien werden am GENERATOR_KNOWLEDGE_CUTOFF in zwei Gruppen
    (vor/nach) geteilt; je Gruppe SAVING_MODE_ANOMALY_COUNT // 2 Fälle, bei
    Unterdeckung füllt die andere Gruppe auf. Sortierschlüssel je Gruppe:
    (keine gecachte Quelle, -|surprise_mad_z|, Datum) — Fälle mit bereits
    vorhandener Quelle zuerst, dann die stärkste Abweichung. Die Cache-Lage ist
    hier nur PRÄFERENZ, kein Ausschluss: eine post-Cutoff-Anomalie ohne Quelle
    muss wählbar bleiben, sonst würde sie nie indiziert und bekäme nie eine.
    """
    pinned = set(rag_config.SAVING_MODE_PINNED)
    if pinned:
        return sorted((b for b in single_stock if _anomaly_key(b) in pinned),
                      key=lambda b: str(b['date']))

    count = rag_config.SAVING_MODE_ANOMALY_COUNT
    have_source = _tickers_with_cached_source(single_stock)

    def _sort_key(b):
        return (_anomaly_key(b) not in have_source,
                -abs(b.get('surprise_mad_z') or 0),
                str(b.get('date')))

    pre, post = [], []
    for b in single_stock:
        (post if str(b.get('date'))[:10] > GENERATOR_KNOWLEDGE_CUTOFF else pre).append(b)
    pre.sort(key=_sort_key)
    post.sort(key=_sort_key)

    half = count // 2
    selected = pre[:half] + post[:half]
    # Unterdeckung einer Seite aus der anderen auffüllen (Reihenfolge bleibt gewahrt).
    if len(selected) < count:
        rest = [b for b in pre[half:] + post[half:] if b not in selected]
        selected += rest[:count - len(selected)]
    return sorted(selected, key=lambda b: str(b['date']))


def select_prompt_anomalies(analysis_data, cap_per_ticker=PROMPT_ANOMALY_CAP_PER_TICKER):
    """Auswahl der zu erklärenden Einzeltitel-Anomalien: je verantwortlichem Titel die
    `cap_per_ticker` stärksten nach |surprise_mad_z|, Rückgabe chronologisch sortiert.

    Anwendung UND rag/evaluation.py bauen je Anomalie einen eigenen Aufruf; diese
    Funktion legt fest, welche Anomalien das sind (in beiden Bedingungen dieselben).

    Im Sparmodus (rag_config.SAVING_MODE) wird die normale Kappung übersprungen und
    stattdessen select_saving_mode_anomalies genutzt — siehe dort.
    """
    single_stock = single_stock_anomalies(analysis_data)
    if rag_config.SAVING_MODE:
        return select_saving_mode_anomalies(single_stock)
    if cap_per_ticker is None:
        return single_stock

    by_ticker = {}
    for b in single_stock:
        by_ticker.setdefault(b['responsible_ticker'], []).append(b)

    selected = []
    for events in by_ticker.values():
        events = sorted(events, key=lambda b: abs(b.get('surprise_mad_z') or 0), reverse=True)
        selected.extend(events[:cap_per_ticker])
    return sorted(selected, key=lambda b: str(b['date']))


def prompt_anomaly_coverage_note(analysis_data):
    """Hinweistext, falls der Prompt nicht alle Einzeltitel-Anomalien listet ('' sonst).

    Wird in BEIDEN Bedingungen unter der Antwort angezeigt: eine Antwort über eine
    Ereignisauswahl darf nicht wie eine Antwort über alle Ereignisse aussehen.
    """
    total = len(single_stock_anomalies(analysis_data))
    selected = select_prompt_anomalies(analysis_data)
    shown = len(selected)
    # Im Sparmodus IMMER anzeigen (auch wenn zufällig shown == total): der Hinweis
    # ist die Absicherung dagegen, dass ein vergessener Sparmodus die für die
    # Thesis zählenden Zahlen unbemerkt verfälscht. Die Schlüssel werden gelistet,
    # damit sie sich direkt nach SAVING_MODE_PINNED (rag/config.py) kopieren lassen.
    if rag_config.SAVING_MODE:
        keys = ", ".join(_anomaly_key(b) for b in selected) or "—"
        return (f" · SPARMODUS aktiv: Prompt umfasst {shown} von {total} "
                f"Einzeltitel-Anomalien (über den Knowledge-Cutoff ausgewogen; "
                f"siehe rag_config.SAVING_MODE). Auswahl: {keys}.")
    if shown >= total:
        return ""
    return (f" · Prompt umfasst {shown} von {total} Einzeltitel-Anomalien "
            f"(je Titel die {PROMPT_ANOMALY_CAP_PER_TICKER} stärksten nach MAD-z).")


def _build_anomaly_list_text(analysis_data):
    """Baut die Anomalie-Liste (Teil 2 des Prompts) aus den einzeltitelbezogenen Ereignissen."""
    selected = select_prompt_anomalies(analysis_data)
    if not selected:
        return NO_ANOMALY_PLACEHOLDER
    lines = []
    for b in selected:
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
            "The news items below are grouped by event, each under its own "
            "'--- Event: ... ---' heading. Use a group's items ONLY to explain that "
            "group's event. If a group says no sources were retrieved, or its items are "
            "not genuinely relevant, say so explicitly instead of forcing a connection "
            "or borrowing another event's sources.\n"
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


# Wartezeit-Hinweis in der Fehlermeldung des Anbieters — Groq gibt je nach Limit-Typ
# unterschiedliche Formate zurück: "try again in 12.5s" (Minutenlimit, TPM) oder
# "try again in 26m18.96s" (Tageslimit, TPD). Alle drei Einheiten sind optional, damit
# beide Formate (und ein möglicher "1h2m3s"-Fall) mit derselben Regex erfasst werden.
_RETRY_AFTER_RE = re.compile(
    r"try again in (?:(?P<h>\d+)h)?(?:(?P<m>\d+)m)?(?:(?P<s>[\d.]+)s)?", re.IGNORECASE)
_RETRY_MAX_ATTEMPTS = 3
_RETRY_FALLBACK_WAIT = 45.0
# Über dieser Wartezeit lohnt sich kein Retry mehr INNERHALB desselben UI-Klicks — das
# deutet auf ein Tages- statt ein Minutenlimit hin (TPD statt TPM). Retrying würde den
# Klick minutenlang blockieren und am Ende doch fehlschlagen; der Fehler wird stattdessen
# sofort weitergereicht (siehe _run_naive_analysis/_run_rag_analysis: dort wird er als
# Alert angezeigt, nicht verschluckt).
_RETRY_MAX_WAIT = 90.0


def _parse_retry_after(message: str) -> Optional[float]:
    match = _RETRY_AFTER_RE.search(message)
    if not match or not any(match.groups()):
        return None
    h, m, s = (float(match.group(g) or 0) for g in ("h", "m", "s"))
    return h * 3600 + m * 60 + s


def llm_chat_with_retry(client, **kwargs):
    """LLM-Aufruf mit Wiederholung ausschließlich bei KURZEN Minutenlimit-Fehlern (429).

    Der Vergleichs-Tab löst Naive- und RAG-Analyse im selben Moment aus; beide Aufrufe
    fallen damit in dasselbe Minutenfenster des Anbieters und der zweite läuft ohne
    Wiederholung in einen Rate-Limit-Fehler. Andere Fehler (u. a. 413, oder ein 429 mit
    langer Wartezeit — Tageslimit) werden NICHT wiederholt — ein erneuter Aufruf würde
    nur Zeit verbrauchen, ohne das Ergebnis zu ändern.
    """
    for attempt in range(_RETRY_MAX_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except (GroqRateLimitError, OpenAIRateLimitError) as e:
            wait = _parse_retry_after(str(e))
            if wait is None:
                wait = _RETRY_FALLBACK_WAIT
            if wait > _RETRY_MAX_WAIT or attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            time.sleep(wait + 1)
        except OpenAIAPITimeoutError:
            # Request-Timeout (siehe _REQUEST_TIMEOUT): eine hängende Verbindung
            # ist oft vorübergehend — einmal kurz warten und neu versuchen, aber
            # beim letzten Versuch durchreichen, damit der Lauf nicht ewig läuft.
            if attempt == _RETRY_MAX_ATTEMPTS - 1:
                raise
            time.sleep(5)


def effective_generation_max_tokens():
    """Tatsächlich verwendetes Antwortbudget — im Sparmodus kleiner, weil dort nur
    wenige Ereignisse im Prompt stehen und entsprechend weniger Antworttext nötig ist.
    EINE Funktion für BEIDE Bedingungen (Naive/RAG greifen beide hierüber zu), damit
    der Parameter zwischen ihnen nie auseinanderläuft — dieselbe Garantie wie bei
    GENERATION_MAX_TOKENS selbst (siehe dortiger Kommentar)."""
    return SAVING_MODE_MAX_TOKENS if rag_config.SAVING_MODE else GENERATION_MAX_TOKENS


def single_anomaly_data(analysis_data, anomaly):
    """analysis_data-Kopie, deren Anomalieliste auf genau EINE Anomalie beschränkt ist.

    Damit baut build_portfolio_prompt einen Prompt, der nur dieses eine Ereignis
    listet — dieselbe Ein-Anomalie-Einschränkung wie in rag/evaluation.py, jetzt
    auch im Anwendungs-Tab. Der Effekt: Anwendung (Demonstration) und RAGAS
    (Evaluation) laufen über dieselbe Ein-Ereignis-Mechanik, die pooled-Variante
    entfällt."""
    return {**(analysis_data or {}), "active_return_breaks": [anomaly]}


def _anomaly_llm_answer(analysis_data, anomaly, news_context=None):
    """Ein LLM-Aufruf für EINE Anomalie. `news_context=None` → Naive-Bedingung,
    sonst RAG-Bedingung. Einziger Unterschied zwischen beiden bleibt der Kontext."""
    prompt = build_portfolio_prompt(single_anomaly_data(analysis_data, anomaly),
                                    news_context=news_context or None)
    response = llm_chat_with_retry(
        make_llm_client(),
        messages=[{"role": "user", "content": prompt}],
        model=GENERATOR_MODEL,
        max_tokens=effective_generation_max_tokens(),
    )
    return response.choices[0].message.content


def _anomaly_answer_cards(analysis_data, answer_fn, title):
    """Baut je Einzeltitel-Anomalie eine Ergebniskarte. `answer_fn(anomaly)` liefert
    den Antworttext. Die Aufrufe laufen nebenläufig (wie in rag/evaluation.py) —
    vier LLM-Aufrufe sind sonst rein sequenzielle Wartezeit."""
    anomalies = select_prompt_anomalies(analysis_data)
    if not anomalies:
        return dbc.Alert("Keine Einzeltitel-Anomalien zu erklären.", color="info")

    results = [None] * len(anomalies)

    def _task(i, b):
        try:
            results[i] = _anomaly_answer_heading(b), answer_fn(b)
        except Exception as e:  # ein fehlgeschlagener Aufruf kippt nicht die ganze Karte
            results[i] = _anomaly_answer_heading(b), f"[Fehler: {e}]"

    with ThreadPoolExecutor(max_workers=min(len(anomalies), 8)) as pool:
        for fut in [pool.submit(_task, i, b) for i, b in enumerate(anomalies)]:
            fut.result()

    cards = [
        dbc.Card([dbc.CardBody([
            html.H6(heading, className="mb-2", style={'color': '#58a6ff'}),
            html.Div(text, style={'whiteSpace': 'pre-wrap', 'lineHeight': '1.6',
                                  'fontSize': '0.95rem', 'color': '#f0f6fc'}),
        ])], className="card-custom", style={'marginBottom': '10px'})
        for heading, text in results
    ]
    return html.Div([
        html.H5(title, className="mb-3", style={'color': '#f0f6fc'}),
        *cards,
        html.Small(
            f"Erstellt mit {LLM_PROVIDER} · je Anomalie ein eigener Aufruf"
            + prompt_anomaly_coverage_note(analysis_data),
            className="text-muted"),
    ])


def _anomaly_answer_heading(b):
    own_return = b.get('ticker_own_return_pct', 0) or 0
    direction = "rose" if own_return >= 0 else "fell"
    return f"{b.get('responsible_ticker')} {direction} {own_return:+.2f}% on {b.get('date')}"


def _run_naive_analysis(analysis_data):
    """Naive-LLM-Analyse: je Einzeltitel-Anomalie ein Aufruf OHNE Nachrichten-Kontext.

    Gemeinsame Logik für den Naive-LLM-Tab UND den Vergleichs-Tab.
    """
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar. Bitte füge zuerst Positionen hinzu.", color="warning")
    try:
        return _anomaly_answer_cards(
            analysis_data,
            lambda b: _anomaly_llm_answer(analysis_data, b, news_context=None),
            "Naive-LLM — Erklärung je Anomalie")
    except Exception as e:
        return dbc.Alert([
            html.H5("Fehler bei der Naive-LLM-Analyse", className="mb-2"),
            html.P(f"Fehlerdetails: {str(e)}"),
        ], color="danger")


def _naive_prompt_component(analysis_data):
    """Debug-Anzeige der Naive-LLM-Prompts (ohne Nachrichten-Kontext) — je Anomalie
    ein Prompt, mit Trenner dazwischen."""
    if not analysis_data or not analysis_data.get('positions'):
        return dbc.Alert("Keine Portfolio-Daten verfügbar.", color="warning")
    anomalies = select_prompt_anomalies(analysis_data)
    if not anomalies:
        return html.Pre(build_portfolio_prompt(analysis_data), style=PROMPT_DEBUG_STYLE)
    parts = [build_portfolio_prompt(single_anomaly_data(analysis_data, b)) for b in anomalies]
    return html.Pre(("\n\n" + "=" * 70 + "\n\n").join(parts), style=PROMPT_DEBUG_STYLE)


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
            # NewsCache liest nur die bestehende SQLite-Datei — kein RAG-Init, kein
            # Fetch. Öffnen dieser Tabelle triggert daher niemals einen Netzabruf.
            cache = NewsCache(DEFAULT_CONFIG.db_path)
            window = DEFAULT_CONFIG.anomaly_window_days
            window_after = DEFAULT_CONFIG.anomaly_window_days_after

            def _ticker_cell(row):
                if pd.isna(row['responsible_ticker']):
                    return text["not_attributable"]
                conc = row.get('concentration')
                conc_label = f" · {conc_labels.get(conc, conc)}" if pd.notna(conc) else ""
                return f"{row['responsible_ticker']} ({row['ticker_contribution_pct']:+.2f}%){conc_label}"

            def _source_cell(b):
                """Zeigt, ob im Cache bereits Quellen liegen, die das Retrieval für diese
                Anomalie tatsächlich sehen kann — Fenster [Tag − window, Tag + window_after],
                identisch zum asymmetrischen Retrieval-Fenster (RAGPipeline.retrieve_for_anomaly).
                Nur für Tage relevant, die überhaupt ein RAG-Ziel sind ('Hisseye özgü' +
                zugeordneter Ticker) — für alle anderen wird nie gefetcht, daher hier auch
                kein Cache-Treffer möglich."""
                ticker = b.get('responsible_ticker')
                if b.get('concentration') != 'Hisseye özgü' or not ticker:
                    return html.Span(text["dash"], className="text-muted")
                try:
                    day = datetime.fromisoformat(str(b['date'])[:10])
                    n = len(cache.get_articles(ticker, day, window, window_days_after=window_after))
                except (ValueError, TypeError):
                    n = 0
                if n == 0:
                    return html.Span(text["dash"], className="text-muted")
                return dbc.Button(
                    text["source_count"].format(n=n),
                    id={'type': 'anomaly-source-btn', 'ticker': ticker, 'date': b['date']},
                    size="sm", color="link", className="p-0",
                    style={'textDecoration': 'underline', 'verticalAlign': 'baseline'},
                )

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
            out.columns = TABLE_HEADERS[lang][:len(out.columns)]

            # dbc.Table.from_dataframe rendert Zellen über str() — für die klickbare
            # Quellen-Spalte (Dash-Komponenten statt Text) wird die Tabelle daher
            # manuell aus thead/tbody gebaut; alle Text-Spalten bleiben unverändert.
            header_row = html.Tr([html.Th(h) for h in TABLE_HEADERS[lang]])
            body_rows = []
            for i, (_, row) in enumerate(out.iterrows()):
                cells = [html.Td(v) for v in row]
                cells.append(html.Td(_source_cell(breaks[i])))
                body_rows.append(html.Tr(cells))
            content = dbc.Table(
                [html.Thead(header_row), html.Tbody(body_rows)],
                striped=True, bordered=True, hover=True, responsive=True, className="mt-2"
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
