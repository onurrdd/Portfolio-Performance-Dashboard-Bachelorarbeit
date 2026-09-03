"""
Alpha Vantage NEWS_SENTIMENT — historische Finanznachrichten-Artikel (ab ca. 03/2022).

Ergänzt SEC EDGAR (offizielle Meldungen, seit 2001) um redaktionelle Nachrichtenartikel
mit echtem Publikationsdatum. Für Fenster vor ca. März 2022 liefert diese Quelle
typischerweise nichts — keine Fehlerbehandlung nötig, einfach leere Liste; EDGAR
deckt diesen Zeitraum ab.
"""
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

import requests

from rag.config import DEFAULT_CONFIG
from rag import config as rag_config  # Sparmodus-Schalter, als Attribut gelesen

logger = logging.getLogger(__name__)

_URL = "https://www.alphavantage.co/query"


class AlphaVantageNewsSource:
    """NewsSource-Adapter für Alpha Vantage NEWS_SENTIMENT."""

    name = "alpha_vantage"
    # time_from/time_to werden serverseitig ausgewertet -> echter Zeitraum-Zugriff.
    supports_date_range = True

    def __init__(self):
        self._api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        # Kontingentschutz hat Vorrang vor dem Key: die Quelle geht nur ans Netz,
        # wenn BEIDES gilt — der Sparmodus begrenzt die Anomalieliste ohnehin auf
        # wenige Fälle (rag_config.SAVING_MODE) UND der harte Schalter ist an
        # (RAGConfig.alpha_vantage_enabled). Ein Vollmodus-Lauf über alle Anomalien
        # würde das Tageskontingent (25/Tag) sprengen und holt AV-Inhalte daher
        # ausschließlich aus dem Cache.
        self._quota_lock = not (rag_config.SAVING_MODE and DEFAULT_CONFIG.alpha_vantage_enabled)
        self._enabled = bool(self._api_key) and not self._quota_lock
        if self._quota_lock:
            logger.info("AlphaVantageNewsSource deaktiviert (Sparmodus aus oder "
                        "RAGConfig.alpha_vantage_enabled=False) — Kontingentschutz, "
                        "es wird keine Anfrage gesendet")
        elif not self._api_key:
            logger.warning("ALPHAVANTAGE_API_KEY nicht gesetzt — AlphaVantageNewsSource deaktiviert "
                          "(siehe .env.example)")

    def fetch(self, ticker: str, limit: int = 10,
              start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]:
        if not self._enabled or start is None or end is None:
            # Ohne Zeitfenster kein sinnvoller Query; ohne Key bzw. bei aktivem
            # Kontingentschutz ist die Quelle inaktiv.
            return []

        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "time_from": start.strftime("%Y%m%dT0000"),
            "time_to": end.strftime("%Y%m%dT2359"),
            "limit": min(limit, 50),
            "sort": "RELEVANCE",
            "apikey": self._api_key,
        }
        try:
            resp = requests.get(_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Alpha Vantage: Anfrage für {ticker} fehlgeschlagen: {e}")
            return []

        # Kontingent erschöpft (25/Tag) oder ungültiger Key -> kein Fehler werfen,
        # nur loggen; andere Quellen (SEC EDGAR, Yahoo) liefern weiter.
        if "Information" in data or "Note" in data or "Error Message" in data:
            msg = data.get("Information") or data.get("Note") or data.get("Error Message")
            logger.warning(f"Alpha Vantage: {msg}")
            return []

        articles = []
        for item in data.get("feed", [])[:limit]:
            articles.append({
                "title": item.get("title", ""),
                "summary": item.get("summary", ""),
                "link": item.get("url", ""),
                "published": _parse_av_timestamp(item.get("time_published", "")),
                "source": self.name,
                "ticker": ticker,
                # Ticker-spezifischer Bezug laut Alpha Vantage (0..1) — NICHT
                # Sentiment. Kommt aus ticker_sentiment, nicht dem Artikel selbst:
                # ein Ticker kann im Feed auftauchen, ohne dass der Artikel von
                # diesem Unternehmen handelt (siehe rag/chunker.py::_mentions_company,
                # dieselbe Lücke auf Textebene). Wird bisher nur mitgeführt, nicht
                # gefiltert — Filterung ist ein möglicher nächster Schritt.
                "relevance_score": _ticker_relevance(item, ticker),
            })
        return articles


def _ticker_relevance(item: dict, ticker: str) -> Optional[float]:
    """Liest den `relevance_score` des angefragten Tickers aus `ticker_sentiment`.

    None, wenn der Ticker dort fehlt oder das Feld ungültig ist — kein 0,0, das
    fälschlich als „irrelevant gemessen" statt „nicht gemessen" gelesen würde."""
    for ts in item.get("ticker_sentiment", []):
        if ts.get("ticker") == ticker:
            try:
                return float(ts.get("relevance_score"))
            except (TypeError, ValueError):
                return None
    return None


def _parse_av_timestamp(raw: str) -> str:
    """Alpha-Vantage-Zeitformat 'YYYYMMDDTHHMMSS' -> ISO-String ('' bei Parse-Fehler)."""
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%S").isoformat()
    except ValueError:
        return ""
