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

logger = logging.getLogger(__name__)

_URL = "https://www.alphavantage.co/query"


class AlphaVantageNewsSource:
    """NewsSource-Adapter für Alpha Vantage NEWS_SENTIMENT."""

    name = "alpha_vantage"
    # time_from/time_to werden serverseitig ausgewertet -> echter Zeitraum-Zugriff.
    supports_date_range = True

    def __init__(self):
        self._api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        self._enabled = bool(self._api_key)
        if not self._api_key:
            logger.warning("ALPHAVANTAGE_API_KEY nicht gesetzt — AlphaVantageNewsSource deaktiviert "
                          "(siehe .env.example)")

    def fetch(self, ticker: str, limit: int = 10,
              start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]:
        if not self._enabled or start is None or end is None:
            # Ohne Zeitfenster kein sinnvoller Query; ohne Key ist die Quelle inaktiv.
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
            })
        return articles


def _parse_av_timestamp(raw: str) -> str:
    """Alpha-Vantage-Zeitformat 'YYYYMMDDTHHMMSS' -> ISO-String ('' bei Parse-Fehler)."""
    try:
        return datetime.strptime(raw, "%Y%m%dT%H%M%S").isoformat()
    except ValueError:
        return ""
