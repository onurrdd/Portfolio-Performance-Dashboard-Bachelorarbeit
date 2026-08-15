"""
SEC EDGAR — historische, tagesgenaue Unternehmensmeldungen (kein API-Key nötig).

Deckt die zentrale Datenlücke ab: Yahoo-RSS liefert nur aktuelle Headlines; für
vergangene Anomalietage braucht das RAG-Fenster [Datum ± window] eine Quelle mit
echtem historischem Zugriff. EDGAR liefert offizielle Meldungen seit 2001, mit
exaktem filingDate — passt direkt auf die (Ticker, Datum)-Filterung der Pipeline.

Fokus auf 8-K (Ad-hoc-Ereignismeldungen, z. B. Item 2.02 = Quartalszahlen,
Item 5.02 = Führungswechsel) statt vollständiger 10-K/10-Q-Berichte: 8-Ks sind
kurz, ereignisbezogen und exakt auf den Tag datiert, an dem das Ereignis
bekanntgegeben wurde — sie stehen der "Nachricht dazu" näher als der volle Bericht.
10-Q/10-K werden zusätzlich mitgenommen, falls sie ins Fenster fallen.
"""
import logging
import os
import re
from datetime import datetime, date
from typing import Dict, List, Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_SHARD_URL = "https://data.sec.gov/submissions/{name}"
_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{doc}"

# Häufige 8-K Item-Codes -> menschenlesbare Bezeichnung (SEC Form 8-K Instructions).
_ITEM_LABELS = {
    "1.01": "Entry into a Material Definitive Agreement",
    "1.02": "Termination of a Material Definitive Agreement",
    "2.01": "Completion of Acquisition or Disposition of Assets",
    "2.02": "Results of Operations and Financial Condition",
    "2.05": "Costs Associated with Exit or Disposal Activities",
    "2.06": "Material Impairments",
    "3.01": "Notice of Delisting or Failure to Satisfy a Listing Rule",
    "4.01": "Changes in Registrant's Certifying Accountant",
    "5.02": "Departure/Election of Directors or Officers",
    "5.03": "Amendments to Articles of Incorporation or Bylaws",
    "7.01": "Regulation FD Disclosure",
    "8.01": "Other Events",
    "9.01": "Financial Statements and Exhibits",
}


def _describe_items(items_field: str) -> str:
    if not items_field:
        return ""
    labels = [_ITEM_LABELS.get(c.strip(), c.strip()) for c in items_field.split(",") if c.strip()]
    return "; ".join(labels)


class SECEdgarSource:
    """NewsSource-Adapter für SEC EDGAR Filings (8-K/10-Q/10-K)."""

    name = "sec_edgar"
    # Echter historischer Zugriff (Filings seit 2001, exaktes filingDate).
    supports_date_range = True
    _cik_map: Optional[Dict[str, str]] = None  # klassenweiter Cache (ein Fetch pro Prozess)

    def __init__(self, forms=("8-K", "10-Q", "10-K"), max_document_chars: int = 12000):
        self.forms = set(forms)
        self.max_document_chars = max_document_chars
        self._session = requests.Session()
        ua = os.environ.get("SEC_USER_AGENT")
        self._enabled = bool(ua)
        if ua:
            self._session.headers.update({"User-Agent": ua})
        else:
            logger.warning("SEC_USER_AGENT nicht gesetzt — SECEdgarSource deaktiviert (siehe .env.example)")

    def fetch(self, ticker: str, limit: int = 10,
              start: Optional[datetime] = None, end: Optional[datetime] = None) -> List[Dict]:
        if not self._enabled:
            return []
        if start is None or end is None:
            # Ohne Zeitfenster ist EDGAR nicht sinnvoll einsetzbar (kein RSS-artiges
            # "aktuell"-Konzept) — lieber nichts liefern als unbestimmt suchen.
            return []

        cik = self._resolve_cik(ticker)
        if not cik:
            logger.info(f"SEC EDGAR: keine CIK für Ticker '{ticker}' gefunden")
            return []

        filings = self._filings_in_range(cik, start.date(), end.date())
        articles = []
        for f in filings[:limit]:
            text = self._fetch_document_text(cik, f["accessionNumber"], f["primaryDocument"])
            if not text:
                continue
            item_desc = _describe_items(f.get("items", ""))
            title = f"{ticker} {f['form']} filing" + (f" — {item_desc}" if item_desc else "")
            link = _ARCHIVE_URL.format(
                cik_int=int(cik), accession_nodash=f["accessionNumber"].replace("-", ""),
                doc=f["primaryDocument"])
            articles.append({
                "title": title,
                "summary": text[:self.max_document_chars],
                "link": link,
                "published": f["filingDate"],  # 'YYYY-MM-DD', ISO-kompatibel
                "source": self.name,
                "ticker": ticker,
            })
        return articles

    # --- intern ---

    def _resolve_cik(self, ticker: str) -> Optional[str]:
        if SECEdgarSource._cik_map is None:
            SECEdgarSource._cik_map = self._load_cik_map()
        return SECEdgarSource._cik_map.get(ticker.upper())

    def _load_cik_map(self) -> Dict[str, str]:
        try:
            resp = self._session.get(_TICKERS_URL, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            return {row["ticker"].upper(): str(row["cik_str"]).zfill(10) for row in data.values()}
        except Exception as e:
            logger.warning(f"SEC EDGAR: Ticker->CIK-Liste konnte nicht geladen werden: {e}")
            return {}

    def _filings_in_range(self, cik: str, start_date: date, end_date: date) -> List[Dict]:
        try:
            resp = self._session.get(_SUBMISSIONS_URL.format(cik=cik), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"SEC EDGAR: Submissions für CIK {cik} nicht ladbar: {e}")
            return []

        recent = data.get("filings", {}).get("recent", {})
        matches = self._filter_rows(self._rows(recent), start_date, end_date)

        # Falls das Fenster vor den ältesten "recent"-Einträgen liegt: passende
        # Archiv-Shards nachladen (files[] enthält filingFrom/filingTo pro Shard).
        recent_dates = recent.get("filingDate", [])
        oldest_recent = min(recent_dates) if recent_dates else None
        if oldest_recent is None or str(start_date) < oldest_recent:
            for shard in data.get("filings", {}).get("files", []):
                if shard.get("filingTo", "9999-99-99") >= str(start_date) and \
                   shard.get("filingFrom", "0000-00-00") <= str(end_date):
                    matches.extend(self._match_shard(shard["name"], start_date, end_date))

        matches.sort(key=lambda f: f["filingDate"])
        return matches

    @staticmethod
    def _rows(block: Dict):
        forms = block.get("form", [])
        items = block.get("items", [""] * len(forms))
        return zip(block.get("accessionNumber", []), block.get("filingDate", []),
                  forms, block.get("primaryDocument", []), items)

    def _match_shard(self, shard_name: str, start_date: date, end_date: date) -> List[Dict]:
        try:
            resp = self._session.get(_SHARD_URL.format(name=shard_name), timeout=15)
            resp.raise_for_status()
            shard = resp.json()
        except Exception as e:
            logger.warning(f"SEC EDGAR: Archiv-Shard {shard_name} nicht ladbar: {e}")
            return []
        return self._filter_rows(self._rows(shard), start_date, end_date)

    def _filter_rows(self, rows, start_date: date, end_date: date) -> List[Dict]:
        out = []
        for accession, filing_date, form, primary_doc, items in rows:
            if form not in self.forms:
                continue
            try:
                d = datetime.strptime(filing_date, "%Y-%m-%d").date()
            except ValueError:
                continue
            if not (start_date <= d <= end_date):
                continue
            out.append({"accessionNumber": accession, "filingDate": filing_date,
                       "form": form, "primaryDocument": primary_doc, "items": items})
        return out

    def _fetch_document_text(self, cik: str, accession: str, primary_doc: str) -> str:
        url = _ARCHIVE_URL.format(cik_int=int(cik), accession_nodash=accession.replace("-", ""),
                                  doc=primary_doc)
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
            # Moderne EDGAR-Dokumente sind Inline-XBRL: <ix:header> enthält die
            # unsichtbaren XBRL-Fakten (contextRef/unit-IDs etc.), keinen lesbaren
            # Fließtext. Ohne diesen Schnitt besteht der extrahierte Text fast nur
            # aus XBRL-Rauschen statt aus der eigentlichen Meldung.
            for tag in soup.find_all(["script", "style", "ix:header"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            return re.sub(r"\s+", " ", text)
        except Exception as e:
            logger.warning(f"SEC EDGAR: Dokument {url} nicht ladbar: {e}")
            return ""
