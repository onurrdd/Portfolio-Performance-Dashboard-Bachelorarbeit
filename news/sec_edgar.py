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

Für 8-Ks wird NICHT das primäre Formulardokument indiziert, sondern — falls
vorhanden — das dazugehörige Pressemitteilungs-Exhibit (EX-99.1): Das primäre
Dokument moderner (Inline-XBRL) 8-Ks ist nur die Formularhülle (Deckblatt,
Item-Nummern, rechtliche Standardklauseln, XBRL-Tags) und enthält den eigentlichen
Ereignisinhalt nicht — der steht im Exhibit. Ohne diese Weiche würde die Pipeline
mit rechtlichem Rahmentext statt der Meldung selbst gefüttert.
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
# Filing-Index-Seite (Liste aller zu einer Einreichung gehörenden Dokumente inkl. Exhibits) —
# im Submissions-Feed NICHT enthalten, daher ein separater Abruf pro 8-K (siehe _find_exhibit_doc).
_INDEX_URL = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_nodash}/{accession}-index.html"

# Pressemitteilungs-Exhibit einer 8-K (fast immer EX-99.1, gelegentlich EX-99.2 etc.).
_EXHIBIT_TYPE_RE = re.compile(r"EX-99(\.\d+)?", re.IGNORECASE)

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


# EDGAR-Dokumentkopf: "EX-99.1 2 tsla-ex99_1.htm EX-99.1" — Dateiname und
# Exhibit-Nummer aus dem Archiv-Header, kein Meldungsinhalt.
_DOC_HEADER_RE = re.compile(
    r"^\s*(?:EX-[\w.]+|\d{1,2})\s+\d{1,3}\s+\S+\.htm\S*\s+(?:EX-[\w.]+\s+)?",
    re.IGNORECASE)

# Gesperrter Satz ("S U M M A R Y   H I G H L I G H T S") entsteht beim Extrahieren
# grafisch gesetzter Ueberschriften. Ohne Normalisierung ist jeder Buchstabe ein
# eigenes Token — die Ueberschrift wird fuer das Embedding unlesbar.
_SPACED_CAPS_RE = re.compile(r"\b(?:[A-Z]\s){2,}[A-Z]\b")


def _despace_caps(text: str) -> str:
    return _SPACED_CAPS_RE.sub(lambda m: m.group(0).replace(" ", ""), text)


def _clean_filing_text(text: str) -> str:
    """Entfernt Archiv-Kopfzeile und repariert gesperrt gesetzte Ueberschriften."""
    return _despace_caps(_DOC_HEADER_RE.sub("", text)).strip()


_MDA_HEADING_RE = re.compile(r"MANAGEMENT.?S DISCUSSION AND ANALYSIS", re.IGNORECASE)


def _slice_relevant_text(text: str, form: str, max_chars: int) -> str:
    """Waehlt bei 10-Q/10-K das Segment AB dem MD&A-Abschnitt (Management's
    Discussion and Analysis of Financial Condition and Results of Operations).

    MD&A ist die einzige Stelle im Dokument, an der Regulation S-K eine
    Ursachenerklaerung in Fliesstext vorschreibt (10-K Item 7 bzw. 10-Q Teil I
    Item 2) — anders als beim 8-K-Pressemitteilungsanhang (siehe Modul-
    Docstring und _SECTION_MARKERS in rag/chunker.py) ist dieser Abschnitt
    also nicht redaktionelle Gewohnheit, sondern regulatorisch fixiert.

    Die Ueberschrift kommt im Dokument mehrfach vor: zuerst im Inhalts-
    verzeichnis (Verweis, nicht die Stelle selbst), danach als tatsaechliche
    Abschnittsueberschrift. Der ZWEITE Treffer markiert daher den Anfang des
    Abschnitts; ein einzelner Treffer laesst offen, ob es sich um den
    Verzeichniseintrag handelt, und wird deshalb nicht verwendet. Bei
    laengeren Berichten liegt dieser zweite Treffer regelmaessig weit hinter
    einer festen Zeichenobergrenze ab Dokumentanfang — ohne dieses Verfahren
    wuerde die Kuerzung den Abschnitt regelmaessig verfehlen.

    Bei 8-K (kein vorgeschriebener MD&A-Abschnitt) oder wenn kein zweiter
    Treffer gefunden wird, bleibt das bisherige Verhalten (Kuerzung ab
    Dokumentanfang) unveraendert."""
    if form not in ("10-Q", "10-K"):
        return text[:max_chars]
    positions = [m.start() for m in _MDA_HEADING_RE.finditer(text)]
    if len(positions) < 2:
        return text[:max_chars]
    start = positions[1]
    return text[start:start + max_chars]


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
            doc = f["primaryDocument"]
            # Bei 8-Ks steckt der eigentliche Inhalt im Pressemitteilungs-Exhibit, nicht
            # im primären Formulardokument (siehe Modul-Docstring). Kein Exhibit gefunden
            # (ältere Filings ohne EX-99, oder Index nicht ladbar) → Fallback aufs primäre
            # Dokument, damit ein Filing nie ganz verloren geht.
            if f["form"] == "8-K":
                exhibit_doc = self._find_exhibit_doc(cik, f["accessionNumber"])
                if exhibit_doc:
                    doc = exhibit_doc
            text = self._fetch_document_text(cik, f["accessionNumber"], doc)
            if not text:
                continue
            item_desc = _describe_items(f.get("items", ""))
            title = f"{ticker} {f['form']} filing" + (f" — {item_desc}" if item_desc else "")
            link = _ARCHIVE_URL.format(
                cik_int=int(cik), accession_nodash=f["accessionNumber"].replace("-", ""), doc=doc)
            articles.append({
                "title": title,
                "summary": _slice_relevant_text(text, f["form"], self.max_document_chars),
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

    def _find_exhibit_doc(self, cik: str, accession: str) -> Optional[str]:
        """Sucht im Filing-Index (Dokumentliste EINER Einreichung) nach dem
        Pressemitteilungs-Exhibit (Type EX-99, meist EX-99.1). Gibt den Dateinamen
        zurück oder None, falls keins existiert oder der Index nicht ladbar ist —
        der Aufrufer fällt dann auf das primäre Dokument zurück.
        """
        url = _INDEX_URL.format(cik_int=int(cik), accession_nodash=accession.replace("-", ""),
                                accession=accession)
        try:
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "html.parser")
        except Exception as e:
            logger.warning(f"SEC EDGAR: Filing-Index {url} nicht ladbar: {e}")
            return None

        best = None
        for row in soup.select("table.tableFile tr"):
            cells = row.find_all("td")
            if len(cells) < 4:
                continue
            doc_type = cells[3].get_text(strip=True)
            if not _EXHIBIT_TYPE_RE.fullmatch(doc_type):
                continue
            link_tag = cells[2].find("a")
            if not link_tag or not link_tag.get("href"):
                continue
            doc_name = link_tag["href"].rsplit("/", 1)[-1]
            if doc_type.upper() == "EX-99.1":
                return doc_name  # bevorzugtes Exhibit — sofort zurückgeben
            if best is None:
                best = doc_name  # anderes EX-99.x (z. B. EX-99.2) als Fallback merken
        return best

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
            return _clean_filing_text(re.sub(r"\s+", " ", text))
        except Exception as e:
            logger.warning(f"SEC EDGAR: Dokument {url} nicht ladbar: {e}")
            return ""
