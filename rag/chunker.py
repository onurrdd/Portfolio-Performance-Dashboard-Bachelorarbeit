"""
Segmentierung (Chunking) für RAG — Exposé 3.2.

Verwendet den RecursiveCharacterTextSplitter (LangChain): trennt Text hierarchisch
nach Absätzen → Sätzen → Wörtern, statt an beliebigen Zeichengrenzen. Dadurch bleibt
die semantische Integrität erhalten. Für kurze RSS-Headlines entsteht i. d. R. ein
einzelner Chunk; für lange Dokumente (Quartalsberichte, FED-Protokolle) greift die
hierarchische Segmentierung.

Jeder Chunk wird mit Metadaten angereichert (Zeitstempel, Ticker, Quelle), damit das
Retrieval später gezielt nach Ticker und Zeitfenster filtern kann.
"""
import logging
import re
from typing import List, Dict

from langchain_text_splitters import RecursiveCharacterTextSplitter

from news.base import published_to_epoch

logger = logging.getLogger(__name__)

# Rechtlich-formelhafte Wendungen aus SEC-Formularhüllen (Deckblatt, Compliance-
# Klauseln) — inhaltsleer für die Ereigniserklärung, aber lexikalisch nah genug an
# echtem Finanztext, um beim Retrieval hohe Ähnlichkeit vorzutäuschen. EIN Treffer
# reicht nicht zum Ausschluss (kann in echtem Fließtext beiläufig vorkommen); ERST
# eine Häufung (siehe _is_boilerplate) markiert einen Chunk als reine Formelhülle.
_BOILERPLATE_PHRASES = (
    # Formularhuelle (Deckblatt, Compliance-Klauseln)
    "emerging growth company",
    "shall not be deemed",
    "securities exchange act of 1934",
    "pursuant to rule",
    "indicate by check mark",
    "soliciting material",
    "pre-commencement communication",
    # Safe-Harbor-Klausel und Kennzahlendefinitionen am Ende jeder
    # Ergebnis-Pressemitteilung. Sie stehen im selben Dokument wie die
    # Ereignisbegruendung, sind sprachlich jedoch reine Rechtsformeln und
    # konkurrieren beim Retrieval unmittelbar mit dem erklaerenden Ausblick:
    # Sie enthalten dieselben Zukunftsbegriffe ("expectations", "results",
    # "guidance"), auf die eine Ursachenfrage semantisch anspricht.
    "forward-looking statements",
    "actual results may differ materially",
    "disclaims any obligation",
    "risk factors",
    "non-gaap financial measures",
    "securities and exchange commission",
)


# Eindeutige Marker: Sie kommen in erklaerendem Fliesstext praktisch nicht vor,
# sondern nur im Definitions- und Haftungsanhang einer Ergebnismitteilung. Hier
# genuegt EIN Treffer — eine Haeufung abzuwarten wuerde diese Chunks im Index
# belassen, obwohl bereits die einzelne Wendung den Anhang eindeutig ausweist.
_STRONG_BOILERPLATE_PHRASES = (
    "forward-looking statements",
    "actual results may differ materially",
    "disclaims any obligation",
    "non-gaap financial information",
    "non-gaap financial measures",
    "is calculated by dividing",
    "constant currency impacts are calculated",
    "indicate by check mark",
    "emerging growth company",
)


def _is_boilerplate(text: str, min_hits: int = 2) -> bool:
    """Prüft, ob ein Chunk reiner Formel- bzw. Anhangstext ist.

    Zwei Stufen: Ein eindeutiger Marker (_STRONG_BOILERPLATE_PHRASES) genügt für
    sich allein, weil solche Wendungen ausschließlich im Haftungs- und
    Definitionsanhang stehen. Für die übrigen, schwächeren Wendungen entscheidet
    die Häufung: echter Fließtext streift eine davon gelegentlich, der Formeltext
    versammelt mehrere auf engem Raum."""
    lowered = text.lower()
    if any(p in lowered for p in _STRONG_BOILERPLATE_PHRASES):
        return True
    hits = sum(lowered.count(p) for p in _BOILERPLATE_PHRASES)
    return hits >= min_hits


# Ein Ereignisgrund ist Fließtext; eine Kennzahlentabelle bleibt nach der
# HTML-zu-Text-Extraktion eine flache Ziffernfolge ohne erklärenden Satzbau und
# ist damit für die Ursachenfrage wertlos, egal wie ähnlich sie der Suchanfrage
# eingebettet erscheint. Der Schwellenwert liegt bewusst in der Mitte einer an
# echten Dokumenten gemessenen Lücke: Fließtext-Segmente liegen bei 0,0–0,1 %
# Ziffernanteil, Tabellensegmente bei 40–48 % — jede Wahl zwischen 10 % und 20 %
# trifft dieselbe Trennung, der Wert ist also nicht scharf justiert.
_MAX_DIGIT_RATIO = 0.15
_MIN_TABLE_LEN = 120  # unterhalb dieser Länge schwankt der Ziffernanteil zu stark


def _is_numeric_table(text: str) -> bool:
    if len(text) < _MIN_TABLE_LEN:
        return False
    return sum(c.isdigit() for c in text) / len(text) > _MAX_DIGIT_RATIO


# Abschnittsüberschriften in Quartalspräsentationen (8-K-Pressemitteilungsanhänge).
# Im Originaldokument grafisch gesperrt gesetzt (Buchstabe-für-Buchstabe mit
# Leerzeichen) und von _despace_caps (news/sec_edgar.py) bereits zu einem
# einzelnen Großbuchstaben-Token zusammengeführt — NUR diese bekannten,
# zusammengeführten Marker werden gesucht, kein allgemeines Großbuchstaben-Muster:
# echter Fließtext enthält gelegentlich beiläufige All-Caps-Akronyme (z. B. "GAAP",
# "EBITDA"), die keine Abschnittsgrenze markieren würden. Diese Struktur ist eine
# redaktionelle Gewohnheit des jeweiligen Unternehmens, keine regulatorische
# Vorgabe (siehe SECEdgarSource) — ein Dokument ohne diese Marker durchläuft die
# Erkennung folgenlos (Abschnitt bleibt "", nichts wird ausgeschlossen).
_SECTION_MARKERS = (
    "SUMMARYHIGHLIGHTS", "FINANCIALSUMMARY", "OPERATIONALSUMMARY",
    "VEHICLECAPACITY", "CORETECHNOLOGY", "OTHERHIGHLIGHTS", "OUTLOOK",
    "FINANCIALSTATEMENTS", "STATEMENTOFOPERATIONS", "BALANCESHEET",
    "STATEMENTOFCASHFLOWS", "ADDITIONALINFORMATION",
)
# Tabellen-/Anhangabschnitte: enthalten dieselben Kennzahlen, die _is_numeric_table
# ohnehin filtert, aber auch kurze, wenig ziffernhaltige Übergangssätze innerhalb
# dieser Abschnitte (z. B. Bilanz-Fußnoten) — die Abschnittszugehörigkeit fängt
# diese zusätzlich ab, wo die reine Ziffernquote nicht greift.
_EXCLUDED_SECTIONS = (
    "FINANCIALSTATEMENTS", "STATEMENTOFOPERATIONS", "BALANCESHEET",
    "STATEMENTOFCASHFLOWS", "ADDITIONALINFORMATION", "KEYMETRICS",
)
_SECTION_MARKER_RE = re.compile(r"\b[A-Z]{6,}\b")


def _tag_sections(full_text: str, text_chunks: List[str]) -> List[str]:
    """Ordnet jedem Chunk den zuletzt VOR seinem Textbeginn gefundenen
    Abschnitts-Marker zu (siehe _SECTION_MARKERS). Kein Marker im Dokument
    gefunden -> für jeden Chunk "" (Fallback, ändert das bisherige Verhalten
    nicht)."""
    markers = []
    for m in _SECTION_MARKER_RE.finditer(full_text):
        tok = m.group(0)
        if tok.startswith("KEYMETRICS"):
            markers.append((m.start(), "KEYMETRICS"))
        elif tok in _SECTION_MARKERS:
            markers.append((m.start(), tok))
    if not markers:
        return [""] * len(text_chunks)

    sections = []
    cursor = 0
    for chunk in text_chunks:
        idx = full_text.find(chunk, cursor)
        if idx == -1:
            idx = full_text.find(chunk)  # Fallback: Overlap kann den Cursor überholt haben
        current = ""
        for pos, name in markers:
            if idx >= 0 and pos <= idx:
                current = name
            else:
                break
        sections.append(current)
        if idx >= 0:
            cursor = idx
    return sections


class NewsChunker:
    """Segmentiert Nachrichtenartikel in überlappende, metadaten-angereicherte Chunks."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

    def process_article(self, article: Dict) -> List[Dict]:
        """Ein Artikel → Liste von Chunk-Dicts (Text + Metadaten).

        Erwartetes Artikel-Schema (news.base): title, summary, link, published, source, ticker.
        """
        full_text = f"{article.get('title', '')}. {article.get('summary', '')}".strip()
        text_chunks = self._splitter.split_text(full_text) if full_text else []
        # Abschnittszugehörigkeit VOR dem Filtern bestimmen: die Marker-Positionen
        # beziehen sich auf full_text, nicht auf die bereits gefilterte Liste.
        sections = _tag_sections(full_text, text_chunks) if full_text else []

        # Formelhülle, Kennzahlentabelle und Tabellen-/Anhangabschnitt VOR dem
        # Embedding verwerfen: keiner dieser drei trägt zur Ereigniserklärung bei
        # und verschlechtert nur die Trefferpräzision (Context Precision) — siehe
        # _is_boilerplate, _is_numeric_table, _EXCLUDED_SECTIONS.
        filtered = [
            (t, sec) for t, sec in zip(text_chunks, sections)
            if not _is_boilerplate(t) and not _is_numeric_table(t)
            and sec not in _EXCLUDED_SECTIONS
        ]

        published = article.get("published", "")
        epoch = published_to_epoch(published)

        chunks = []
        for i, (chunk_text, section) in enumerate(filtered):
            chunks.append({
                "text": chunk_text,
                "title": article.get("title", ""),
                "ticker": article.get("ticker", ""),
                "link": article.get("link", ""),
                "published": published,              # ISO-String (Anzeige)
                "published_epoch": epoch,            # Unix-Sek. (Zeitfenster-Filter)
                "source": article.get("source", ""),
                "section": section,                  # "" falls Dokument ohne Marker
                "chunk_index": i,
                "total_chunks": len(filtered),
            })
        return chunks

    def process_articles(self, articles: List[Dict]) -> List[Dict]:
        """Mehrere Artikel → flache Liste aller Chunks."""
        all_chunks = []
        for article in articles:
            all_chunks.extend(self.process_article(article))
        logger.info(f"Processed {len(articles)} articles into {len(all_chunks)} chunks")
        return all_chunks
