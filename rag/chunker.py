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


# Firmenbezeichnungen je Kürzel — Grundlage der Relevanzprüfung unten. Alpha Vantage
# ordnet einem Kürzel auch Artikel zu, in denen das Unternehmen nur am Rande erwähnt
# wird; die Zuordnung sagt also nicht, dass der Artikel VON diesem Unternehmen
# handelt. Mehrere Schreibweisen je Kürzel, weil Meldungen mal die Marke, mal den
# Firmennamen verwenden.
_TICKER_ALIASES = {
    "AAPL": ("apple",),
    "GME": ("gamestop",),
    "GOOGL": ("google", "alphabet"),
    "MSFT": ("microsoft",),
    "NVDA": ("nvidia",),
    "TSLA": ("tesla",),
}


def _mentions_company(text: str, ticker: str) -> bool:
    """Prüft, ob ein Artikeltext das Unternehmen des Kürzels überhaupt nennt.

    Ein Artikel, der weder den Firmennamen noch das Kürzel enthält, handelt nicht
    von diesem Unternehmen — er wurde nur zugeordnet, weil das Kürzel im
    Datenbestand der Quelle mit ihm verknüpft ist (etwa ein Marktbericht, der
    andere Titel derselben Branche bespricht). Solche Artikel treten beim Abruf
    gegen den erklärenden Absatz an, ohne zur Erklärung beitragen zu können.

    Unbekanntes Kürzel -> True: lieber ein Artikel zu viel als eine stumme Lücke,
    wenn das Portfolio um einen Titel erweitert wird, der oben fehlt."""
    aliases = _TICKER_ALIASES.get((ticker or "").upper())
    if not aliases:
        return True
    lowered = text.lower()
    return (ticker or "").lower() in lowered or any(a in lowered for a in aliases)


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


def _split_by_section(full_text: str) -> List[tuple]:
    """Zerlegt ein Dokument an seinen Abschnittsmarkern in (Abschnitt, Text)-Paare.

    Grundlage für eine abschnittsweise Segmentierung: Wird ein Dokument erst als
    Ganzes in 800-Zeichen-Blöcke geschnitten und der Abschnitt anschließend über die
    Startposition zugewiesen (siehe _tag_sections), fallen Blockgrenze und
    Abschnittsgrenze auseinander. Ein Block beginnt dann im erklärenden Ausblick und
    endet in der folgenden Kennzahlentabelle — er trägt das Etikett "OUTLOOK",
    besteht inhaltlich aber überwiegend aus Tabellenresten. Die eigentliche
    Begründung landet umgekehrt im Blockende des Vorgängerabschnitts und ist über
    das Etikett nicht mehr auffindbar.

    Der Text VOR dem ersten Marker wird verworfen: In einem gegliederten Bericht
    stehen dort Deckblatt und Inhaltsverzeichnis. Das Verzeichnis führt die
    Abschnittsnamen samt Seitenzahlen auf ("Outlook 12", "Key Metrics 22") und
    liegt einer Ursachenfrage im Einbettungsraum dadurch nahe, ohne eine einzige
    Aussage zu enthalten — das Retrieval fände dort den Namen des Abschnitts
    statt seines Inhalts. Kein Bericht begründet ein Ereignis vor seiner ersten
    Überschrift, die Regel ist daher nicht an ein bestimmtes Unternehmen gebunden.

    Ohne Marker — Nachrichtenmeldungen — bleibt der Text unangetastet und wird als
    einziges Paar ("", Text) zurückgegeben; die Regel greift nur dort, wo es
    überhaupt eine Gliederung gibt.
    """
    markers = []
    for m in _SECTION_MARKER_RE.finditer(full_text):
        tok = m.group(0)
        if tok.startswith("KEYMETRICS"):
            markers.append((m.start(), m.end(), "KEYMETRICS"))
        elif tok in _SECTION_MARKERS:
            markers.append((m.start(), m.end(), tok))
    if not markers:
        return [("", full_text)]

    parts = []
    for i, (start, end, name) in enumerate(markers):
        stop = markers[i + 1][0] if i + 1 < len(markers) else len(full_text)
        # Ab `end`: der Marker selbst ist eine Überschrift, kein Inhalt.
        body = full_text[end:stop].strip()
        if body:
            parts.append((name, body))
    return parts


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
        # NUR der Fließtext wird eingebettet, NICHT die Überschrift. Eine
        # Nachrichtenüberschrift ist redaktionell auf Aufmerksamkeit getrimmt und
        # trägt gerade jene Wörter, auf die eine Ursachenfrage anspringt ("Stocks",
        # "$5,000", "Million"), ohne dass der Artikel etwas über die Kursbewegung
        # sagt. Sie verschafft kurzen Meldungen damit einen Vorsprung, der aus der
        # Aufmachung stammt und nicht aus dem Inhalt. Ein Absatz aus einem
        # Geschäftsbericht hat keine solche Überschrift und verliert den Vergleich
        # allein dadurch. Für Anzeige und Quellenangabe bleibt der Titel als
        # Metadatum erhalten (siehe unten `title`) — er verschwindet nur aus dem
        # eingebetteten Text.
        full_text = str(article.get("summary", "")).strip()

        # Relevanzprüfung VOR der Segmentierung: Nennt der Artikel das Unternehmen
        # nirgends — weder im Titel noch im Text —, wird er gar nicht erst
        # indiziert. Die Prüfung schließt den Titel ausdrücklich ein, obwohl dieser
        # nicht eingebettet wird: Er sagt, wovon der Artikel handelt, auch wenn er
        # als Suchtext ungeeignet ist. Ohne diese Prüfung stehen Meldungen im Index,
        # die der Quelle zufolge zum Kürzel gehören, das Unternehmen aber nur als
        # Teil einer Aufzählung führen (siehe _mentions_company).
        if not _mentions_company(
                f"{article.get('title', '')} {full_text}", article.get("ticker", "")):
            logger.info("Artikel ohne Unternehmensbezug übersprungen (%s): %s",
                        article.get("ticker", ""), (article.get("title", "") or "")[:60])
            return []

        # ABSCHNITTSWEISE segmentieren: erst am Abschnittsmarker trennen, dann jeden
        # Abschnitt für sich in Blöcke schneiden. Dadurch liegt keine Blockgrenze quer
        # über einer Abschnittsgrenze, und das Etikett beschreibt tatsächlich den
        # Inhalt des Blocks (siehe _split_by_section). Dokumente ohne Marker —
        # Nachrichtenartikel — ergeben ein einziges Paar ("", Text) und durchlaufen
        # damit unverändert dieselbe Segmentierung wie zuvor.
        text_chunks, sections = [], []
        for section, body in (_split_by_section(full_text) if full_text else []):
            for part in self._splitter.split_text(body):
                text_chunks.append(part)
                sections.append(section)

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
        # Präfix NUR für Blöcke aus gegliederten Dokumenten (Abschnitt erkannt, also
        # Geschäftsbericht statt Nachrichtenmeldung) und NUR das Tickerkürzel. Ein
        # solcher Block stammt aus der Mitte eines 35.000-Zeichen-Berichts und nennt
        # die Firma nirgends — er spricht von "our company" und "our teams". Das
        # Kürzel stellt diesen Bezug her.
        # Der vollständige Dokumenttitel wäre dafür untauglich: Er ist mit rund 100
        # Zeichen länger als mancher Block und in allen Blöcken desselben Berichts
        # identisch. Er verschöbe damit die Einbettung aller Blöcke in dieselbe
        # Richtung und ebnete gerade den inhaltlichen Unterschied ein, auf den es
        # beim Sortieren ankommt — ein nahezu leerer Block ("(Unaudited)") rückte
        # allein durch das Präfix nach vorn.
        # Nachrichtenmeldungen bleiben ausgenommen: Ihr Titel ist redaktionell auf
        # Aufmerksamkeit getrimmt ("Stocks", "$5,000", "Million") und verschaffte
        # ihnen einen Vorsprung aus der Aufmachung statt aus dem Inhalt (siehe oben,
        # full_text ohne Überschrift).
        prefix = article.get("ticker", "") if any(sec for _, sec in filtered) else ""

        chunks = []
        for i, (chunk_text, section) in enumerate(filtered):
            text = f"{prefix}. {chunk_text}" if prefix else chunk_text
            chunks.append({
                "text": text,
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
