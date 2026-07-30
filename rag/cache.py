"""
Persistenter Nachrichten-Cache (SQLite, stdlib — keine zusätzliche Abhängigkeit).

Zweck: bereits geholte Nachrichten NICHT bei jedem Neustart / Portfolio-Wechsel erneut
aus dem Netz laden. Der Cache ist zeit- und tickerbewusst (Artikel werden mit ihrem
`published`-Datum gespeichert), denn Nachrichten ändern sich über die Zeit — ein reines
"Ticker schon geholt"-Flag wäre falsch.

Zwei Ebenen:
  1. Artikel-Speicher: dedupliziert über `link` (Primärschlüssel), jeder Artikel mit
     eigenem Zeitstempel. Kanonische Quelle dafür, welche Artikel existieren.
  2. Abdeckungs-Abfrage (has_coverage): "Gibt es für Ticker X im Zeitfenster [Tag ± N]
     bereits Artikel?" → steuert, ob überhaupt aus dem Netz geholt werden muss.
     Genau die vom Nutzer beschriebene (Ticker, Datum)-Logik; funktioniert für den
     Anomalietag-Fall ebenso wie für "aktuelle Nachrichten" (Tag = heute).

FAISS bleibt der Suchindex; dieser Cache ist die Fetch-/Dedup-Schicht davor.
"""
import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timedelta
from typing import List, Dict, Optional

from news.base import published_to_epoch

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    link            TEXT PRIMARY KEY,
    ticker          TEXT,
    title           TEXT,
    summary         TEXT,
    published       TEXT,
    published_epoch INTEGER,
    source          TEXT,
    indexed         INTEGER DEFAULT 0,
    fetched_at      TEXT
);
CREATE INDEX IF NOT EXISTS idx_ticker_epoch ON articles(ticker, published_epoch);
"""


class NewsCache:
    """Zeitbewusster Nachrichten-Cache auf SQLite-Basis."""

    def __init__(self, db_path: str = "./data/news_cache.db"):
        self.db_path = db_path
        with self._connect() as con:
            con.executescript(_SCHEMA)
        logger.info(f"NewsCache bereit ({db_path}, {self.count()} Artikel)")

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def has_coverage(self, ticker: str, day: datetime, window_days: int) -> bool:
        """Existiert für `ticker` mindestens ein datierter Artikel in [day ± window_days]?

        Undatierte Artikel (published_epoch = 0) zählen nicht als Abdeckung.
        """
        epoch_from = int((day - timedelta(days=window_days)).timestamp())
        epoch_to = int((day + timedelta(days=window_days)).timestamp())
        with self._connect() as con:
            row = con.execute(
                "SELECT COUNT(*) FROM articles "
                "WHERE ticker=? AND published_epoch BETWEEN ? AND ? AND published_epoch > 0",
                (ticker, epoch_from, epoch_to),
            ).fetchone()
        return row[0] > 0

    def get_articles(self, ticker: str, day: Optional[datetime] = None,
                     window_days: int = 3) -> List[Dict]:
        """Artikel für einen Ticker; optional auf ein Zeitfenster um `day` beschränkt."""
        cols = "link, ticker, title, summary, published, published_epoch, source, indexed"
        with self._connect() as con:
            if day is not None:
                epoch_from = int((day - timedelta(days=window_days)).timestamp())
                epoch_to = int((day + timedelta(days=window_days)).timestamp())
                rows = con.execute(
                    f"SELECT {cols} FROM articles "
                    "WHERE ticker=? AND published_epoch BETWEEN ? AND ? AND published_epoch > 0",
                    (ticker, epoch_from, epoch_to),
                ).fetchall()
            else:
                rows = con.execute(
                    f"SELECT {cols} FROM articles WHERE ticker=?", (ticker,)
                ).fetchall()
        keys = cols.replace(" ", "").split(",")
        return [dict(zip(keys, r)) for r in rows]

    def upsert_articles(self, articles: List[Dict]) -> List[Dict]:
        """Fügt Artikel ein (dedupliziert über `link`). Gibt nur die NEU eingefügten zurück.

        So embedded/indexiert die Pipeline ausschließlich neue Artikel.
        """
        new_articles = []
        now = datetime.now().isoformat()
        with self._connect() as con:
            for a in articles:
                link = a.get("link") or ""
                if not link:
                    continue  # ohne Link keine Deduplizierung möglich → überspringen
                epoch = published_to_epoch(a.get("published", ""))
                cur = con.execute(
                    "INSERT OR IGNORE INTO articles"
                    "(link, ticker, title, summary, published, published_epoch, source, indexed, fetched_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (link, a.get("ticker", ""), a.get("title", ""), a.get("summary", ""),
                     a.get("published", ""), epoch, a.get("source", ""), now),
                )
                if cur.rowcount == 1:
                    new_articles.append(a)
        return new_articles

    def mark_indexed(self, links: List[str]) -> None:
        """Markiert Artikel als in den Vektorindex aufgenommen."""
        with self._connect() as con:
            con.executemany("UPDATE articles SET indexed=1 WHERE link=?",
                            [(l,) for l in links if l])

    def all_articles(self, limit: int = 200) -> List[Dict]:
        """Alle gecachten Artikel (neueste zuerst) — für die Debug-/Nachrichtentabelle."""
        cols = "ticker, published, title, source, indexed, link"
        with self._connect() as con:
            rows = con.execute(
                f"SELECT {cols} FROM articles ORDER BY published_epoch DESC, ticker ASC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = cols.replace(" ", "").split(",")
        return [dict(zip(keys, r)) for r in rows]

    def count(self) -> int:
        with self._connect() as con:
            return con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
