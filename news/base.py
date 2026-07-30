"""
Kaynak arayüzü (NewsSource) — RAG için haber/veri kaynaklarının takılıp çıkarıldığı yer.

Yeni bir kaynak eklemek (ör. Finnhub API, NewsAPI, çeyrek raporu scraper'ı):
  1) Bu Protocol'ü uygulayan bir sınıf yaz (name + fetch()).
  2) get_sources() listesine bir satır ekle.
Pipeline'ın geri kalanı (chunking, embedding, arama) hiç değişmez.

Her kaynağın fetch()'i AYNI normalize şemada dict listesi döndürmelidir:
    {title, summary, link, published (ISO-String | ""), source, ticker}
"""
from typing import Protocol, List, Dict, Optional
from email.utils import parsedate_to_datetime
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class NewsSource(Protocol):
    """Bir haber kaynağının uyması gereken sözleşme.

    `start`/`end` en-iyi-çaba ipuçlarıdır: tarihsel erişimi olmayan kaynaklar
    (ör. saf RSS feed'i) bunları yok sayıp en güncel makaleleri döndürebilir.
    Tarihsel/tarih-aralıklı kaynaklar (çeyrek raporları, haber arşivleri)
    bunları kullanarak anomali gününe denk pencereyi hedefleyebilir.
    """
    name: str

    def fetch(self, ticker: str, limit: int = 10,
              start: Optional[datetime] = None,
              end: Optional[datetime] = None) -> List[Dict]:
        ...


def normalize_published(raw) -> str:
    """Herhangi bir published değerini ISO-8601 string'e çevirir ('' = parse edilemedi).

    RFC-822 (RSS, 'Mon, 25 Jul 2026 14:30:00 GMT') ve ISO formatlarını kapsar.
    """
    if not raw:
        return ""
    if isinstance(raw, datetime):
        return raw.isoformat()
    raw = str(raw).strip()
    try:
        dt = parsedate_to_datetime(raw)
        if dt is not None:
            return dt.isoformat()
    except (TypeError, ValueError):
        pass
    try:
        return datetime.fromisoformat(raw).isoformat()
    except ValueError:
        return ""


def published_to_epoch(iso_str: str) -> int:
    """ISO string → Unix saniye (int). Boş/geçersiz ise 0.

    0 sentinel'i: tarihsiz belgeler, tarih-aralığı filtreli sorgularda dışlanır.
    """
    if not iso_str:
        return 0
    try:
        return int(datetime.fromisoformat(iso_str).timestamp())
    except (ValueError, TypeError):
        return 0


def get_sources() -> List[NewsSource]:
    """Aktif kaynakların kayıt listesi. Yeni kaynak = burada bir satır."""
    from news.yahoo_fetcher import YahooRSSSource
    return [YahooRSSSource()]
