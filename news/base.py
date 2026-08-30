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

    `supports_date_range` bu ayrımı kaynağın KENDİSİNİN bildirdiği bir yetenek
    (Fähigkeit) haline getirir; böylece pipeline, geçmişe ait bir pencere için
    tarihsel erişimi olmayan kaynakları hiç çağırmaz (bkz. RAGPipeline). Bayrak
    olmadan bu bilgi yalnızca dolaylıydı: kaynak start/end'i sessizce yok sayıp
    güncel haberi döndürüyor, sonuç da pencere filtresinde eleniyordu — yani
    her tarihsel anomali için boşa bir ağ isteği yapılıyordu.
    """
    name: str
    # False = kaynak yalnızca "şu an"ı biliyor (start/end'i yok sayar).
    supports_date_range: bool

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
    """Aktif kaynakların kayıt listesi. Yeni kaynak = burada bir satır.

    Sıralama önemli değil (pipeline hepsinden toplar), ama Yahoo en ucuz/hızlı
    olduğu için önce dener. SEC EDGAR ve Alpha Vantage anahtar/ayar yoksa kendini
    otomatik devre dışı bırakır (bkz. .env.example) — sistem yine de çalışır.
    Alpha Vantage ayrıca RAGConfig.alpha_vantage_enabled bayrağıyla kontenjan
    koruması (Kontingentschutz) altındadır: bayrak kapalıyken kaynak hiçbir
    istek göndermez.
    """
    from news.yahoo_fetcher import YahooRSSSource
    from news.sec_edgar import SECEdgarSource
    from news.alpha_vantage import AlphaVantageNewsSource
    from rag.config import DEFAULT_CONFIG

    return [
        YahooRSSSource(),
        SECEdgarSource(forms=DEFAULT_CONFIG.sec_forms,
                       max_document_chars=DEFAULT_CONFIG.max_document_chars),
        AlphaVantageNewsSource(),
    ]


def clean_url(url: str) -> str:
    """Entfernt Tracking-/Session-Parameter aus einer Artikel-URL.

    Begruendung: Die URL wandert als `Link:`-Zeile in den Prompt. Bei manchen
    Nachrichtenportalen bestehen die Query-Parameter fast vollstaendig aus
    Tracking-Tokens (gaa_at, gaa_sig, ...) und werden dadurch laenger als der
    Artikeltext selbst — reiner Token-Verbrauch ohne Informationswert fuer das
    Modell. Der Pfad bleibt erhalten, die Quelle also identifizierbar und
    zitierbar; nur der Parameteranteil faellt weg.
    """
    if not url:
        return ""
    base = url.split("?", 1)[0].split("#", 1)[0]
    return base or url


def source_label(link: str, source: str) -> str:
    """Kaynak etiketi: prompt'taki `Source:` satırına yazılacak değer.

    Alpha Vantage bir haber AGREGATÖRÜ; `source` alanı ("alpha_vantage") gerçek
    yayın organı değil ve modele kaynak olarak verilirse doğrulanamaz bir ad
    üretir. Gerçek yayıncı yalnızca URL'nin alan adında saklı — onu çıkarıp
    döndürüyoruz (marketwatch.com, nytimes.com, ...). SEC EDGAR ise anlamlı bir
    kaynak adı, olduğu gibi bırakılır. Link yoksa `source` değerine düşer.
    """
    if source == "sec_edgar":
        return "SEC EDGAR filing"
    base = clean_url(link)
    if not base:
        return source
    netloc = base.split("://", 1)[-1].split("/", 1)[0]
    return netloc[4:] if netloc.startswith("www.") else (netloc or source)
