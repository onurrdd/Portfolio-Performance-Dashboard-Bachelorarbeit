# Portfolio Performance Dashboard — Mimari Özeti

## Ne yapar?

Hisse senedi portföyünü analiz eden, risk metrikleri hesaplayan ve Yahoo Finance haberleriyle RAG destekli soru-cevap yapabilen bir **Dash web uygulaması**.

Çalıştır: `python dashboard.py` → http://localhost:8050

---

## Dosya Yapısı

```
dashboard.py          ← Uygulama giriş noktası, layout, RAG init
auto_load.py          ← Startup'ta portfolio_5_ticker.csv'yi otomatik yükler

callbacks/
  __init__.py         ← register_all(app, rag_pipeline) — hepsini bağlar
  portfolio.py        ← Pozisyon ekle/sil/CSV yükle/temizle
  charts.py           ← Tüm grafik ve metrik hesaplamaları (en büyük dosya)
  ai_analysis.py      ← Groq LLM ile portföy analizi
  rag.py              ← Haber indeksleme ve RAG sorgusu

utils/
  finance.py          ← yfinance yardımcıları (fiyat çek, split düzelt, zaman serisi)
  metrics.py          ← Finansal metrik hesaplamaları (Sortino, Calmar, TWR, vb.)

news/
  yahoo_fetcher.py    ← Yahoo Finance RSS + HTML fallback ile haber çekme

rag/
  pipeline.py         ← RAGPipeline ana sınıfı (orkestratör)
  chunker.py          ← Haberleri ~400 char parçalara böler
  embeddings.py       ← sentence-transformers ile yerel embedding (all-MiniLM-L6-v2, 384-dim)
  vectorstore.py      ← FAISS index (index.faiss + metadata.pkl olarak disk'e kaydeder)

data/faiss/           ← FAISS kalıcı depolama (otomatik oluşturulur)
portfolio_5_ticker.csv ← Startup'ta otomatik yüklenen örnek portföy
```

---

## Veri Akışı

### 1. Portföy Yönetimi
```
Kullanıcı girişi (ticker, adet, tarih)
  → callbacks/portfolio.py
  → yfinance'den tarihteki fiyatı çek (utils/finance.py)
  → dcc.Store('portfolio-store') içine kaydet (tarayıcı hafızası)
```

### 2. Dashboard Güncelleme
```
portfolio-store değişince
  → callbacks/charts.py: update_dashboard()
  → utils/finance.py: calculate_portfolio_timeseries() → günlük portföy değeri
  → utils/metrics.py: calculate_twr_metrics() → TWR bazlı metrikler
    (Time-Weighted Return: nakit akışı etkisini elimine eder)
  → 6 grafik + metrik kartlar + analysis-data store güncellenir
```

### 3. AI Analizi
```
"Portfolio analysieren" butonu
  → callbacks/ai_analysis.py
  → analysis-data store'dan metrikleri al (charts.py'nin doldurduğu)
  → Groq API (llama-3.3-70b) → Türkçe analiz metni
```

### 4. RAG Pipeline
```
Adım 1 — İndeksleme:
  Ticker girişi → news/yahoo_fetcher.py (RSS önce, HTML fallback)
    → rag/chunker.py: metin ~400 char parçalara bölünür
    → rag/embeddings.py: her chunk için 384-dim vektör (yerel model, offline)
    → rag/vectorstore.py: FAISS'e ekle, disk'e kaydet

Adım 2 — Sorgulama:
  Kullanıcı sorusu → embedding → FAISS L2 mesafe araması
    → Top-K chunk'lar → context string oluştur
    → Groq API: context + soru → cevap
    → Kaynak linklerle birlikte göster
```

---

## Tab'lar ve İçerikleri

| Tab | Gösterir |
|-----|----------|
| Overview | Pozisyon tablosu, P&L özeti, Sortino/Calmar/Drawdown/Volatility kartları, portföy değer grafiği |
| Analytics | Rolling Sharpe (120 gün), Drawdown grafiği, Allocation pie chart, Korelasyon ısı haritası |
| Benchmark Comparison | Portföy vs S&P 500 (SPY), normalize karşılaştırma, outperformance hesabı |
| AI Risk Analysis | Groq LLM ile risk analizi (6 soru formatında) |
| Nachrichten & RAG | Yahoo haber indeksleme + FAISS araması + Groq cevabı |

---

## Önemli Teknik Detaylar

### Split Düzeltmesi
`utils/finance.py` → `adjust_shares_for_splits()` ve `adjust_price_for_splits()`
Alış tarihinden sonra gerçekleşen hisse bölünmelerini (stock split) otomatik uygular.
Örnek: 10 hisse alındıktan sonra 2:1 split → 20 hisse gösterilir.

### TWR (Time-Weighted Return)
`utils/metrics.py` → `calculate_twr_metrics()`
Her pozisyon ekleme tarihi bir dönem sınırı sayılır. Nakit akışlarından bağımsız performans ölçer.
TWR hesaplanamazsa basit portföy değeri zaman serisi fallback olarak kullanılır.

### RAG Başlatma
`dashboard.py` satır 34-45: RAG pipeline arka planda thread'de başlar, uygulamayı bloklamaz.
Embedding modeli (`all-MiniLM-L6-v2`) ilk çalıştırmada indirilir (~80MB), sonrası cache'den gelir.

### Auto-Load
`auto_load.py`: Uygulama açıldıktan 500ms sonra (dcc.Interval, max_intervals=1) `portfolio_5_ticker.csv` varsa otomatik yüklenir.

### Durum Yönetimi (State)
Tüm portföy verisi `dcc.Store` ile tarayıcıda tutulur — sunucuda oturum yok.
- `portfolio-store`: Ham pozisyon listesi
- `analysis-data`: Hesaplanmış metrikler (AI analizi için)
- `rag-status`: İndekslenen ticker ve belge sayısı

---

## API Bağımlılıkları

| Servis | Kullanım | Key |
|--------|----------|-----|
| yfinance | Fiyat verileri, split tarihleri | Yok (ücretsiz) |
| Groq API | AI analizi + RAG cevapları (llama-3.3-70b) | `GROQ_API_KEY` env var |
| Yahoo Finance RSS | Haber çekme | Yok |

---

## Metrikler Referansı

- **Sortino Ratio**: Sadece negatif volatiliteyi cezalandıran risk/getiri oranı. >1 iyi, >2 çok iyi.
- **Calmar Ratio**: Yıllık getiri / Max Drawdown. >0.5 makul.
- **Max Drawdown**: Tepe değerden en büyük düşüş. Kayıp toleransını ölçer.
- **Rolling Sharpe (120g)**: 120 günlük pencerede hesaplanan Sharpe oranı. Performans tutarlılığını gösterir.
- **TWR**: Nakit akışı olmayan dönemler birleştirilerek hesaplanan getiri. Fon yöneticisi performansı değerlendirmede standart.
