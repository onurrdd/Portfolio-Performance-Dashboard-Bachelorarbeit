# Portfolio Dashboard with RAG Integration

Professionelle Portfolio-Performance-Dashboard mit AI-gestützter Analyse und Yahoo Finance Nachrichten-RAG Retrieval.

## Features

- 📊 **Portfolio Management**: Positionen hinzufügen, verkaufen, oder CSV importieren
- 📈 **Advanced Metrics**: Sortino Ratio, Calmar Ratio, Rolling Sharpe, Max Drawdown
- 📉 **Detaillierte Analyse**: Korrelationsmatrix, Asset Allocation, Benchmark Vergleich
- 🤖 **AI-gestützte Risikoanalyse**: Groq LLM für Portfolio-Feedback
- 📰 **RAG News Integration**: Yahoo Finance Nachrichten mit Retrieval-Augmented Generation

## Installation

### Anforderungen
- Python 3.8+
- pip oder conda
- GROQ_API_KEY (für LLM Analysen) in `.env`

### Setup

1. **Clone/Setup Projekt**
```bash
cd "c:\Users\darende\Documents\11. Semester\Bachelorarbeit\BaseforBachelorArbeit"
```

2. **Virtual Environment (optional)**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

3. **Abhängigkeiten installieren**
```bash
pip install -r requirements.txt
```

4. **Environment Variablen konfigurieren**

Erstelle `.env` Datei im Projekt-Verzeichnis:
```
GROQ_API_KEY=your_groq_api_key_here
```

**GROQ_API_KEY bekommen:**
- Gehe zu https://console.groq.com
- Registrieren/Login
- API Key generieren
- In `.env` einfügen

5. **Dashboard starten**
```bash
python dashboard.py
```

Dashboard öffnet sich unter `http://localhost:8050`

---

## Verwendung

### Portfolio Management

#### Position hinzufügen
1. Gehe zum **Overview** Tab
2. Gib Ticker, Anzahl Aktien, und Kaufdatum ein
3. Klick "Hinzufügen"

**Beispiel:**
- Ticker: `AAPL`
- Shares: `10`
- Buy Date: `2022-01-01`

#### Positionen verkaufen
1. Nutze "Verkaufen" Toggle
2. Gib Ticker und Anzahl zu verkaufender Aktien ein
3. Klick "Verkaufen"

#### Portfolio aus CSV laden
CSV muss folgende Spalten enthalten:
```
ticker,shares,buy_date[,buy_price]
AAPL,10,2022-01-01
MSFT,5,2023-06-15
TSLA,3,2024-01-10
```

### Dashboard Tabs

#### 1. Overview
- Aktuelle Portfolio-Positionen
- Gesamte P&L Metriken
- Portfolio-Wert Entwicklung über Zeit

#### 2. Analytics
- Rolling Sharpe Ratio (120 Tage)
- Drawdown Analyse
- Asset Allocation (Pie Chart)
- Korrelationsmatrix

#### 3. Benchmark Comparison
- Portfolio vs. S&P 500 (SPY) Vergleich
- Outperformance Metriken
- Volatilitäts Vergleich

#### 4. AI Risk Analysis
- Groq LLM basierte Portfolio-Bewertung
- Automatische Risikoanalyse
- Performance-Feedback

#### 5. Nachrichten & RAG (NEU)
Yahoo Finance Nachrichten mit RAG-Retrieval.

**Schritt 1: Nachrichten indizieren**
1. Gib Ticker komma-getrennt ein (z.B. `AAPL,MSFT,TSLA`)
2. Setze Limit für Artikel pro Ticker (default: 5)
3. Klick "Nachrichten abrufen & indizieren"
4. Warte bis Erfolg-Nachricht angezeigt wird

**Schritt 2: Fragen stellen**
1. Gib eine Frage ein (z.B. "Was sind die neuesten Nachrichten über Apple?")
2. Setze Top-K Ergebnisse (default: 5)
3. Klick "Mit RAG abfragen"
4. RAG liefert:
   - Groq LLM Antwort mit Kontext
   - Top-K relevante Nachrichten mit Links

---

## RAG System Details

### Architektur

```
Hacker Flow:
1. News Fetch (Yahoo Finance RSS + Scraping)
   ↓
2. Text Chunking (400 char chunks, 50 char overlap)
   ↓
3. Embedding Generation (sentence-transformers: all-MiniLM-L6-v2)
   ↓
4. Vector Storage (Chroma: local file-based persistence)
   ↓
5. Retrieval & Reranking (cosine similarity, top-k)
   ↓
6. LLM Augmentation (Groq: llama-3.3-70b-versatile)
```

### Komponenten

#### Yahoo Finance Fetcher (`news/yahoo_fetcher.py`)
- Holt Nachrichten von Yahoo Finance RSS Feeds
- Fallback: HTML Scraping falls RSS fehlschlägt
- Output: News Dictionaries mit Title, Summary, Link, Published, Ticker

#### Text Chunker (`rag/chunker.py`)
- Teilt lange Texte in überlappende Chunks
- Normalisiert und bereinigt Text
- Preserviert Metadaten (Ticker, Link, Source)

#### Embedding Generator (`rag/embeddings.py`)
- Nutzt `sentence-transformers` (lokal, offline)
- Modell: `all-MiniLM-L6-v2` (384-dim, lightweight)
- Keine API Keys notwendig

#### Chroma Vector Store (`rag/vectorstore.py`)
- Persistente Speicherung in `./data/chroma/`
- Kosinus Distanz für Similarity
- Collection: `news_articles`

#### RAG Pipeline (`rag/pipeline.py`)
- Orchestriert alle Komponenten
- `index_news_for_tickers()` - indiziere Nachrichten
- `retrieve_context()` - hole relevante Chunks
- `query_with_rag()` - end-to-end RAG Query

### Konfiguration

**Persistenz**: `./data/chroma/` (wird automatisch erstellt)

**Modelle**:
- Embedding: `all-MiniLM-L6-v2` (384-dimensional)
- LLM: `llama-3.3-70b-versatile` (via Groq)

**Chunk Parameter** (in `rag/pipeline.py`):
```python
chunk_size=400  # Characters
overlap=50      # Characters overlap
```

**Retrieval Parameter**:
```python
top_k=5         # Default: top 5 chunks
max_tokens=2000 # Context für LLM
```

---

## Performance & Skalierung

### Gegenwärtig (Stand Mai 2026)
- ✅ Hacker Architektur (RSS + Scraping, keine API Kosten)
- ✅ Lokal speicherbar (Chroma)
- ✅ Offline Embedding (sentence-transformers)
- ✅ Schnelle Retrieval (<1 Sekunde)

### Zukünftige Skalierungen
Falls Anzahl Nachrichten > 10,000 oder Performance nötig:

1. **Embedding Provider wechseln**:
   - OpenAI Embeddings (besser, kostet $)
   - Cohere API

2. **Vector DB wechseln**:
   - Pinecone (managed, cloud-hosted)
   - Weaviate (open-source, flexible)
   - Milvus (high-performance)

3. **News Source erweitern**:
   - ResearchAPI, Finnhub, NewsAPI
   - SEC EDGAR Filings
   - Telegram/Slack Channels

4. **Pipeline erweitern**:
   - Scheduled news fetching (cron job)
   - Real-time streaming (WebSocket)
   - Sentiment Analysis (VADER, Transformers)
   - Named Entity Recognition (NER)

---

## Fehlerbehebung

### Problem: "RAG Pipeline nicht verfügbar"
**Ursache**: RAG Module nicht installiert oder Fehler beim Import

**Lösung**:
```bash
pip install --upgrade chromadb sentence-transformers
python -c "from rag.pipeline import RAGPipeline; print('OK')"
```

### Problem: Keine Nachrichten abgerufen
**Ursache**: Yahoo Finance RSS/Scraping blockiert oder Ticker ungültig

**Lösung**:
1. Prüfe Ticker Rechtschreibung
2. Kontrolliere Internet-Verbindung
3. Prüfe Logs in Console
4. Versuche mit anderem Ticker (z.B. `AAPL`)

### Problem: Embedding Generation sehr langsam
**Ursache**: Modell wird beim ersten Mal heruntergeladen (~500 MB)

**Lösung**: Warten Sie beim ersten Run oder pre-download:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Problem: "GROQ_API_KEY nicht gesetzt"
**Ursache**: `.env` Datei nicht gefunden oder falsch formatiert

**Lösung**:
1. Erstelle `.env` im Projekt-Verzeichnis
2. Format: `GROQ_API_KEY=your_key_here` (keine Anführungszeichen)
3. Speichere und starte Dashboard neu

### Problem: Chroma Datenbankfehler
**Ursache**: Korrupte Daten im `./data/chroma/`

**Lösung**:
```bash
# Lösche alte Datenbank
rmdir /s ./data/chroma
# Neu indizieren
```

---

## API & Vorsicht

### Yahoo Finance
- **Rate Limit**: ~2000 Requests/Stunde (RSS) oder ~100/Minute (Scraping)
- **ToS**: Scraping ist legal wenn nicht exzessiv (siehe [tos.finance.yahoo.com](https://legal.yahoo.com/us/en/yahoo/terms/))
- **Datenqualität**: RSS kann Verzögerungen haben; Scraping ist aktueller

### Groq API
- **Free Tier**: ~1 Million Tokens/Tag
- **Rate Limit**: ~30 Requests/Minute
- **Model**: `llama-3.3-70b-versatile` (aktuell August 2024)

---

## Entwicklung

### Struktur
```
dashboard.py          # Dash Web App
news/
  ├── __init__.py
  └── yahoo_fetcher.py  # Holt Nachrichten
rag/
  ├── __init__.py
  ├── chunker.py        # Text Chunking
  ├── embeddings.py     # Embedding Gen
  ├── vectorstore.py    # Chroma Adapter
  └── pipeline.py       # Orchestration
data/
  └── chroma/           # Vector DB (persistent)
requirements.txt       # Dependencies
.env                   # API Keys
README.md             # Diese Datei
```

### Logging
```python
import logging
logger = logging.getLogger(__name__)
logger.info("Info Message")
logger.warning("Warning")
logger.error("Error")
```

---

## Lizenz & Disclaimers

⚠️ **Haftungsausschluss**:
- Dieses Dashboard ist zu Bildungszwecken gedacht
- Keine Finanzberatung
- Teste gründlich bevor du es mit echtem Geld nutzt
- Yahoo Finance ToS beachten

---

## Support

Bei Fragen/Issues:
1. Prüfe Logs in der Console
2. Aktiviere Debug Mode: `app.run(debug=True, ...)`
3. Konfiguriere Logging Level
4. Öffne Issue auf GitHub

---

**Viel Spaß beim Investieren! 📈**
