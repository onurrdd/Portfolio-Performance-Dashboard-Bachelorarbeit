"""
Yahoo Finance News Fetcher - Retrieves financial news from Yahoo Finance RSS feeds
"""
import feedparser
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class YahooFinanceFetcher:
    """
    Fetches financial news from Yahoo Finance RSS feeds for specific tickers.
    Falls back to HTML scraping if RSS fails.
    """

    BASE_RSS_URL = "https://feeds.finance.yahoo.com/rss/2.0/headline"
    BASE_NEWS_URL = "https://finance.yahoo.com/quote"
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

    def fetch_news_for_ticker(self, ticker: str, limit: int = 10) -> List[Dict]:
        """
        Fetch latest news for a specific ticker.
        
        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            limit: Maximum number of news articles to fetch
            
        Returns:
            List of dictionaries with keys: title, summary, link, published, source, ticker
        """
        news_articles = []

        # Try RSS feed first
        try:
            rss_url = f"{self.BASE_RSS_URL}?s={ticker}"
            logger.info(f"Fetching news for {ticker} from RSS: {rss_url}")
            feed = feedparser.parse(rss_url)

            if feed.entries:
                for entry in feed.entries[:limit]:
                    try:
                        article = {
                            "title": entry.get("title", ""),
                            "summary": entry.get("summary", ""),
                            "link": entry.get("link", ""),
                            "published": entry.get("published", str(datetime.now())),
                            "source": "Yahoo Finance RSS",
                            "ticker": ticker,
                        }
                        news_articles.append(article)
                    except Exception as e:
                        logger.warning(f"Error parsing RSS entry for {ticker}: {e}")
                        continue

                logger.info(f"Successfully fetched {len(news_articles)} articles for {ticker} from RSS")
                return news_articles

        except Exception as e:
            logger.warning(f"RSS fetch failed for {ticker}: {e}. Trying alternative method...")

        # Fallback: Try to scrape from Yahoo Finance news page
        try:
            news_url = f"{self.BASE_NEWS_URL}/{ticker}/news"
            logger.info(f"Attempting fallback scrape from {news_url}")
            response = self.session.get(news_url, timeout=10)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")
            news_items = soup.find_all("a", {"data-test": "quoteNewsLink"})

            for item in news_items[:limit]:
                try:
                    title = item.get_text(strip=True)
                    link = item.get("href", "")

                    if not link.startswith("http"):
                        link = f"https://finance.yahoo.com{link}"

                    if title:
                        article = {
                            "title": title,
                            "summary": f"News from Yahoo Finance about {ticker}",
                            "link": link,
                            "published": str(datetime.now()),
                            "source": "Yahoo Finance (Scraped)",
                            "ticker": ticker,
                        }
                        news_articles.append(article)
                except Exception as e:
                    logger.warning(f"Error parsing scraped news item: {e}")
                    continue

            logger.info(f"Successfully scraped {len(news_articles)} articles for {ticker}")
            return news_articles

        except Exception as e:
            logger.error(f"Both RSS and scrape failed for {ticker}: {e}")
            return []

    def fetch_news_for_tickers(self, tickers: List[str], limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Fetch news for multiple tickers.
        
        Args:
            tickers: List of ticker symbols
            limit: Max articles per ticker
            
        Returns:
            Dictionary mapping ticker to list of news articles
        """
        all_news = {}
        for ticker in tickers:
            all_news[ticker] = self.fetch_news_for_ticker(ticker, limit)
        return all_news


def fetch_news_batch(tickers: List[str], limit: int = 5) -> Dict[str, List[Dict]]:
    """
    Convenience function to fetch news for multiple tickers.

    Args:
        tickers: List of ticker symbols
        limit: Max articles per ticker

    Returns:
        Dictionary mapping ticker to list of articles
    """
    fetcher = YahooFinanceFetcher()
    return fetcher.fetch_news_for_tickers(tickers, limit)


class YahooRSSSource:
    """NewsSource-Adapter für den bestehenden YahooFinanceFetcher.

    Macht den Fetcher zu einer austauschbaren Quelle (news.base.NewsSource) und
    normalisiert das published-Feld (RSS RFC-822 → ISO). RSS liefert nur die
    neuesten Artikel, daher werden start/end ignoriert.
    """
    name = "yahoo_rss"

    def __init__(self):
        self._fetcher = YahooFinanceFetcher()

    def fetch(self, ticker: str, limit: int = 10, start=None, end=None) -> List[Dict]:
        from news.base import normalize_published
        articles = self._fetcher.fetch_news_for_ticker(ticker, limit)
        for a in articles:
            a["published"] = normalize_published(a.get("published"))
        return articles
