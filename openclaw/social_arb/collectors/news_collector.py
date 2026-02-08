"""
News collector using NewsAPI and RSS feed fallback.

Aggregates financial news headlines and articles to detect
breaking stories that could move markets before they're priced in.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any
from xml.etree import ElementTree

import httpx

from openclaw.social_arb.config import config, TICKER_EXCLUSION_LIST, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.signals.signal_schema import MentionData, SourceType
from openclaw.social_arb.utils.logger import get_logger
from openclaw.social_arb.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = get_logger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"

CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
ALLCAPS_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")

# Financial RSS feeds for fallback when NewsAPI quota is exceeded
RSS_FEEDS: list[dict[str, str]] = [
    {"name": "Reuters Business", "url": "https://feeds.reuters.com/reuters/businessNews"},
    {"name": "CNBC Top News", "url": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114"},
    {"name": "MarketWatch", "url": "https://feeds.marketwatch.com/marketwatch/topstories/"},
    {"name": "Yahoo Finance", "url": "https://finance.yahoo.com/news/rssindex"},
    {"name": "Seeking Alpha", "url": "https://seekingalpha.com/market_currents.xml"},
]


class NewsCollector:
    """
    Aggregates financial news from multiple sources.

    Primary: NewsAPI for comprehensive coverage with search
    Fallback: RSS feeds for zero-cost continuous monitoring
    """

    def __init__(self):
        self.api_key = config.api_keys.newsapi_key
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit.newsapi_requests_per_day,
            period_seconds=86400,  # 24 hours
        )

    def _extract_tickers(self, text: str) -> list[str]:
        """Extract ticker symbols from news text."""
        tickers: set[str] = set()
        for match in CASHTAG_PATTERN.findall(text):
            tickers.add(match)
        for match in ALLCAPS_PATTERN.findall(text):
            if match in KNOWN_CRYPTO_TICKERS or match not in TICKER_EXCLUSION_LIST:
                tickers.add(match)
        return list(tickers)

    async def _fetch_newsapi(
        self,
        query: str | None = None,
        category: str = "business",
        page_size: int = 50,
    ) -> list[dict[str, Any]]:
        """
        Fetch articles from NewsAPI.

        Uses 'everything' endpoint for queries, 'top-headlines' for categories.
        """
        await self.rate_limiter.acquire()

        async with httpx.AsyncClient(timeout=30) as client:
            if query:
                params = {
                    "q": query,
                    "language": "en",
                    "sortBy": "publishedAt",
                    "pageSize": page_size,
                    "apiKey": self.api_key,
                }
                resp = await client.get(f"{NEWSAPI_BASE}/everything", params=params)
            else:
                params = {
                    "category": category,
                    "language": "en",
                    "country": "us",
                    "pageSize": page_size,
                    "apiKey": self.api_key,
                }
                resp = await client.get(f"{NEWSAPI_BASE}/top-headlines", params=params)

            resp.raise_for_status()
            data = resp.json()

        return data.get("articles", [])

    async def _fetch_rss(self, feed: dict[str, str]) -> list[dict[str, Any]]:
        """Parse an RSS feed and return articles as dicts."""
        articles: list[dict[str, Any]] = []
        try:
            async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
                resp = await client.get(feed["url"])
                resp.raise_for_status()

            root = ElementTree.fromstring(resp.text)
            # Handle both RSS 2.0 and Atom formats
            items = root.findall(".//item") or root.findall(
                ".//{http://www.w3.org/2005/Atom}entry"
            )

            for item in items[:30]:  # limit per feed
                title = ""
                description = ""
                link = ""
                pub_date = ""

                title_el = item.find("title")
                if title_el is not None and title_el.text:
                    title = title_el.text

                desc_el = item.find("description")
                if desc_el is not None and desc_el.text:
                    description = desc_el.text

                link_el = item.find("link")
                if link_el is not None and link_el.text:
                    link = link_el.text

                date_el = item.find("pubDate")
                if date_el is not None and date_el.text:
                    pub_date = date_el.text

                articles.append({
                    "title": title,
                    "description": description,
                    "url": link,
                    "publishedAt": pub_date,
                    "source": {"name": feed["name"]},
                })
        except Exception as e:
            logger.error(
                f"RSS feed fetch failed: {feed['name']}",
                extra={"data": {"feed": feed["name"], "error": str(e)}},
            )
        return articles

    def _article_to_mentions(self, article: dict[str, Any]) -> list[MentionData]:
        """Convert a news article into MentionData objects."""
        title = article.get("title", "") or ""
        description = article.get("description", "") or ""
        text = f"{title} {description}"
        tickers = self._extract_tickers(text)

        # Even if no explicit ticker, the keyword is useful for ticker_mapper
        keyword = title[:200] if title else None

        if not tickers and not keyword:
            return []

        published = article.get("publishedAt", "")
        try:
            if published and "T" in published:
                timestamp = datetime.fromisoformat(published.replace("Z", "+00:00"))
            else:
                timestamp = datetime.now(timezone.utc)
        except (ValueError, TypeError):
            timestamp = datetime.now(timezone.utc)

        source_name = article.get("source", {}).get("name", "unknown")

        # If we found tickers, create one mention per ticker
        if tickers:
            return [
                MentionData(
                    source=SourceType.NEWS,
                    ticker=ticker,
                    keyword=keyword,
                    mention_count=1,
                    timestamp=timestamp,
                    permalink=article.get("url"),
                    text_snippet=text[:500],
                    metadata={
                        "news_source": source_name,
                        "author": article.get("author"),
                    },
                )
                for ticker in tickers
            ]
        else:
            # No ticker but potentially mappable keyword
            return [
                MentionData(
                    source=SourceType.NEWS,
                    keyword=keyword,
                    mention_count=1,
                    timestamp=timestamp,
                    permalink=article.get("url"),
                    text_snippet=text[:500],
                    metadata={
                        "news_source": source_name,
                        "author": article.get("author"),
                        "needs_ticker_mapping": True,
                    },
                )
            ]

    async def collect_newsapi(self, queries: list[str] | None = None) -> list[MentionData]:
        """
        Collect from NewsAPI — both headlines and targeted queries.

        Default queries cover major market-moving categories.
        """
        mentions: list[MentionData] = []

        if not self.api_key:
            logger.warning("NewsAPI key not configured, skipping NewsAPI collection")
            return mentions

        default_queries = queries or [
            "stock market earnings",
            "FDA approval drug",
            "SEC investigation",
            "IPO filing",
            "short squeeze",
            "cryptocurrency regulation",
        ]

        # Fetch top headlines
        try:
            headlines = await retry_with_backoff(
                self._fetch_newsapi,
                max_attempts=config.rate_limit.retry_max_attempts,
                base_delay=config.rate_limit.retry_base_delay,
            )
            for article in headlines:
                mentions.extend(self._article_to_mentions(article))
        except Exception as e:
            logger.error(f"NewsAPI headlines failed: {e}")

        # Fetch query-specific articles
        for query in default_queries:
            try:
                articles = await retry_with_backoff(
                    self._fetch_newsapi,
                    query,
                    max_attempts=config.rate_limit.retry_max_attempts,
                    base_delay=config.rate_limit.retry_base_delay,
                )
                for article in articles:
                    mentions.extend(self._article_to_mentions(article))
            except Exception as e:
                logger.error(f"NewsAPI query '{query}' failed: {e}")
                continue

        return mentions

    async def collect_rss(self) -> list[MentionData]:
        """Collect from all configured RSS feeds concurrently."""
        tasks = [self._fetch_rss(feed) for feed in RSS_FEEDS]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        mentions: list[MentionData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"RSS collection failed: {result}")
                continue
            for article in result:
                mentions.extend(self._article_to_mentions(article))

        return mentions

    async def collect(self) -> list[MentionData]:
        """Run both NewsAPI and RSS collection."""
        newsapi_task = self.collect_newsapi()
        rss_task = self.collect_rss()

        results = await asyncio.gather(newsapi_task, rss_task, return_exceptions=True)

        all_mentions: list[MentionData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"News collection failed: {result}")
                continue
            all_mentions.extend(result)

        logger.info(
            f"News collection complete: {len(all_mentions)} mentions",
            extra={"data": {"total_mentions": len(all_mentions)}},
        )
        return all_mentions
