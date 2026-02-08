"""
Twitter/X collector using the v2 API via httpx.

Scans for cashtag mentions, financial influencer posts,
and volume spikes in ticker-related conversations.

Falls back to search-based collection if streaming is unavailable.
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx

from openclaw.social_arb.config import config, TICKER_EXCLUSION_LIST, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.signals.signal_schema import MentionData, SourceType
from openclaw.social_arb.utils.logger import get_logger
from openclaw.social_arb.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = get_logger(__name__)

TWITTER_API_BASE = "https://api.twitter.com/2"

CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")


class TwitterCollector:
    """
    Collects ticker mentions and sentiment from Twitter/X.

    Uses the v2 recent search endpoint to find cashtag mentions
    and high-engagement financial posts.
    """

    def __init__(self):
        self.bearer_token = config.api_keys.twitter_bearer_token
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit.twitter_requests_per_15min,
            period_seconds=900,  # 15 minutes
        )
        self.headers = {
            "Authorization": f"Bearer {self.bearer_token}",
            "Content-Type": "application/json",
        }
        # Financial cashtags to actively track
        self._tracked_tickers: list[str] = []

    async def _search_recent(
        self, query: str, max_results: int = 100
    ) -> list[dict[str, Any]]:
        """
        Search recent tweets using Twitter API v2.

        Args:
            query: Twitter search query (supports operators like $ for cashtags)
            max_results: Number of results (10-100 for recent search)
        """
        await self.rate_limiter.acquire()

        params = {
            "query": query,
            "max_results": min(max_results, 100),
            "tweet.fields": "created_at,public_metrics,author_id,lang",
            "expansions": "author_id",
            "user.fields": "created_at,public_metrics,verified",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{TWITTER_API_BASE}/tweets/search/recent",
                headers=self.headers,
                params=params,
            )
            resp.raise_for_status()
            data = resp.json()

        tweets = data.get("data", [])
        # Build author lookup from includes
        users = {
            u["id"]: u for u in data.get("includes", {}).get("users", [])
        }

        # Attach user data to tweets
        for tweet in tweets:
            author_id = tweet.get("author_id")
            if author_id and author_id in users:
                tweet["_author"] = users[author_id]

        return tweets

    def _extract_tickers_from_tweet(self, text: str) -> list[str]:
        """Extract cashtag tickers from tweet text."""
        tickers = set()
        for match in CASHTAG_PATTERN.findall(text):
            if match in KNOWN_CRYPTO_TICKERS or match not in TICKER_EXCLUSION_LIST:
                tickers.add(match)
        return list(tickers)

    def _tweet_to_mentions(self, tweet: dict[str, Any]) -> list[MentionData]:
        """Convert a single tweet into MentionData objects."""
        text = tweet.get("text", "")
        tickers = self._extract_tickers_from_tweet(text)

        if not tickers:
            return []

        metrics = tweet.get("public_metrics", {})
        author = tweet.get("_author", {})
        author_metrics = author.get("public_metrics", {})

        # Calculate author credibility signals
        followers = author_metrics.get("followers_count", 0)
        following = author_metrics.get("following_count", 1)
        follower_ratio = followers / max(following, 1)

        created_at = tweet.get("created_at")
        if created_at:
            timestamp = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            timestamp = datetime.now(timezone.utc)

        mentions = []
        for ticker in tickers:
            mentions.append(
                MentionData(
                    source=SourceType.TWITTER,
                    ticker=ticker,
                    keyword=text[:100],
                    mention_count=1,
                    timestamp=timestamp,
                    author=author.get("username", "unknown"),
                    text_snippet=text[:500],
                    upvotes=metrics.get("like_count", 0),
                    metadata={
                        "retweet_count": metrics.get("retweet_count", 0),
                        "reply_count": metrics.get("reply_count", 0),
                        "quote_count": metrics.get("quote_count", 0),
                        "impression_count": metrics.get("impression_count", 0),
                        "author_followers": followers,
                        "author_follower_ratio": follower_ratio,
                        "author_verified": author.get("verified", False),
                        "author_created_at": author.get("created_at"),
                        "tweet_id": tweet.get("id"),
                    },
                )
            )
        return mentions

    async def collect_cashtags(self, tickers: list[str] | None = None) -> list[MentionData]:
        """
        Search for cashtag mentions of specific tickers.

        If no tickers provided, searches for general financial cashtag activity.
        """
        mentions: list[MentionData] = []

        if tickers:
            # Batch tickers into groups (Twitter OR query has limits)
            batches = [tickers[i : i + 10] for i in range(0, len(tickers), 10)]
            for batch in batches:
                query = " OR ".join(f"${t}" for t in batch)
                query += " lang:en -is:retweet"
                try:
                    tweets = await retry_with_backoff(
                        self._search_recent,
                        query,
                        max_attempts=config.rate_limit.retry_max_attempts,
                        base_delay=config.rate_limit.retry_base_delay,
                    )
                    for tweet in tweets:
                        mentions.extend(self._tweet_to_mentions(tweet))
                except Exception as e:
                    logger.error(f"Twitter cashtag search failed: {e}")
        else:
            # General financial cashtag search
            query = "($ stock OR $ shares OR $ earnings OR $ buy OR $ sell) lang:en -is:retweet"
            try:
                tweets = await retry_with_backoff(
                    self._search_recent,
                    query,
                    max_attempts=config.rate_limit.retry_max_attempts,
                    base_delay=config.rate_limit.retry_base_delay,
                )
                for tweet in tweets:
                    mentions.extend(self._tweet_to_mentions(tweet))
            except Exception as e:
                logger.error(f"Twitter general search failed: {e}")

        logger.info(
            f"Twitter collection: {len(mentions)} mentions",
            extra={"data": {"count": len(mentions)}},
        )
        return mentions

    async def collect_trending_financial(self) -> list[MentionData]:
        """Search for high-engagement financial tweets that might indicate trends."""
        mentions: list[MentionData] = []
        queries = [
            '"short squeeze" lang:en -is:retweet',
            '"to the moon" (stock OR crypto) lang:en -is:retweet',
            '"breaking" (earnings OR FDA OR SEC) lang:en -is:retweet',
        ]

        for query in queries:
            try:
                tweets = await retry_with_backoff(
                    self._search_recent,
                    query,
                    max_results=50,
                    max_attempts=config.rate_limit.retry_max_attempts,
                    base_delay=config.rate_limit.retry_base_delay,
                )
                for tweet in tweets:
                    mentions.extend(self._tweet_to_mentions(tweet))
            except Exception as e:
                logger.error(f"Twitter trending financial search failed: {e}")
                continue

        return mentions

    async def collect(self) -> list[MentionData]:
        """Run all Twitter collection strategies concurrently."""
        cashtag_task = self.collect_cashtags(self._tracked_tickers or None)
        trending_task = self.collect_trending_financial()

        results = await asyncio.gather(cashtag_task, trending_task, return_exceptions=True)

        all_mentions: list[MentionData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Twitter collection failed: {result}")
                continue
            all_mentions.extend(result)

        return all_mentions

    def track_tickers(self, tickers: list[str]) -> None:
        """Add tickers to the active tracking list for cashtag searches."""
        self._tracked_tickers = list(set(self._tracked_tickers + tickers))
