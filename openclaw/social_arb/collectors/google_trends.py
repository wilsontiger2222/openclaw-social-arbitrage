"""
Google Trends monitor using pytrends.

Two modes of operation:
1. Watchlist mode — track known tickers/products for sudden interest spikes
2. Discovery mode — scan trending searches for anything mappable to tradeable assets

Compares current interest to 30-day baselines to detect anomalous spikes.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone, timedelta
from typing import Any

from pytrends.request import TrendReq

from openclaw.social_arb.config import config, GOOGLE_TRENDS_SEED_KEYWORDS
from openclaw.social_arb.signals.signal_schema import MentionData, SourceType
from openclaw.social_arb.utils.logger import get_logger
from openclaw.social_arb.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = get_logger(__name__)


class GoogleTrendsCollector:
    """
    Monitors Google Trends for sudden interest spikes in
    financial keywords and trending searches.
    """

    def __init__(self):
        self.pytrends = TrendReq(hl="en-US", tz=360)
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit.google_trends_requests_per_hour,
            period_seconds=3600,
        )
        self.watchlist = config.trends_keywords.copy()
        # Spike threshold: current interest must be >2x the 30-day average
        self.spike_multiplier = 2.0

    async def _get_interest_over_time(
        self, keywords: list[str], timeframe: str = "now 7-d"
    ) -> dict[str, float]:
        """
        Get interest-over-time data for a batch of keywords (max 5 per API call).

        Returns the most recent interest score for each keyword.
        """
        await self.rate_limiter.acquire()
        loop = asyncio.get_event_loop()

        def _fetch():
            self.pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo="US")
            df = self.pytrends.interest_over_time()
            if df.empty:
                return {}
            # Get the most recent row's values
            latest = df.iloc[-1]
            return {kw: float(latest.get(kw, 0)) for kw in keywords if kw in df.columns}

        try:
            return await loop.run_in_executor(None, _fetch)
        except Exception as e:
            logger.error(f"Google Trends fetch failed: {e}")
            return {}

    async def _get_30day_baseline(self, keywords: list[str]) -> dict[str, float]:
        """Get 30-day average interest for baseline comparison."""
        await self.rate_limiter.acquire()
        loop = asyncio.get_event_loop()

        def _fetch():
            self.pytrends.build_payload(
                keywords, cat=0, timeframe="today 1-m", geo="US"
            )
            df = self.pytrends.interest_over_time()
            if df.empty:
                return {}
            return {kw: float(df[kw].mean()) for kw in keywords if kw in df.columns}

        try:
            return await loop.run_in_executor(None, _fetch)
        except Exception as e:
            logger.error(f"Google Trends baseline fetch failed: {e}")
            return {}

    async def _get_trending_searches(self) -> list[str]:
        """Fetch currently trending searches from Google."""
        await self.rate_limiter.acquire()
        loop = asyncio.get_event_loop()

        def _fetch():
            df = self.pytrends.trending_searches(pn="united_states")
            return df[0].tolist() if not df.empty else []

        try:
            return await loop.run_in_executor(None, _fetch)
        except Exception as e:
            logger.error(f"Trending searches fetch failed: {e}")
            return []

    async def collect_watchlist(self) -> list[MentionData]:
        """
        Mode 1: Monitor known watchlist keywords for interest spikes.

        Compares current 7-day interest to 30-day baseline.
        Flags anything with >2x normal interest as a potential signal.
        """
        mentions: list[MentionData] = []

        # pytrends allows max 5 keywords per request
        batches = [
            self.watchlist[i : i + 5] for i in range(0, len(self.watchlist), 5)
        ]

        for batch in batches:
            try:
                current = await retry_with_backoff(
                    self._get_interest_over_time,
                    batch,
                    max_attempts=config.rate_limit.retry_max_attempts,
                    base_delay=config.rate_limit.retry_base_delay,
                )
                baseline = await retry_with_backoff(
                    self._get_30day_baseline,
                    batch,
                    max_attempts=config.rate_limit.retry_max_attempts,
                    base_delay=config.rate_limit.retry_base_delay,
                )

                for keyword in batch:
                    cur_score = current.get(keyword, 0)
                    base_score = baseline.get(keyword, 0)

                    # Avoid division by zero; if baseline is 0 and current > 0, that's a spike
                    if base_score > 0:
                        ratio = cur_score / base_score
                    elif cur_score > 0:
                        ratio = float("inf")
                    else:
                        continue

                    if ratio >= self.spike_multiplier:
                        mentions.append(
                            MentionData(
                                source=SourceType.GOOGLE_TRENDS,
                                keyword=keyword,
                                mention_count=int(cur_score),
                                timestamp=datetime.now(timezone.utc),
                                metadata={
                                    "current_interest": cur_score,
                                    "baseline_interest": base_score,
                                    "spike_ratio": ratio,
                                    "mode": "watchlist",
                                },
                            )
                        )
                        logger.info(
                            f"Google Trends spike: '{keyword}' at {ratio:.1f}x baseline",
                            extra={"data": {"keyword": keyword, "ratio": ratio}},
                        )
            except Exception as e:
                logger.error(f"Watchlist batch failed: {e}")
                continue

        return mentions

    async def collect_trending(self) -> list[MentionData]:
        """
        Mode 2: Scan Google's trending searches for anything
        potentially mappable to a tradeable asset.

        These get passed to the ticker_mapper downstream.
        """
        mentions: list[MentionData] = []

        try:
            trending = await retry_with_backoff(
                self._get_trending_searches,
                max_attempts=config.rate_limit.retry_max_attempts,
                base_delay=config.rate_limit.retry_base_delay,
            )
            for term in trending:
                mentions.append(
                    MentionData(
                        source=SourceType.GOOGLE_TRENDS,
                        keyword=term,
                        mention_count=1,
                        timestamp=datetime.now(timezone.utc),
                        metadata={"mode": "discovery", "trending_search": True},
                    )
                )

            logger.info(
                f"Found {len(trending)} trending searches",
                extra={"data": {"count": len(trending)}},
            )
        except Exception as e:
            logger.error(f"Trending search collection failed: {e}")

        return mentions

    async def collect(self) -> list[MentionData]:
        """Run both watchlist and trending discovery modes."""
        watchlist_mentions, trending_mentions = await asyncio.gather(
            self.collect_watchlist(),
            self.collect_trending(),
            return_exceptions=True,
        )

        all_mentions: list[MentionData] = []
        for result in [watchlist_mentions, trending_mentions]:
            if isinstance(result, Exception):
                logger.error(f"Trends collection failed: {result}")
                continue
            all_mentions.extend(result)

        return all_mentions

    def add_to_watchlist(self, keywords: list[str]) -> None:
        """Dynamically expand the watchlist with new keywords."""
        for kw in keywords:
            if kw not in self.watchlist:
                self.watchlist.append(kw)
                logger.info(f"Added '{kw}' to Google Trends watchlist")
