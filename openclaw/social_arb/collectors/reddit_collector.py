"""
Reddit collector using PRAW.

Monitors configured subreddits for ticker mention spikes,
sentiment shifts, and viral post detection (upvote/award velocity).
"""

from __future__ import annotations

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import Any

import praw
from praw.models import Submission

from openclaw.social_arb.config import config, TICKER_EXCLUSION_LIST, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.signals.signal_schema import MentionData, SourceType
from openclaw.social_arb.utils.logger import get_logger
from openclaw.social_arb.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = get_logger(__name__)

# Regex patterns for ticker extraction
# $AAPL style explicit tickers
CASHTAG_PATTERN = re.compile(r"\$([A-Z]{1,5})\b")
# ALL-CAPS words that might be tickers (needs exclusion filtering)
ALLCAPS_PATTERN = re.compile(r"\b([A-Z]{2,5})\b")


class RedditCollector:
    """
    Scans Reddit for ticker mentions, sentiment, and viral signals.

    Uses PRAW in a sync-to-async wrapper since PRAW is not natively async.
    Collects from hot, new, and rising posts in monitored subreddits.
    """

    def __init__(self):
        self.reddit = praw.Reddit(
            client_id=config.api_keys.reddit_client_id,
            client_secret=config.api_keys.reddit_secret,
            user_agent=config.api_keys.reddit_user_agent,
        )
        self.rate_limiter = RateLimiter(
            max_calls=config.rate_limit.reddit_requests_per_minute,
            period_seconds=60,
        )
        self.subreddits = config.subreddits

    def extract_tickers(self, text: str) -> list[str]:
        """
        Extract potential ticker symbols from text.

        Priority:
        1. Cashtag mentions ($AAPL) — highest confidence
        2. All-caps words filtered against exclusion list
        3. Known crypto tickers bypass exclusion
        """
        tickers: set[str] = set()

        # Cashtags are always tickers
        for match in CASHTAG_PATTERN.findall(text):
            tickers.add(match)

        # All-caps words, filtered
        for match in ALLCAPS_PATTERN.findall(text):
            if match in KNOWN_CRYPTO_TICKERS:
                tickers.add(match)
            elif match not in TICKER_EXCLUSION_LIST and len(match) >= 2:
                tickers.add(match)

        return list(tickers)

    def _submission_to_mentions(self, submission: Submission) -> list[MentionData]:
        """Convert a Reddit submission into MentionData objects, one per ticker found."""
        text = f"{submission.title} {submission.selftext}"
        tickers = self.extract_tickers(text)

        if not tickers:
            return []

        mentions = []
        for ticker in tickers:
            mentions.append(
                MentionData(
                    source=SourceType.REDDIT,
                    ticker=ticker,
                    keyword=submission.title[:100],
                    mention_count=1,
                    timestamp=datetime.fromtimestamp(
                        submission.created_utc, tz=timezone.utc
                    ),
                    permalink=f"https://reddit.com{submission.permalink}",
                    author=str(submission.author) if submission.author else "[deleted]",
                    text_snippet=text[:500],
                    upvotes=submission.score,
                    awards=getattr(submission, "total_awards_received", 0),
                    metadata={
                        "subreddit": str(submission.subreddit),
                        "num_comments": submission.num_comments,
                        "upvote_ratio": submission.upvote_ratio,
                        "is_self": submission.is_self,
                        "flair": submission.link_flair_text,
                    },
                )
            )
        return mentions

    async def _fetch_subreddit(
        self, subreddit_name: str, limit: int = 50
    ) -> list[MentionData]:
        """Fetch recent posts from a single subreddit."""
        mentions: list[MentionData] = []
        try:
            await self.rate_limiter.acquire()
            subreddit = self.reddit.subreddit(subreddit_name)

            # Run PRAW's sync calls in a thread pool
            loop = asyncio.get_event_loop()

            # Fetch hot, new, and rising to catch signals at different stages
            for sort_method in ["hot", "new", "rising"]:
                listing = getattr(subreddit, sort_method)
                submissions = await loop.run_in_executor(
                    None, lambda lm=listing: list(lm(limit=limit))
                )
                for submission in submissions:
                    mentions.extend(self._submission_to_mentions(submission))

            logger.info(
                f"Collected {len(mentions)} mentions from r/{subreddit_name}",
                extra={"data": {"subreddit": subreddit_name, "count": len(mentions)}},
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch r/{subreddit_name}: {e}",
                extra={"data": {"subreddit": subreddit_name, "error": str(e)}},
            )
        return mentions

    async def collect(self) -> list[MentionData]:
        """
        Collect mentions from all monitored subreddits concurrently.

        Returns a flat list of all MentionData found across all subreddits.
        """
        tasks = [
            retry_with_backoff(
                self._fetch_subreddit,
                sub,
                max_attempts=config.rate_limit.retry_max_attempts,
                base_delay=config.rate_limit.retry_base_delay,
            )
            for sub in self.subreddits
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_mentions: list[MentionData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Subreddit collection failed: {result}")
                continue
            all_mentions.extend(result)

        logger.info(
            f"Reddit collection complete: {len(all_mentions)} total mentions",
            extra={"data": {"total_mentions": len(all_mentions)}},
        )
        return all_mentions

    async def get_comment_sentiment_sample(
        self, permalink: str, sample_size: int = 20
    ) -> list[str]:
        """
        Fetch top comments from a specific post for sentiment analysis.

        Used when a post is flagged as high-volume to gauge comment sentiment.
        """
        try:
            await self.rate_limiter.acquire()
            loop = asyncio.get_event_loop()
            submission = await loop.run_in_executor(
                None, lambda: self.reddit.submission(url=permalink)
            )
            await loop.run_in_executor(
                None, lambda: submission.comments.replace_more(limit=0)
            )
            comments = [
                c.body for c in submission.comments[:sample_size] if hasattr(c, "body")
            ]
            return comments
        except Exception as e:
            logger.error(f"Failed to fetch comments from {permalink}: {e}")
            return []
