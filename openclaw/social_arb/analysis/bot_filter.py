"""
Bot and spam detection filter.

Identifies and scores automated, coordinated, or spam accounts
to prevent manipulation-driven signals from reaching the trade engine.

Detection strategies:
- Account age/karma checks (Reddit)
- Follower ratio analysis (Twitter)
- Coordinated posting detection (same content, short timeframe)
- Posting frequency anomalies
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher
from typing import Any

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import (
    BotFilterResult,
    MentionData,
    SourceType,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class BotFilter:
    """
    Filters out bot/spam/coordinated activity from social data.

    Each mention gets a bot_probability score (0-1).
    Mentions scoring above the cutoff threshold are discarded.
    """

    def __init__(self):
        self.min_account_age_days = config.bot_filter.min_account_age_days
        self.min_karma = config.bot_filter.min_karma
        self.bot_cutoff = config.bot_filter.bot_probability_cutoff
        self.coordinated_window = config.bot_filter.coordinated_window_seconds
        self.coordinated_min_accounts = config.bot_filter.coordinated_min_accounts

    def _score_reddit_account(self, mention: MentionData) -> float:
        """
        Score bot probability for a Reddit mention based on account signals.

        Factors:
        - Account age < 30 days: suspicious
        - Low karma: suspicious
        - [deleted] author: can't verify, moderate risk
        """
        score = 0.0
        metadata = mention.metadata

        author = mention.author or "[deleted]"
        if author == "[deleted]":
            score += 0.3  # can't verify, moderate risk

        # Account age check (if available in metadata)
        account_age_days = metadata.get("account_age_days")
        if account_age_days is not None:
            if account_age_days < 7:
                score += 0.5  # very new account
            elif account_age_days < self.min_account_age_days:
                score += 0.3  # new account

        # Karma check
        karma = metadata.get("author_karma", None)
        if karma is not None and karma < self.min_karma:
            score += 0.2

        # Upvote ratio — genuine posts tend to have healthy ratios
        upvote_ratio = metadata.get("upvote_ratio", 0.5)
        if upvote_ratio < 0.3:
            score += 0.1  # heavily downvoted

        return min(1.0, score)

    def _score_twitter_account(self, mention: MentionData) -> float:
        """
        Score bot probability for a Twitter mention.

        Factors:
        - Suspicious follower/following ratio
        - Default avatar
        - Account creation date
        - Posting frequency (inhuman intervals)
        """
        score = 0.0
        metadata = mention.metadata

        # Follower ratio: bots often follow many but have few followers
        follower_ratio = metadata.get("author_follower_ratio", 1.0)
        if follower_ratio < 0.1:
            score += 0.3  # follows many, few follow back
        elif follower_ratio < 0.3:
            score += 0.15

        # Account verification — verified accounts are less likely bots
        if metadata.get("author_verified", False):
            score -= 0.2  # reduce bot score for verified

        # Very low followers for financial commentary is suspicious
        followers = metadata.get("author_followers", 0)
        if followers < 10:
            score += 0.3
        elif followers < 50:
            score += 0.15

        # Account age
        created_at = metadata.get("author_created_at")
        if created_at:
            try:
                created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - created).days
                if age_days < 30:
                    score += 0.3
                elif age_days < 90:
                    score += 0.1
            except (ValueError, TypeError):
                pass

        return max(0.0, min(1.0, score))

    def _detect_coordinated_posting(
        self, mentions: list[MentionData]
    ) -> set[str]:
        """
        Detect coordinated behavior: multiple accounts posting
        the same or very similar content within a short timeframe.

        Returns set of author names flagged as coordinated.
        """
        flagged_authors: set[str] = set()

        # Group by time window
        time_groups: dict[int, list[MentionData]] = defaultdict(list)
        for m in mentions:
            # Bucket by time window
            bucket = int(m.timestamp.timestamp()) // self.coordinated_window
            time_groups[bucket].append(m)

        for bucket, group in time_groups.items():
            if len(group) < self.coordinated_min_accounts:
                continue

            # Check for similar content within the window
            texts = [(m.author or "anon", m.text_snippet or "") for m in group]
            for i in range(len(texts)):
                similar_count = 0
                similar_authors = {texts[i][0]}
                for j in range(i + 1, len(texts)):
                    if texts[i][0] == texts[j][0]:
                        continue  # same author
                    # Use SequenceMatcher for fuzzy string comparison
                    similarity = SequenceMatcher(
                        None, texts[i][1][:200], texts[j][1][:200]
                    ).ratio()
                    if similarity > 0.7:  # 70% similar text
                        similar_count += 1
                        similar_authors.add(texts[j][0])

                if similar_count >= self.coordinated_min_accounts - 1:
                    flagged_authors.update(similar_authors)
                    logger.warning(
                        f"Coordinated posting detected: {similar_authors}",
                        extra={"data": {"authors": list(similar_authors)}},
                    )

        return flagged_authors

    def score_mention(self, mention: MentionData) -> float:
        """
        Compute bot probability for a single mention.

        Routes to source-specific scoring based on the mention's source.
        """
        if mention.source == SourceType.REDDIT:
            return self._score_reddit_account(mention)
        elif mention.source == SourceType.TWITTER:
            return self._score_twitter_account(mention)
        else:
            # News and on-chain sources are inherently less bot-prone
            return 0.0

    def filter_mentions(
        self, mentions: list[MentionData], ticker: str | None = None
    ) -> BotFilterResult:
        """
        Filter a list of mentions, removing those with high bot probability.

        Returns a BotFilterResult with clean mentions and statistics.
        """
        # Score each mention
        for mention in mentions:
            mention.bot_probability = self.score_mention(mention)

        # Detect coordinated posting across all mentions
        coordinated_authors = self._detect_coordinated_posting(mentions)

        # Boost bot score for coordinated accounts
        for mention in mentions:
            if mention.author in coordinated_authors:
                mention.bot_probability = min(
                    1.0, mention.bot_probability + 0.4
                )

        # Separate clean from bot mentions
        clean = [m for m in mentions if m.bot_probability < self.bot_cutoff]
        flagged = [
            m.author or "anon"
            for m in mentions
            if m.bot_probability >= self.bot_cutoff
        ]

        result = BotFilterResult(
            ticker=ticker or "unknown",
            total_mentions=len(mentions),
            filtered_mentions=len(clean),
            bot_ratio=(len(mentions) - len(clean)) / max(len(mentions), 1),
            flagged_authors=list(set(flagged)),
            clean_mentions=clean,
        )

        if flagged:
            logger.info(
                f"Bot filter: {len(flagged)} of {len(mentions)} mentions flagged "
                f"for ticker {ticker}",
                extra={
                    "data": {
                        "ticker": ticker,
                        "total": len(mentions),
                        "filtered": len(clean),
                        "bot_ratio": result.bot_ratio,
                    }
                },
            )

        return result
