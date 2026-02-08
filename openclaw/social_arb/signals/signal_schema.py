"""
Pydantic models for all data flowing through the Social Arbitrage pipeline.

Structured schemas ensure every component speaks the same language
and signals are fully auditable for backtesting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class AssetClass(str, Enum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    ETF = "etf"


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"


class SentimentLabel(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


class SourceType(str, Enum):
    REDDIT = "reddit"
    TWITTER = "twitter"
    GOOGLE_TRENDS = "google_trends"
    NEWS = "news"
    ONCHAIN = "onchain"


# ---------------------------------------------------------------------------
# Collector-level data models
# ---------------------------------------------------------------------------

class MentionData(BaseModel):
    """Single mention/data point from any collector."""
    source: SourceType
    ticker: str | None = None
    keyword: str | None = None
    mention_count: int = 1
    sentiment_score: float | None = None        # -1.0 (bearish) to 1.0 (bullish)
    sentiment_label: SentimentLabel | None = None
    confidence: float | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    permalink: str | None = None
    author: str | None = None
    text_snippet: str | None = None             # first 500 chars for audit
    upvotes: int | None = None
    awards: int | None = None
    bot_probability: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AggregatedMentions(BaseModel):
    """Aggregated mention stats for a single ticker/keyword from one source."""
    source: SourceType
    ticker: str
    total_mentions: int
    unique_authors: int
    avg_sentiment: float
    sentiment_label: SentimentLabel
    max_upvotes: int = 0
    total_awards: int = 0
    period_start: datetime
    period_end: datetime
    mentions: list[MentionData] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Anomaly Detection Models
# ---------------------------------------------------------------------------

class AnomalyResult(BaseModel):
    """Result of anomaly detection for a single ticker/source pair."""
    ticker: str
    source: SourceType
    current_value: float                        # current mention count or interest score
    rolling_mean: float
    rolling_std: float
    z_score: float
    velocity_z_score: float | None = None       # rate-of-change z-score
    is_anomaly: bool
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Bot Filter Models
# ---------------------------------------------------------------------------

class BotFilterResult(BaseModel):
    """Result of bot/spam filtering on a set of mentions."""
    ticker: str
    total_mentions: int
    filtered_mentions: int                      # after removing bots
    bot_ratio: float                            # percentage flagged as bot
    flagged_authors: list[str] = Field(default_factory=list)
    clean_mentions: list[MentionData] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Cross-Validation Models
# ---------------------------------------------------------------------------

class CrossValidationResult(BaseModel):
    """Multi-source confirmation scoring."""
    ticker: str
    sources_detected: list[SourceType]
    source_count: int
    raw_confidence: float
    price_already_moved: bool = False
    price_move_pct: float | None = None
    final_confidence: float
    is_tradeable: bool                          # meets min confidence threshold
    evidence: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# The Main Trade Signal
# ---------------------------------------------------------------------------

class SocialArbSignal(BaseModel):
    """
    The final output of the Social Arbitrage pipeline.

    This is what gets sent to OpenClaw's execution layer.
    Every field is designed for auditability and backtesting.
    """
    signal_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    signal_type: Literal["social_arb"] = "social_arb"
    ticker: str
    asset_class: AssetClass
    direction: Direction
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str]                          # e.g. ["reddit_spike", "google_trends"]
    detected_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    edge_decay_estimate: str                    # e.g. "2-6 hours", "1-3 days"
    suggested_position_pct: float = Field(ge=0.0, le=100.0)
    stop_loss_pct: float
    take_profit_pct: float
    raw_data: dict[str, Any] = Field(default_factory=dict)
    bot_risk_score: float = Field(ge=0.0, le=1.0)
    hype_risk_score: float = Field(ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def is_high_confidence(self) -> bool:
        return self.confidence >= 0.7

    def is_safe_to_trade(self) -> bool:
        """Conservative safety check before execution."""
        return (
            self.confidence >= 0.55
            and self.bot_risk_score < 0.5
            and self.hype_risk_score < 0.7
        )


# ---------------------------------------------------------------------------
# Edge Estimation Models
# ---------------------------------------------------------------------------

class EdgeEstimate(BaseModel):
    """Estimate of how long the information edge will last."""
    ticker: str
    virality_velocity: float                    # mentions per hour
    primary_source: SourceType
    market_is_open: bool
    asset_liquidity: Literal["high", "medium", "low"]
    estimated_edge_hours_min: float
    estimated_edge_hours_max: float
    edge_decay_label: str                       # human-readable, e.g. "2-6 hours"
    reasoning: str
