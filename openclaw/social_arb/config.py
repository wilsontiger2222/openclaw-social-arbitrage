"""
Configuration for the Social Arbitrage Sub-Agent.

All secrets are loaded from environment variables.
Thresholds and intervals are tuned for detecting social signals
before they are priced into the market.
"""

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ---------------------------------------------------------------------------
# API Keys — all loaded from env vars, never hard-coded
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class APIKeys:
    reddit_client_id: str = os.getenv("REDDIT_CLIENT_ID", "")
    reddit_secret: str = os.getenv("REDDIT_SECRET", "")
    reddit_user_agent: str = os.getenv("REDDIT_USER_AGENT", "OpenClaw Social Arb v1.0")
    twitter_bearer_token: str = os.getenv("TWITTER_BEARER_TOKEN", "")
    newsapi_key: str = os.getenv("NEWSAPI_KEY", "")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")


# ---------------------------------------------------------------------------
# Scanning intervals (seconds)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ScanIntervals:
    reddit: int = 300          # 5 minutes
    google_trends: int = 900   # 15 minutes
    twitter: int = 300         # 5 minutes
    news: int = 600            # 10 minutes
    onchain: int = 120         # 2 minutes — whale moves are time-critical


# ---------------------------------------------------------------------------
# Anomaly Detection Thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AnomalyThresholds:
    z_score_threshold: float = 2.5          # standard deviations above mean
    velocity_z_threshold: float = 3.0       # rate-of-change anomaly threshold
    min_cross_sources: int = 2              # minimum independent sources for a valid signal
    min_mentions_absolute: int = 10         # ignore tickers with < 10 total mentions
    rolling_window_days: int = 7            # baseline window for Reddit/Twitter
    trends_baseline_days: int = 30          # baseline window for Google Trends
    ewma_span: int = 48                     # exponentially weighted moving average span (hours)
    price_already_moved_pct: float = 10.0   # if price already moved this much, penalise confidence
    confidence_penalty_late: float = 0.3    # confidence reduction for late signals


# ---------------------------------------------------------------------------
# Bot Filtering
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BotFilterConfig:
    min_account_age_days: int = 30
    min_karma: int = 100
    bot_probability_cutoff: float = 0.7     # discard data points above this
    coordinated_window_seconds: int = 300   # detect same content posted within 5 min
    coordinated_min_accounts: int = 3       # flag if >= 3 accounts post similar content


# ---------------------------------------------------------------------------
# Sentiment Analysis
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SentimentConfig:
    finbert_model: str = "ProsusAI/finbert"
    batch_size: int = 32
    confidence_threshold: float = 0.6       # below this, classify as NEUTRAL
    hype_volume_threshold: int = 50         # high volume threshold for hype detection
    hype_sentiment_threshold: float = 0.8   # overly positive sentiment threshold
    hype_substance_threshold: float = 0.3   # low substance (short posts, no DD)


# ---------------------------------------------------------------------------
# Subreddits & Seed Keywords
# ---------------------------------------------------------------------------

MONITORED_SUBREDDITS: list[str] = [
    "wallstreetbets",
    "stocks",
    "cryptocurrency",
    "technology",
    "investing",
    "options",
    "pennystocks",
    "CryptoMoonShots",
]

GOOGLE_TRENDS_SEED_KEYWORDS: list[str] = [
    "stock market",
    "crypto",
    "bitcoin",
    "earnings report",
    "IPO",
    "SEC filing",
    "short squeeze",
    "meme stock",
]

# Common English words that look like tickers but aren't
TICKER_EXCLUSION_LIST: set[str] = {
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL",
    "CAN", "HAD", "HER", "WAS", "ONE", "OUR", "OUT", "HAS",
    "HIS", "HOW", "ITS", "MAY", "NEW", "NOW", "OLD", "SEE",
    "WAY", "WHO", "BOY", "DID", "GET", "HIM", "LET", "SAY",
    "SHE", "TOO", "USE", "DAD", "MOM", "CEO", "CFO", "COO",
    "ETF", "IPO", "SEC", "FDA", "FBI", "CIA", "GDP", "USA",
    "IMO", "LOL", "FYI", "TBH", "OMG", "WTF", "DD", "PT",
    "EPS", "PE", "RSI", "ATH", "ATL", "HODL", "YOLO", "FOMO",
    "TLDR", "EDIT", "LINK", "LONG", "HOLD", "SELL", "JUST",
    "LIKE", "THIS", "THAT", "WHAT", "WHEN", "WILL", "WITH",
    "FROM", "THEY", "BEEN", "HAVE", "MANY", "SOME", "THEM",
    "THAN", "EACH", "MAKE", "VERY", "AFTER", "ALSO", "MADE",
    "PUMP", "DUMP", "MOON", "BEAR", "BULL", "CALL", "PUTS",
    "HIGH", "LOWS", "GAIN", "LOSS", "FREE", "BEST", "HUGE",
    "MOST", "MUST", "NEXT", "ONLY", "OVER", "SAME", "GOOD",
    "WANT", "TAKE", "COME", "MORE", "BACK", "KNOW", "WELL",
    "EVEN", "THEN", "YEAR", "MUCH", "LAST", "HERE", "KEEP",
    "STOP", "STILL", "INTO", "DOWN", "WORK", "DOES", "DONE",
    "HELP", "NEED", "SURE", "LOOK", "THINK", "REAL", "WENT",
    "SAVE", "RISK", "SAFE", "OPEN", "CARE", "MOVE", "LIFE",
    "FUND", "CASH", "DEBT", "RATE", "PLAN", "BANK", "LOAN",
    "BOND", "GOLD", "TECH", "POST", "READ", "NEWS",
}

# Known crypto tickers (these bypass normal word-exclusion logic)
KNOWN_CRYPTO_TICKERS: set[str] = {
    "BTC", "ETH", "SOL", "ADA", "DOT", "AVAX", "MATIC",
    "LINK", "UNI", "AAVE", "DOGE", "SHIB", "XRP", "BNB",
    "LTC", "ATOM", "NEAR", "APT", "ARB", "OP", "FTM",
    "CRO", "ALGO", "MANA", "SAND", "AXS", "GALA", "IMX",
    "FIL", "ICP", "HBAR", "VET", "EOS", "XLM", "TRX",
    "PEPE", "BONK", "WIF", "FLOKI", "RENDER", "INJ",
}


# ---------------------------------------------------------------------------
# Cross-Validation Scoring Matrix
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CrossValidationConfig:
    """Weights for multi-source signal confirmation."""
    single_source_confidence: float = 0.4       # log but don't trade
    two_source_confidence: float = 0.7          # reddit + trends
    three_source_confidence: float = 0.85       # reddit + trends + news
    onchain_bonus: float = 0.1                  # bonus if on-chain confirms (crypto)
    late_penalty: float = 0.3                   # penalty if price already moved >10%
    min_tradeable_confidence: float = 0.55      # below this, signal is info-only


# ---------------------------------------------------------------------------
# Signal Defaults
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SignalDefaults:
    default_stop_loss_pct: float = 5.0
    default_take_profit_pct: float = 15.0
    max_position_pct: float = 2.0               # max 2% of portfolio per signal
    min_position_pct: float = 0.25


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StorageConfig:
    baseline_db_path: str = "data/baselines.db"
    signal_log_path: str = "data/signals.jsonl"
    ticker_map_path: str = "data/ticker_mappings.json"


# ---------------------------------------------------------------------------
# Rate Limiting
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RateLimitConfig:
    reddit_requests_per_minute: int = 30
    twitter_requests_per_15min: int = 300
    newsapi_requests_per_day: int = 100
    openai_requests_per_minute: int = 20
    google_trends_requests_per_hour: int = 30
    retry_max_attempts: int = 3
    retry_base_delay: float = 1.0               # seconds, doubles each retry


# ---------------------------------------------------------------------------
# Master Config — single import point
# ---------------------------------------------------------------------------

@dataclass
class Config:
    api_keys: APIKeys = field(default_factory=APIKeys)
    scan_intervals: ScanIntervals = field(default_factory=ScanIntervals)
    anomaly: AnomalyThresholds = field(default_factory=AnomalyThresholds)
    bot_filter: BotFilterConfig = field(default_factory=BotFilterConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    signal_defaults: SignalDefaults = field(default_factory=SignalDefaults)
    storage: StorageConfig = field(default_factory=StorageConfig)
    rate_limit: RateLimitConfig = field(default_factory=RateLimitConfig)
    subreddits: list[str] = field(default_factory=lambda: MONITORED_SUBREDDITS.copy())
    trends_keywords: list[str] = field(default_factory=lambda: GOOGLE_TRENDS_SEED_KEYWORDS.copy())


# Global config instance
config = Config()
