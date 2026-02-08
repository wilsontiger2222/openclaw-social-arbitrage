# OpenClaw Social Arbitrage Sub-Agent

Autonomous social media scanner that detects emerging trends **before** they are priced into the market, generating structured trade signals for OpenClaw's execution layer.

## Architecture

```
Collect → Detect Anomalies → Filter Bots → Analyze Sentiment → Cross-Validate → Generate Signal
```

### Data Sources
- **Reddit** — PRAW-based scanner monitoring WSB, r/stocks, r/cryptocurrency, etc.
- **Google Trends** — Watchlist monitoring + trending search discovery
- **Twitter/X** — Cashtag and financial keyword tracking via v2 API
- **News** — NewsAPI + RSS feed aggregation from Reuters, CNBC, MarketWatch
- **On-chain** — CoinGecko-based volume spike and whale activity detection

### Analysis Pipeline
- **Anomaly Detection** — Z-score against EWMA rolling baselines + velocity detection
- **Sentiment Analysis** — FinBERT (local, zero API cost) with OpenAI fallback
- **Bot Filtering** — Account age, karma, follower ratio, coordinated posting detection
- **Cross-Validation** — Multi-source confirmation scoring (single source = 0.4, two = 0.7, three = 0.85)
- **Ticker Mapping** — Product/brand → ticker resolution with LLM fallback and persistent cache
- **Edge Estimation** — Predicts how long the information edge lasts based on virality, source type, liquidity

## Setup

```bash
# Clone and install
pip install -e .

# Copy and fill in API keys
cp .env.example .env

# Run the orchestrator
social-arb run

# Run with limited cycles (for testing)
social-arb run --cycles 5

# Check signal history
social-arb status
```

## Signal Output

Every signal includes:
- Ticker, direction (long/short), confidence (0-1)
- Source attribution (which sources confirmed)
- Edge decay estimate (how long the alpha lasts)
- Position sizing, stop loss, take profit suggestions
- Bot risk and hype/pump-and-dump risk scores

## Configuration

All settings are in `openclaw/social_arb/config.py`. Key thresholds:
- `z_score_threshold: 2.5` — standard deviations for anomaly detection
- `min_cross_sources: 2` — minimum independent sources for a tradeable signal
- `bot_probability_cutoff: 0.7` — discard mentions above this bot score
- `min_tradeable_confidence: 0.55` — minimum confidence to generate a signal

## License

MIT
