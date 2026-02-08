"""
On-chain data collector for crypto assets.

Detects whale wallet movements, unusual volume spikes, and
large token transfers that often precede significant price moves.

Uses public blockchain APIs (Etherscan, etc.) and DeFi analytics.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

import httpx

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import MentionData, SourceType
from openclaw.social_arb.utils.logger import get_logger
from openclaw.social_arb.utils.rate_limiter import RateLimiter, retry_with_backoff

logger = get_logger(__name__)

# Public API endpoints for on-chain data
COINGECKO_BASE = "https://api.coingecko.com/api/v3"
WHALE_ALERT_API = "https://api.whale-alert.io/v1"

# Top crypto IDs on CoinGecko for monitoring
MONITORED_COINS: dict[str, str] = {
    "bitcoin": "BTC",
    "ethereum": "ETH",
    "solana": "SOL",
    "cardano": "ADA",
    "polkadot": "DOT",
    "avalanche-2": "AVAX",
    "matic-network": "MATIC",
    "chainlink": "LINK",
    "uniswap": "UNI",
    "dogecoin": "DOGE",
    "ripple": "XRP",
    "binancecoin": "BNB",
}


class OnchainCollector:
    """
    Monitors on-chain activity for trading-relevant signals.

    Detects:
    - Volume spikes (24h volume vs 7-day average)
    - Large price moves (potential whale accumulation/distribution)
    - Market cap shifts indicating capital rotation
    """

    def __init__(self):
        self.rate_limiter = RateLimiter(
            max_calls=30,        # CoinGecko free tier: ~30 calls/min
            period_seconds=60,
        )
        # Cache for 7-day volume averages
        self._volume_baselines: dict[str, float] = {}

    async def _fetch_market_data(self) -> list[dict[str, Any]]:
        """Fetch current market data for monitored coins from CoinGecko."""
        await self.rate_limiter.acquire()

        coin_ids = ",".join(MONITORED_COINS.keys())
        params = {
            "ids": coin_ids,
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "sparkline": "false",
            "price_change_percentage": "1h,24h,7d",
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{COINGECKO_BASE}/coins/markets", params=params
            )
            resp.raise_for_status()
            return resp.json()

    async def _fetch_coin_history(
        self, coin_id: str, days: int = 7
    ) -> dict[str, Any]:
        """Fetch historical volume data for baseline calculation."""
        await self.rate_limiter.acquire()

        params = {"vs_currency": "usd", "days": str(days)}

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(
                f"{COINGECKO_BASE}/coins/{coin_id}/market_chart", params=params
            )
            resp.raise_for_status()
            return resp.json()

    async def _compute_volume_baseline(self, coin_id: str) -> float:
        """Compute 7-day average daily volume for a coin."""
        if coin_id in self._volume_baselines:
            return self._volume_baselines[coin_id]

        try:
            history = await retry_with_backoff(
                self._fetch_coin_history,
                coin_id,
                max_attempts=2,
                base_delay=2.0,
            )
            volumes = history.get("total_volumes", [])
            if volumes:
                daily_volumes = [v[1] for v in volumes]
                avg_volume = sum(daily_volumes) / len(daily_volumes)
                self._volume_baselines[coin_id] = avg_volume
                return avg_volume
        except Exception as e:
            logger.error(f"Volume baseline fetch failed for {coin_id}: {e}")

        return 0.0

    async def detect_volume_spikes(self) -> list[MentionData]:
        """
        Detect coins with 24h volume significantly above their 7-day average.

        A volume spike often indicates whale accumulation or distribution
        before a major price move.
        """
        mentions: list[MentionData] = []

        try:
            market_data = await retry_with_backoff(
                self._fetch_market_data,
                max_attempts=config.rate_limit.retry_max_attempts,
                base_delay=config.rate_limit.retry_base_delay,
            )
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            return mentions

        for coin in market_data:
            coin_id = coin.get("id", "")
            ticker = MONITORED_COINS.get(coin_id, coin.get("symbol", "").upper())
            current_volume = coin.get("total_volume", 0)

            if not current_volume:
                continue

            # Get baseline (cached after first fetch)
            baseline = await self._compute_volume_baseline(coin_id)
            if baseline <= 0:
                continue

            volume_ratio = current_volume / baseline

            # Volume >2x baseline is noteworthy
            if volume_ratio >= 2.0:
                price_change_24h = coin.get("price_change_percentage_24h", 0) or 0
                price_change_1h = coin.get(
                    "price_change_percentage_1h_in_currency", 0
                ) or 0

                mentions.append(
                    MentionData(
                        source=SourceType.ONCHAIN,
                        ticker=ticker,
                        keyword=f"{ticker} volume spike",
                        mention_count=int(volume_ratio * 10),  # scale for anomaly detection
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "signal_type": "volume_spike",
                            "current_volume_usd": current_volume,
                            "baseline_volume_usd": baseline,
                            "volume_ratio": volume_ratio,
                            "price_change_24h_pct": price_change_24h,
                            "price_change_1h_pct": price_change_1h,
                            "market_cap": coin.get("market_cap", 0),
                            "current_price": coin.get("current_price", 0),
                        },
                    )
                )
                logger.info(
                    f"Volume spike: {ticker} at {volume_ratio:.1f}x baseline",
                    extra={"data": {"ticker": ticker, "ratio": volume_ratio}},
                )

        return mentions

    async def detect_price_anomalies(self) -> list[MentionData]:
        """
        Detect unusual price movements that might indicate
        insider activity or whale manipulation.
        """
        mentions: list[MentionData] = []

        try:
            market_data = await retry_with_backoff(
                self._fetch_market_data,
                max_attempts=config.rate_limit.retry_max_attempts,
                base_delay=config.rate_limit.retry_base_delay,
            )
        except Exception as e:
            logger.error(f"Market data fetch failed: {e}")
            return mentions

        for coin in market_data:
            coin_id = coin.get("id", "")
            ticker = MONITORED_COINS.get(coin_id, coin.get("symbol", "").upper())
            price_change_1h = coin.get(
                "price_change_percentage_1h_in_currency", 0
            ) or 0
            price_change_24h = coin.get("price_change_percentage_24h", 0) or 0

            # Flag >5% move in 1 hour or >15% in 24 hours as unusual
            if abs(price_change_1h) >= 5 or abs(price_change_24h) >= 15:
                mentions.append(
                    MentionData(
                        source=SourceType.ONCHAIN,
                        ticker=ticker,
                        keyword=f"{ticker} price anomaly",
                        mention_count=1,
                        timestamp=datetime.now(timezone.utc),
                        metadata={
                            "signal_type": "price_anomaly",
                            "price_change_1h_pct": price_change_1h,
                            "price_change_24h_pct": price_change_24h,
                            "current_price": coin.get("current_price", 0),
                            "market_cap": coin.get("market_cap", 0),
                            "already_moved": True,
                        },
                    )
                )

        return mentions

    async def collect(self) -> list[MentionData]:
        """Run all on-chain collection strategies."""
        volume_task = self.detect_volume_spikes()
        price_task = self.detect_price_anomalies()

        results = await asyncio.gather(volume_task, price_task, return_exceptions=True)

        all_mentions: list[MentionData] = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"On-chain collection failed: {result}")
                continue
            all_mentions.extend(result)

        logger.info(
            f"On-chain collection complete: {len(all_mentions)} signals",
            extra={"data": {"total_signals": len(all_mentions)}},
        )
        return all_mentions
