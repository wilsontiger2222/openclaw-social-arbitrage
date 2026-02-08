"""
Maps trending topics, products, and brand names to tradeable ticker symbols.

Maintains a local cache of known mappings and uses LLM for unknown ones.
Handles edge cases like private companies, subsidiaries, and conglomerates.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx

from openclaw.social_arb.config import config, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.signals.signal_schema import AssetClass
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)

# Pre-loaded well-known mappings that don't need LLM calls
KNOWN_MAPPINGS: dict[str, dict[str, str]] = {
    # Tech giants
    "apple": {"ticker": "AAPL", "asset_class": "equity"},
    "iphone": {"ticker": "AAPL", "asset_class": "equity"},
    "ipad": {"ticker": "AAPL", "asset_class": "equity"},
    "macbook": {"ticker": "AAPL", "asset_class": "equity"},
    "microsoft": {"ticker": "MSFT", "asset_class": "equity"},
    "windows": {"ticker": "MSFT", "asset_class": "equity"},
    "xbox": {"ticker": "MSFT", "asset_class": "equity"},
    "google": {"ticker": "GOOGL", "asset_class": "equity"},
    "alphabet": {"ticker": "GOOGL", "asset_class": "equity"},
    "youtube": {"ticker": "GOOGL", "asset_class": "equity"},
    "amazon": {"ticker": "AMZN", "asset_class": "equity"},
    "aws": {"ticker": "AMZN", "asset_class": "equity"},
    "meta": {"ticker": "META", "asset_class": "equity"},
    "facebook": {"ticker": "META", "asset_class": "equity"},
    "instagram": {"ticker": "META", "asset_class": "equity"},
    "whatsapp": {"ticker": "META", "asset_class": "equity"},
    "tesla": {"ticker": "TSLA", "asset_class": "equity"},
    "nvidia": {"ticker": "NVDA", "asset_class": "equity"},
    "netflix": {"ticker": "NFLX", "asset_class": "equity"},
    "disney": {"ticker": "DIS", "asset_class": "equity"},
    "spotify": {"ticker": "SPOT", "asset_class": "equity"},
    "uber": {"ticker": "UBER", "asset_class": "equity"},
    "airbnb": {"ticker": "ABNB", "asset_class": "equity"},
    "coinbase": {"ticker": "COIN", "asset_class": "equity"},
    "palantir": {"ticker": "PLTR", "asset_class": "equity"},
    "amd": {"ticker": "AMD", "asset_class": "equity"},
    "intel": {"ticker": "INTC", "asset_class": "equity"},
    "salesforce": {"ticker": "CRM", "asset_class": "equity"},
    "shopify": {"ticker": "SHOP", "asset_class": "equity"},
    "paypal": {"ticker": "PYPL", "asset_class": "equity"},
    "snap": {"ticker": "SNAP", "asset_class": "equity"},
    "snapchat": {"ticker": "SNAP", "asset_class": "equity"},
    "twitter": {"ticker": "TWTR", "asset_class": "equity"},
    "gamestop": {"ticker": "GME", "asset_class": "equity"},
    "amc": {"ticker": "AMC", "asset_class": "equity"},
    # Pharma / biotech
    "pfizer": {"ticker": "PFE", "asset_class": "equity"},
    "moderna": {"ticker": "MRNA", "asset_class": "equity"},
    "johnson & johnson": {"ticker": "JNJ", "asset_class": "equity"},
    # Crypto
    "bitcoin": {"ticker": "BTC", "asset_class": "crypto"},
    "ethereum": {"ticker": "ETH", "asset_class": "crypto"},
    "solana": {"ticker": "SOL", "asset_class": "crypto"},
    "dogecoin": {"ticker": "DOGE", "asset_class": "crypto"},
    "cardano": {"ticker": "ADA", "asset_class": "crypto"},
    "ripple": {"ticker": "XRP", "asset_class": "crypto"},
    "polkadot": {"ticker": "DOT", "asset_class": "crypto"},
    # ETFs
    "s&p 500": {"ticker": "SPY", "asset_class": "etf"},
    "sp500": {"ticker": "SPY", "asset_class": "etf"},
    "nasdaq": {"ticker": "QQQ", "asset_class": "etf"},
    "ark innovation": {"ticker": "ARKK", "asset_class": "etf"},
    "vix": {"ticker": "VIX", "asset_class": "etf"},
}


class TickerMapper:
    """
    Maps trending topics and product names to tradeable ticker symbols.

    Uses a three-tier strategy:
    1. Check pre-loaded known mappings (instant, no cost)
    2. Check persistent JSON cache (fast, no cost)
    3. Fall back to LLM for unknown mappings (slow, costs money)

    All discovered mappings are cached for future use.
    """

    def __init__(self):
        self.cache_path = Path(config.storage.ticker_map_path)
        self._cache: dict[str, dict[str, str]] = {}
        self._load_cache()

    def _load_cache(self) -> None:
        """Load the persistent ticker mapping cache from disk."""
        # Start with known mappings
        self._cache = dict(KNOWN_MAPPINGS)

        # Overlay any cached mappings from disk
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r") as f:
                    disk_cache = json.load(f)
                self._cache.update(disk_cache)
                logger.info(f"Loaded {len(disk_cache)} cached ticker mappings")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Failed to load ticker cache: {e}")

    def _save_cache(self) -> None:
        """Persist the ticker mapping cache to disk."""
        # Only save entries that aren't in KNOWN_MAPPINGS (avoid duplication)
        to_save = {
            k: v for k, v in self._cache.items() if k not in KNOWN_MAPPINGS
        }
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.cache_path, "w") as f:
                json.dump(to_save, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save ticker cache: {e}")

    def lookup(self, topic: str) -> dict[str, str] | None:
        """
        Look up a topic in the local cache.

        Returns {"ticker": "AAPL", "asset_class": "equity"} or None.
        """
        normalized = topic.lower().strip()
        result = self._cache.get(normalized)

        # Also check if the topic IS a known ticker
        upper = topic.upper().strip()
        if upper in KNOWN_CRYPTO_TICKERS:
            return {"ticker": upper, "asset_class": "crypto"}

        return result

    async def resolve(self, topic: str) -> dict[str, str] | None:
        """
        Resolve a topic to a ticker, using LLM if needed.

        Returns {"ticker": "AAPL", "asset_class": "equity"}
        or None if the topic is not mappable to a public company.
        """
        # Check cache first
        cached = self.lookup(topic)
        if cached is not None:
            return cached

        # Use LLM to resolve unknown topics
        if not config.api_keys.openai_api_key:
            logger.warning(f"No OpenAI key — cannot resolve topic: {topic}")
            return None

        try:
            result = await self._llm_resolve(topic)
            if result:
                # Cache the result
                self._cache[topic.lower().strip()] = result
                self._save_cache()
                logger.info(
                    f"Mapped '{topic}' → {result['ticker']}",
                    extra={"data": {"topic": topic, "mapping": result}},
                )
            else:
                # Cache negative results too, to avoid repeated LLM calls
                self._cache[topic.lower().strip()] = {"ticker": "NONE", "asset_class": "none"}
                self._save_cache()
            return result
        except Exception as e:
            logger.error(f"LLM ticker mapping failed for '{topic}': {e}")
            return None

    async def _llm_resolve(self, topic: str) -> dict[str, str] | None:
        """
        Use OpenAI to map a trending topic to a tradeable ticker.

        Handles edge cases:
        - Private companies → returns None
        - Subsidiaries → returns parent company ticker
        - Multiple possible tickers → returns the most direct one
        """
        prompt = (
            "What publicly traded company (if any) is most directly associated with "
            f"the following trending topic?\n\n"
            f"Topic: {topic}\n\n"
            "Respond in exactly this format:\n"
            "TICKER: <symbol or PRIVATE or NONE>\n"
            "ASSET_CLASS: <equity or crypto or etf>\n"
            "REASONING: <one sentence explanation>\n\n"
            "Rules:\n"
            "- If the company is private, respond TICKER: PRIVATE\n"
            "- If no company is associated, respond TICKER: NONE\n"
            "- If it's a subsidiary, return the publicly traded parent\n"
            "- Prefer the most directly related company\n"
            "- Use the US-listed ticker symbol"
        )

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {config.api_keys.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 100,
                    "temperature": 0.0,
                },
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]

        # Parse response
        ticker = None
        asset_class = "equity"

        for line in content.strip().split("\n"):
            line = line.strip()
            if line.startswith("TICKER:"):
                raw = line.split(":", 1)[1].strip().upper()
                if raw not in ("PRIVATE", "NONE", "N/A", ""):
                    ticker = raw
            elif line.startswith("ASSET_CLASS:"):
                raw = line.split(":", 1)[1].strip().lower()
                if raw in ("equity", "crypto", "etf"):
                    asset_class = raw

        if ticker:
            return {"ticker": ticker, "asset_class": asset_class}
        return None

    async def resolve_batch(
        self, topics: list[str]
    ) -> dict[str, dict[str, str] | None]:
        """
        Resolve multiple topics, returning mappings for each.

        Uses cache where possible, batches LLM calls for unknowns.
        """
        results: dict[str, dict[str, str] | None] = {}
        to_resolve: list[str] = []

        for topic in topics:
            cached = self.lookup(topic)
            if cached is not None:
                # Filter out negative cache entries
                if cached.get("ticker") not in ("NONE", "PRIVATE"):
                    results[topic] = cached
                else:
                    results[topic] = None
            else:
                to_resolve.append(topic)

        # Resolve unknowns concurrently (with rate limiting via the LLM call)
        if to_resolve:
            tasks = [self.resolve(topic) for topic in to_resolve]
            resolved = await asyncio.gather(*tasks, return_exceptions=True)

            for topic, result in zip(to_resolve, resolved):
                if isinstance(result, Exception):
                    logger.error(f"Failed to resolve '{topic}': {result}")
                    results[topic] = None
                else:
                    results[topic] = result

        return results
