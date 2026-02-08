"""
Main orchestrator for the Social Arbitrage Sub-Agent.

Coordinates the full pipeline:
Collect → Detect Anomalies → Filter Bots → Analyze Sentiment →
Cross-Validate → Map Tickers → Estimate Edge → Generate Signal

Runs as an async loop with configurable intervals per source.
Implements circuit breakers for resilience when APIs go down.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Awaitable

from openclaw.social_arb.config import config, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.collectors.reddit_collector import RedditCollector
from openclaw.social_arb.collectors.google_trends import GoogleTrendsCollector
from openclaw.social_arb.collectors.twitter_collector import TwitterCollector
from openclaw.social_arb.collectors.news_collector import NewsCollector
from openclaw.social_arb.collectors.onchain_collector import OnchainCollector
from openclaw.social_arb.analysis.anomaly_detector import AnomalyDetector
from openclaw.social_arb.analysis.sentiment_analyzer import SentimentAnalyzer
from openclaw.social_arb.analysis.bot_filter import BotFilter
from openclaw.social_arb.analysis.cross_validator import CrossValidator
from openclaw.social_arb.analysis.ticker_mapper import TickerMapper
from openclaw.social_arb.signals.signal_generator import SignalGenerator
from openclaw.social_arb.signals.edge_estimator import EdgeEstimator
from openclaw.social_arb.signals.signal_schema import (
    AnomalyResult,
    MentionData,
    SentimentLabel,
    SocialArbSignal,
    SourceType,
)
from openclaw.social_arb.storage.signal_log import SignalLog
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class CircuitBreaker:
    """
    Simple circuit breaker to handle API failures gracefully.

    After `failure_threshold` consecutive failures, the circuit opens
    and the collector is skipped for `recovery_timeout` seconds.
    This prevents hammering a dead API and wasting rate limit budget.
    """

    def __init__(self, failure_threshold: int = 3, recovery_timeout: float = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure: datetime | None = None
        self.is_open = False

    def record_success(self) -> None:
        self.failure_count = 0
        self.is_open = False

    def record_failure(self) -> None:
        self.failure_count += 1
        self.last_failure = datetime.now(timezone.utc)
        if self.failure_count >= self.failure_threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker opened after {self.failure_count} failures"
            )

    def can_execute(self) -> bool:
        if not self.is_open:
            return True
        # Check if recovery timeout has elapsed
        if self.last_failure:
            elapsed = (datetime.now(timezone.utc) - self.last_failure).total_seconds()
            if elapsed >= self.recovery_timeout:
                logger.info("Circuit breaker half-open, allowing retry")
                self.is_open = False
                self.failure_count = 0
                return True
        return False


class Orchestrator:
    """
    Main coordination loop for the Social Arbitrage pipeline.

    Manages all collectors, analyzers, and signal generators.
    Runs continuously with configurable scan intervals per source.
    Outputs valid signals via a callback function.
    """

    def __init__(
        self,
        signal_callback: Callable[[SocialArbSignal], Awaitable[None]] | None = None,
    ):
        # Collectors
        self.reddit = RedditCollector()
        self.google_trends = GoogleTrendsCollector()
        self.twitter = TwitterCollector()
        self.news = NewsCollector()
        self.onchain = OnchainCollector()

        # Analyzers
        self.anomaly_detector = AnomalyDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.bot_filter = BotFilter()
        self.cross_validator = CrossValidator()
        self.ticker_mapper = TickerMapper()

        # Signal generation
        self.signal_generator = SignalGenerator()
        self.edge_estimator = EdgeEstimator()
        self.signal_log = SignalLog()

        # Callback for emitting signals to OpenClaw's execution layer
        self.signal_callback = signal_callback

        # Circuit breakers per collector
        self.circuit_breakers: dict[str, CircuitBreaker] = {
            "reddit": CircuitBreaker(),
            "google_trends": CircuitBreaker(),
            "twitter": CircuitBreaker(),
            "news": CircuitBreaker(),
            "onchain": CircuitBreaker(),
        }

        # Track last run time per collector for interval management
        self._last_run: dict[str, datetime] = {}
        self._running = False

    async def _safe_collect(
        self, name: str, collector_fn: Callable[[], Awaitable[list[MentionData]]]
    ) -> list[MentionData]:
        """Collect from a source with circuit breaker protection."""
        cb = self.circuit_breakers[name]
        if not cb.can_execute():
            logger.info(f"Circuit breaker open for {name}, skipping")
            return []

        try:
            mentions = await collector_fn()
            cb.record_success()
            return mentions
        except Exception as e:
            cb.record_failure()
            logger.error(
                f"Collection failed for {name}: {e}",
                extra={"data": {"collector": name, "error": str(e)}},
            )
            return []

    def _should_run(self, name: str, interval: int) -> bool:
        """Check if enough time has passed since last run."""
        last = self._last_run.get(name)
        if last is None:
            return True
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed >= interval

    async def collect_all(self) -> list[MentionData]:
        """
        Collect from all sources concurrently, respecting intervals.

        Only runs each collector if its configured interval has elapsed.
        """
        tasks: list[asyncio.Task] = []
        task_names: list[str] = []

        intervals = config.scan_intervals

        if self._should_run("reddit", intervals.reddit):
            tasks.append(asyncio.create_task(
                self._safe_collect("reddit", self.reddit.collect)
            ))
            task_names.append("reddit")
            self._last_run["reddit"] = datetime.now(timezone.utc)

        if self._should_run("google_trends", intervals.google_trends):
            tasks.append(asyncio.create_task(
                self._safe_collect("google_trends", self.google_trends.collect)
            ))
            task_names.append("google_trends")
            self._last_run["google_trends"] = datetime.now(timezone.utc)

        if self._should_run("twitter", intervals.twitter):
            tasks.append(asyncio.create_task(
                self._safe_collect("twitter", self.twitter.collect)
            ))
            task_names.append("twitter")
            self._last_run["twitter"] = datetime.now(timezone.utc)

        if self._should_run("news", intervals.news):
            tasks.append(asyncio.create_task(
                self._safe_collect("news", self.news.collect)
            ))
            task_names.append("news")
            self._last_run["news"] = datetime.now(timezone.utc)

        if self._should_run("onchain", intervals.onchain):
            tasks.append(asyncio.create_task(
                self._safe_collect("onchain", self.onchain.collect)
            ))
            task_names.append("onchain")
            self._last_run["onchain"] = datetime.now(timezone.utc)

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_mentions: list[MentionData] = []
        for name, result in zip(task_names, results):
            if isinstance(result, Exception):
                logger.error(f"Collection task {name} raised: {result}")
                continue
            all_mentions.extend(result)

        logger.info(
            f"Collection cycle complete: {len(all_mentions)} total mentions from {len(task_names)} sources",
            extra={"data": {"total": len(all_mentions), "sources": task_names}},
        )
        return all_mentions

    def _aggregate_mentions(
        self, mentions: list[MentionData]
    ) -> dict[str, dict[str, list[MentionData]]]:
        """
        Group mentions by ticker, then by source.

        Returns: {ticker: {source: [mentions]}}
        """
        grouped: dict[str, dict[str, list[MentionData]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for m in mentions:
            ticker = m.ticker
            if not ticker:
                continue
            grouped[ticker][m.source.value].append(m)
        return grouped

    async def process_cycle(self) -> list[SocialArbSignal]:
        """
        Run one complete processing cycle.

        Pipeline: Collect → Aggregate → Detect Anomalies → Filter Bots →
        Analyze Sentiment → Cross-Validate → Map Tickers →
        Estimate Edge → Generate Signals
        """
        signals: list[SocialArbSignal] = []

        # Step 1: Collect from all sources
        all_mentions = await self.collect_all()
        if not all_mentions:
            logger.info("No mentions collected this cycle")
            return signals

        # Step 2: Resolve any keywords without tickers
        unmapped_keywords = set()
        for m in all_mentions:
            if not m.ticker and m.keyword:
                unmapped_keywords.add(m.keyword)

        if unmapped_keywords:
            mappings = await self.ticker_mapper.resolve_batch(list(unmapped_keywords))
            for m in all_mentions:
                if not m.ticker and m.keyword and m.keyword in mappings:
                    mapping = mappings[m.keyword]
                    if mapping:
                        m.ticker = mapping["ticker"]

        # Step 3: Aggregate by ticker and source
        grouped = self._aggregate_mentions(all_mentions)

        if not grouped:
            logger.info("No ticker-attributable mentions this cycle")
            return signals

        # Step 4: For each ticker, run the full analysis pipeline
        for ticker, sources in grouped.items():
            try:
                signal = await self._process_ticker(ticker, sources)
                if signal:
                    signals.append(signal)
            except Exception as e:
                logger.error(
                    f"Failed to process ticker {ticker}: {e}",
                    extra={"data": {"ticker": ticker, "error": str(e)}},
                    exc_info=True,
                )
                continue

        # Step 5: Log and emit all signals
        for signal in signals:
            self.signal_log.log_signal(signal)
            if self.signal_callback:
                try:
                    await self.signal_callback(signal)
                except Exception as e:
                    logger.error(f"Signal callback failed: {e}")

        logger.info(
            f"Cycle complete: {len(signals)} signals generated",
            extra={"data": {"signal_count": len(signals)}},
        )
        return signals

    async def _process_ticker(
        self, ticker: str, sources: dict[str, list[MentionData]]
    ) -> SocialArbSignal | None:
        """
        Run the full analysis pipeline for a single ticker.

        Returns a SocialArbSignal if the ticker passes all quality gates,
        or None if it doesn't meet thresholds.
        """
        # Flatten all mentions for this ticker
        all_mentions: list[MentionData] = []
        for src_mentions in sources.values():
            all_mentions.extend(src_mentions)

        # --- Anomaly Detection ---
        anomalies: list[AnomalyResult] = []
        for source_val, src_mentions in sources.items():
            count = len(src_mentions)
            source = SourceType(source_val)
            anomaly = self.anomaly_detector.detect(ticker, source, float(count))
            anomalies.append(anomaly)

        # Check if any source detected an anomaly
        has_anomaly = any(a.is_anomaly for a in anomalies)
        if not has_anomaly:
            return None  # no anomaly, no signal

        # --- Bot Filtering ---
        bot_result = self.bot_filter.filter_mentions(all_mentions, ticker)
        clean_mentions = bot_result.clean_mentions
        bot_risk = bot_result.bot_ratio

        if not clean_mentions:
            self.signal_log.log_event("all_bot_filtered", {"ticker": ticker})
            return None

        # --- Sentiment Analysis ---
        enriched = self.sentiment_analyzer.enrich_mentions(clean_mentions)

        # Compute aggregate sentiment
        sentiments = [
            m.sentiment_score for m in enriched if m.sentiment_score is not None
        ]
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0.0

        if avg_sentiment > 0.1:
            sentiment_label = SentimentLabel.BULLISH
        elif avg_sentiment < -0.1:
            sentiment_label = SentimentLabel.BEARISH
        else:
            sentiment_label = SentimentLabel.NEUTRAL

        # Compute hype score
        avg_text_len = (
            sum(len(m.text_snippet or "") for m in enriched) / len(enriched)
            if enriched
            else 0
        )
        hype_risk = self.sentiment_analyzer.compute_hype_score(
            enriched, avg_sentiment, avg_text_len
        )

        # --- Cross-Validation ---
        is_crypto = ticker in KNOWN_CRYPTO_TICKERS
        # Check if we have price data from on-chain collector
        price_change = None
        for m in all_mentions:
            if m.metadata.get("price_change_24h_pct") is not None:
                price_change = m.metadata["price_change_24h_pct"]
                break

        validation = self.cross_validator.validate(
            ticker=ticker,
            anomalies=anomalies,
            price_change_pct=price_change,
            is_crypto=is_crypto,
        )

        if not validation.is_tradeable:
            self.signal_log.log_event(
                "not_tradeable",
                {
                    "ticker": ticker,
                    "confidence": validation.final_confidence,
                    "sources": [s.value for s in validation.sources_detected],
                },
            )
            return None

        # --- Edge Estimation ---
        # Calculate virality velocity (mentions per hour)
        timestamps = [m.timestamp for m in enriched]
        if len(timestamps) >= 2:
            time_span = (max(timestamps) - min(timestamps)).total_seconds() / 3600
            velocity = len(enriched) / max(time_span, 0.1)
        else:
            velocity = float(len(enriched))

        # Determine primary source (strongest anomaly)
        primary_source = max(
            (a for a in anomalies if a.is_anomaly),
            key=lambda a: a.z_score,
        ).source

        liquidity = "high" if is_crypto and ticker in {"BTC", "ETH"} else "medium"
        if not is_crypto:
            # Rough heuristic: well-known tickers = high liquidity
            high_liq_tickers = {"AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "SPY", "QQQ"}
            if ticker in high_liq_tickers:
                liquidity = "high"

        edge_estimate = self.edge_estimator.estimate(
            ticker=ticker,
            virality_velocity=velocity,
            primary_source=primary_source,
            asset_liquidity=liquidity,
        )

        # --- Signal Generation ---
        signal = self.signal_generator.generate(
            ticker=ticker,
            validation=validation,
            edge_estimate=edge_estimate,
            mentions=enriched,
            avg_sentiment=avg_sentiment,
            sentiment_label=sentiment_label,
            bot_risk_score=bot_risk,
            hype_risk_score=hype_risk,
        )

        return signal

    async def run(self, max_cycles: int | None = None) -> None:
        """
        Main event loop — runs continuously until stopped.

        Args:
            max_cycles: If set, stop after this many cycles (for testing).
        """
        self._running = True
        cycle_count = 0

        logger.info("Social Arbitrage Orchestrator starting...")

        while self._running:
            cycle_count += 1
            logger.info(f"--- Cycle {cycle_count} ---")

            try:
                signals = await self.process_cycle()
                if signals:
                    logger.info(
                        f"Cycle {cycle_count} produced {len(signals)} signals: "
                        + ", ".join(f"{s.ticker} ({s.direction.value})" for s in signals)
                    )
            except Exception as e:
                logger.error(
                    f"Cycle {cycle_count} failed: {e}",
                    exc_info=True,
                )

            if max_cycles and cycle_count >= max_cycles:
                logger.info(f"Reached max cycles ({max_cycles}), stopping")
                break

            # Wait for the shortest interval before next cycle
            min_interval = min(
                config.scan_intervals.reddit,
                config.scan_intervals.google_trends,
                config.scan_intervals.twitter,
                config.scan_intervals.news,
                config.scan_intervals.onchain,
            )
            logger.info(f"Sleeping {min_interval}s until next cycle...")
            await asyncio.sleep(min_interval)

        logger.info("Orchestrator stopped")

    def stop(self) -> None:
        """Signal the orchestrator to stop after the current cycle."""
        self._running = False
        logger.info("Stop signal received")
