"""
Multi-source signal cross-validation.

This is the CRITICAL quality gate — a signal must appear in
at least 2 independent sources to be considered tradeable.

Scoring matrix:
- Reddit spike + Google Trends spike = 0.7
- Reddit + Google + News = 0.85
- Single source only = 0.4 (log but don't trade)
- +0.1 if on-chain data confirms (crypto)
- -0.3 if price already moved >10% (late to the trade)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import (
    AnomalyResult,
    CrossValidationResult,
    SourceType,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class CrossValidator:
    """
    Validates signals across multiple independent sources.

    The key insight: a single source can be gamed or noisy.
    When multiple independent sources confirm the same signal,
    confidence increases dramatically. This is the main defense
    against false positives and manipulation.
    """

    def __init__(self):
        self.cfg = config.cross_validation

    def validate(
        self,
        ticker: str,
        anomalies: list[AnomalyResult],
        price_change_pct: float | None = None,
        is_crypto: bool = False,
    ) -> CrossValidationResult:
        """
        Cross-validate a ticker's anomalies across sources.

        Args:
            ticker: The ticker symbol being validated
            anomalies: List of AnomalyResults from different sources
            price_change_pct: How much the price has already moved (if known)
            is_crypto: Whether this is a crypto asset (enables on-chain bonus)

        Returns:
            CrossValidationResult with final confidence and tradeable flag
        """
        # Collect unique sources that flagged this ticker as anomalous
        detected_sources: list[SourceType] = []
        evidence: dict[str, Any] = {}

        for anomaly in anomalies:
            if anomaly.is_anomaly:
                detected_sources.append(anomaly.source)
                evidence[anomaly.source.value] = {
                    "z_score": anomaly.z_score,
                    "velocity_z_score": anomaly.velocity_z_score,
                    "current_value": anomaly.current_value,
                    "rolling_mean": anomaly.rolling_mean,
                }

        # Deduplicate sources
        unique_sources = list(set(detected_sources))
        source_count = len(unique_sources)

        # Base confidence from source count
        if source_count >= 3:
            raw_confidence = self.cfg.three_source_confidence
        elif source_count == 2:
            raw_confidence = self.cfg.two_source_confidence
        elif source_count == 1:
            raw_confidence = self.cfg.single_source_confidence
        else:
            raw_confidence = 0.0

        # On-chain bonus for crypto assets
        if is_crypto and SourceType.ONCHAIN in unique_sources:
            raw_confidence = min(1.0, raw_confidence + self.cfg.onchain_bonus)

        # Z-score strength bonus — higher z-scores = stronger signal
        max_z = max((a.z_score for a in anomalies if a.is_anomaly), default=0)
        if max_z > 4.0:
            raw_confidence = min(1.0, raw_confidence + 0.05)

        # Late-to-trade penalty
        price_already_moved = False
        final_confidence = raw_confidence
        if price_change_pct is not None and abs(price_change_pct) > config.anomaly.price_already_moved_pct:
            price_already_moved = True
            final_confidence = max(0.0, final_confidence - self.cfg.late_penalty)
            logger.info(
                f"Late penalty applied to {ticker}: price already moved {price_change_pct:.1f}%",
                extra={"data": {"ticker": ticker, "price_move": price_change_pct}},
            )

        # Tradeable determination
        is_tradeable = (
            final_confidence >= self.cfg.min_tradeable_confidence
            and source_count >= config.anomaly.min_cross_sources
        )

        result = CrossValidationResult(
            ticker=ticker,
            sources_detected=[s for s in unique_sources],
            source_count=source_count,
            raw_confidence=raw_confidence,
            price_already_moved=price_already_moved,
            price_move_pct=price_change_pct,
            final_confidence=final_confidence,
            is_tradeable=is_tradeable,
            evidence=evidence,
        )

        log_level = "info" if is_tradeable else "debug"
        logger.log(
            20 if is_tradeable else 10,
            f"Cross-validation for {ticker}: "
            f"confidence={final_confidence:.2f}, "
            f"sources={source_count}, "
            f"tradeable={is_tradeable}",
            extra={"data": result.model_dump()},
        )

        return result

    def validate_batch(
        self,
        ticker_anomalies: dict[str, list[AnomalyResult]],
        price_changes: dict[str, float] | None = None,
        crypto_tickers: set[str] | None = None,
    ) -> list[CrossValidationResult]:
        """
        Validate multiple tickers at once.

        Args:
            ticker_anomalies: dict mapping ticker → list of AnomalyResults
            price_changes: optional dict mapping ticker → price change %
            crypto_tickers: set of tickers that are crypto assets
        """
        results = []
        price_changes = price_changes or {}
        crypto_tickers = crypto_tickers or set()

        for ticker, anomalies in ticker_anomalies.items():
            result = self.validate(
                ticker=ticker,
                anomalies=anomalies,
                price_change_pct=price_changes.get(ticker),
                is_crypto=ticker in crypto_tickers,
            )
            results.append(result)

        # Sort by confidence descending — highest confidence signals first
        results.sort(key=lambda r: r.final_confidence, reverse=True)
        return results
