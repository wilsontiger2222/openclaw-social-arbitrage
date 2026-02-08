"""
Statistical anomaly detection for social media mention spikes.

Uses z-score method against rolling baselines (EWMA) to detect
when a ticker's social activity is abnormally high. Also detects
velocity anomalies — sudden rate-of-change spikes that indicate
a topic is going viral right now.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import (
    AnomalyResult,
    MentionData,
    SourceType,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class AnomalyDetector:
    """
    Detects anomalous spikes in ticker mention frequency.

    Maintains per-ticker, per-source rolling baselines using
    Exponentially Weighted Moving Average (EWMA) so the baseline
    adapts to gradually changing popularity without flagging
    normal growth as anomalous.

    Two detection modes:
    1. Level anomaly: Current mention count is z_score_threshold
       standard deviations above the rolling mean.
    2. Velocity anomaly: The rate of change (mentions per hour)
       itself is anomalous — catches things going viral even if
       the absolute level isn't huge yet.
    """

    def __init__(self):
        self.z_threshold = config.anomaly.z_score_threshold
        self.velocity_z_threshold = config.anomaly.velocity_z_threshold
        self.min_mentions = config.anomaly.min_mentions_absolute
        self.ewma_span = config.anomaly.ewma_span

        # Rolling baselines: {(ticker, source): [historical_counts]}
        self._history: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._timestamps: dict[tuple[str, str], list[datetime]] = defaultdict(list)

        # Maximum history length to prevent unbounded memory growth
        self._max_history = 500

    def _ewma_stats(self, values: list[float]) -> tuple[float, float]:
        """
        Compute exponentially weighted mean and std dev.

        EWMA gives more weight to recent observations, so the baseline
        adapts to trends while still detecting sudden spikes.
        """
        if len(values) < 3:
            return float(np.mean(values)), max(float(np.std(values)), 1.0)

        arr = np.array(values, dtype=np.float64)
        # Compute EWMA using pandas-style formula
        alpha = 2.0 / (self.ewma_span + 1)
        weights = np.array([(1 - alpha) ** i for i in range(len(arr) - 1, -1, -1)])
        weights /= weights.sum()

        ewma_mean = float(np.dot(weights, arr))
        # Weighted std dev
        ewma_var = float(np.dot(weights, (arr - ewma_mean) ** 2))
        ewma_std = max(np.sqrt(ewma_var), 1.0)  # floor at 1 to avoid div-by-zero

        return ewma_mean, ewma_std

    def _compute_velocity(
        self, key: tuple[str, str], current_value: float
    ) -> float | None:
        """
        Compute the rate-of-change (velocity) of mentions.

        Returns mentions-per-hour delta, or None if insufficient history.
        Velocity matters: 10 → 100 mentions in 1 hour is more significant
        than the same change over a week.
        """
        timestamps = self._timestamps[key]
        values = self._history[key]

        if len(values) < 2 or len(timestamps) < 2:
            return None

        prev_value = values[-1]
        prev_time = timestamps[-1]
        now = datetime.now(timezone.utc)

        hours_elapsed = max((now - prev_time).total_seconds() / 3600, 0.01)
        velocity = (current_value - prev_value) / hours_elapsed

        return velocity

    def update_baseline(
        self, ticker: str, source: SourceType, value: float
    ) -> None:
        """
        Add a new observation to the rolling baseline.

        Called after each collection cycle to keep baselines current.
        """
        key = (ticker, source.value)
        self._history[key].append(value)
        self._timestamps[key].append(datetime.now(timezone.utc))

        # Trim to prevent memory growth
        if len(self._history[key]) > self._max_history:
            self._history[key] = self._history[key][-self._max_history:]
            self._timestamps[key] = self._timestamps[key][-self._max_history:]

    def detect(
        self, ticker: str, source: SourceType, current_value: float
    ) -> AnomalyResult:
        """
        Check if the current mention count is anomalous.

        Returns an AnomalyResult with z-scores and anomaly flag.
        The anomaly is flagged if EITHER the level or velocity
        exceeds their respective thresholds.
        """
        key = (ticker, source.value)
        history = self._history[key]

        # Need minimum history for meaningful detection
        if len(history) < 5:
            # Not enough data — record but don't flag
            self.update_baseline(ticker, source, current_value)
            return AnomalyResult(
                ticker=ticker,
                source=source,
                current_value=current_value,
                rolling_mean=current_value,
                rolling_std=0.0,
                z_score=0.0,
                velocity_z_score=None,
                is_anomaly=False,
            )

        # Compute baseline stats using EWMA
        ewma_mean, ewma_std = self._ewma_stats(history)

        # Z-score for level anomaly
        z_score = (current_value - ewma_mean) / ewma_std

        # Velocity anomaly detection
        velocity = self._compute_velocity(key, current_value)
        velocity_z = None
        if velocity is not None and len(history) >= 10:
            # Compute historical velocities for baseline
            velocities = []
            for i in range(1, len(history)):
                time_delta = max(
                    (self._timestamps[key][i] - self._timestamps[key][i - 1]).total_seconds() / 3600,
                    0.01,
                )
                velocities.append((history[i] - history[i - 1]) / time_delta)

            if velocities:
                vel_mean = float(np.mean(velocities))
                vel_std = max(float(np.std(velocities)), 0.1)
                velocity_z = (velocity - vel_mean) / vel_std

        # Flag as anomaly if level OR velocity exceeds threshold
        is_anomaly = (
            z_score >= self.z_threshold
            or (velocity_z is not None and velocity_z >= self.velocity_z_threshold)
        ) and current_value >= self.min_mentions

        if is_anomaly:
            logger.info(
                f"Anomaly detected: {ticker} from {source.value} "
                f"(z={z_score:.2f}, vel_z={velocity_z or 0:.2f}, value={current_value})",
                extra={
                    "data": {
                        "ticker": ticker,
                        "source": source.value,
                        "z_score": z_score,
                        "velocity_z": velocity_z,
                        "current": current_value,
                        "mean": ewma_mean,
                    }
                },
            )

        # Update baseline AFTER detection so current spike doesn't
        # immediately become the new normal
        self.update_baseline(ticker, source, current_value)

        return AnomalyResult(
            ticker=ticker,
            source=source,
            current_value=current_value,
            rolling_mean=ewma_mean,
            rolling_std=ewma_std,
            z_score=z_score,
            velocity_z_score=velocity_z,
            is_anomaly=is_anomaly,
        )

    def detect_batch(
        self, aggregated: dict[tuple[str, str], float]
    ) -> list[AnomalyResult]:
        """
        Detect anomalies for a batch of (ticker, source) pairs.

        Args:
            aggregated: dict mapping (ticker, source_value) to mention counts
        """
        results = []
        for (ticker, source_val), value in aggregated.items():
            source = SourceType(source_val)
            result = self.detect(ticker, source, value)
            results.append(result)
        return results

    def get_baseline(self, ticker: str, source: SourceType) -> dict[str, float]:
        """Get current baseline stats for debugging/monitoring."""
        key = (ticker, source.value)
        history = self._history[key]
        if not history:
            return {"mean": 0.0, "std": 0.0, "samples": 0}
        mean, std = self._ewma_stats(history)
        return {"mean": mean, "std": std, "samples": len(history)}
