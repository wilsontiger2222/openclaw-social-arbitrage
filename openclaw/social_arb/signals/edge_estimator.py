"""
Edge decay estimator.

Estimates how long the information edge will last based on:
- Virality velocity (faster spread = shorter edge)
- Source type (Reddit DD = longer edge, Twitter viral = shorter)
- Market hours (after-hours signal = edge lasts until open)
- Asset liquidity (large cap = faster decay, small cap = slower)

The edge is the time window during which the social signal
has not yet been fully priced into the market.
"""

from __future__ import annotations

from datetime import datetime, timezone, time as dtime
from typing import Literal

from openclaw.social_arb.signals.signal_schema import (
    EdgeEstimate,
    SourceType,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)

# US equity market hours (ET)
MARKET_OPEN = dtime(9, 30)
MARKET_CLOSE = dtime(16, 0)


def _is_market_hours(dt: datetime) -> bool:
    """Check if current time is during US equity market hours (rough EST check)."""
    # Rough ET conversion: UTC - 5
    et_hour = (dt.hour - 5) % 24
    et_time = dtime(et_hour, dt.minute)
    weekday = dt.weekday()  # 0=Monday
    return weekday < 5 and MARKET_OPEN <= et_time <= MARKET_CLOSE


class EdgeEstimator:
    """
    Estimates the remaining alpha window for a social signal.

    The faster information spreads and the more liquid the asset,
    the shorter the trading edge. Signals detected after market
    hours have a natural edge extension until the next open.
    """

    # Base edge windows by primary source (hours)
    SOURCE_EDGE_BASE: dict[str, tuple[float, float]] = {
        SourceType.REDDIT.value: (4.0, 24.0),       # DD posts take time to spread
        SourceType.TWITTER.value: (1.0, 6.0),        # Twitter moves fast
        SourceType.GOOGLE_TRENDS.value: (2.0, 12.0), # Trends build more slowly
        SourceType.NEWS.value: (0.5, 4.0),            # News gets priced in quickly
        SourceType.ONCHAIN.value: (1.0, 8.0),         # On-chain takes time to interpret
    }

    # Liquidity multipliers (smaller = faster decay)
    LIQUIDITY_MULTIPLIER: dict[str, float] = {
        "high": 0.5,    # mega-cap, high volume → edge decays fast
        "medium": 1.0,  # mid-cap, normal
        "low": 2.0,     # small/micro-cap → less efficient, edge lasts longer
    }

    def estimate(
        self,
        ticker: str,
        virality_velocity: float,
        primary_source: SourceType,
        asset_liquidity: Literal["high", "medium", "low"] = "medium",
        detected_at: datetime | None = None,
    ) -> EdgeEstimate:
        """
        Estimate the remaining edge for a given signal.

        Args:
            ticker: Asset symbol
            virality_velocity: Mentions per hour (higher = faster spread)
            primary_source: The strongest source contributing to the signal
            asset_liquidity: How liquid/efficient the asset is
            detected_at: When the signal was first detected

        Returns:
            EdgeEstimate with min/max hours and human-readable label
        """
        if detected_at is None:
            detected_at = datetime.now(timezone.utc)

        # Start with base edge from source type
        base_min, base_max = self.SOURCE_EDGE_BASE.get(
            primary_source.value, (2.0, 12.0)
        )

        # Apply liquidity multiplier
        liq_mult = self.LIQUIDITY_MULTIPLIER.get(asset_liquidity, 1.0)
        adj_min = base_min * liq_mult
        adj_max = base_max * liq_mult

        # Virality velocity adjustment:
        # Higher velocity = information spreads faster = shorter edge
        # Use logarithmic scaling to avoid extreme values
        if virality_velocity > 0:
            import math
            velocity_factor = 1.0 / (1.0 + math.log1p(virality_velocity / 10))
            adj_min *= velocity_factor
            adj_max *= velocity_factor

        # Market hours adjustment:
        # If detected after hours, edge extends to next market open
        market_open = _is_market_hours(detected_at)
        reasoning_parts = []

        if not market_open and asset_liquidity != "crypto":
            # Add hours until next market open
            # Rough estimate: if it's evening, ~14-16 hours until open
            hours_to_open = 14.0  # conservative estimate
            adj_min += hours_to_open * 0.5  # some pre-market activity
            adj_max += hours_to_open
            reasoning_parts.append("after-hours detection extends edge to next open")

        # Floor values
        adj_min = max(0.5, adj_min)
        adj_max = max(adj_min + 0.5, adj_max)

        # Build human-readable label
        if adj_max <= 2:
            label = f"{adj_min:.0f}-{adj_max:.0f} hours"
        elif adj_max <= 24:
            label = f"{adj_min:.0f}-{adj_max:.0f} hours"
        elif adj_max <= 72:
            label = f"{adj_min / 24:.0f}-{adj_max / 24:.0f} days"
        else:
            label = f"{adj_min / 24:.0f}-{adj_max / 24:.0f} days"

        # Build reasoning
        reasoning_parts.insert(
            0,
            f"Source: {primary_source.value} (base: {base_min:.0f}-{base_max:.0f}h)"
        )
        reasoning_parts.append(
            f"Liquidity: {asset_liquidity} (mult: {liq_mult}x)"
        )
        reasoning_parts.append(
            f"Virality: {virality_velocity:.1f} mentions/hr"
        )
        reasoning = "; ".join(reasoning_parts)

        estimate = EdgeEstimate(
            ticker=ticker,
            virality_velocity=virality_velocity,
            primary_source=primary_source,
            market_is_open=market_open,
            asset_liquidity=asset_liquidity,
            estimated_edge_hours_min=round(adj_min, 1),
            estimated_edge_hours_max=round(adj_max, 1),
            edge_decay_label=label,
            reasoning=reasoning,
        )

        logger.info(
            f"Edge estimate for {ticker}: {label}",
            extra={"data": estimate.model_dump()},
        )

        return estimate
