"""
Signal generator — produces structured trade signals from validated data.

Takes cross-validated anomalies, sentiment data, and edge estimates
to produce the final SocialArbSignal that OpenClaw's execution layer consumes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from openclaw.social_arb.config import config, KNOWN_CRYPTO_TICKERS
from openclaw.social_arb.signals.signal_schema import (
    AssetClass,
    CrossValidationResult,
    Direction,
    EdgeEstimate,
    MentionData,
    SentimentLabel,
    SocialArbSignal,
    SourceType,
)
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """
    Produces final trade signals from validated social data.

    Combines:
    - Cross-validation confidence
    - Sentiment direction
    - Edge decay estimate
    - Risk scores (bot, hype)
    - Position sizing
    """

    def __init__(self):
        self.defaults = config.signal_defaults

    def _determine_asset_class(self, ticker: str) -> AssetClass:
        """Classify a ticker into its asset class."""
        if ticker in KNOWN_CRYPTO_TICKERS:
            return AssetClass.CRYPTO

        # Common ETF tickers
        etf_tickers = {
            "SPY", "QQQ", "IWM", "DIA", "VTI", "VOO", "ARKK",
            "XLF", "XLE", "XLK", "GLD", "SLV", "TLT", "VIX",
        }
        if ticker in etf_tickers:
            return AssetClass.ETF

        return AssetClass.EQUITY

    def _determine_direction(
        self,
        avg_sentiment: float,
        sentiment_label: SentimentLabel,
    ) -> Direction:
        """
        Determine trade direction from sentiment analysis.

        BULLISH sentiment → LONG
        BEARISH sentiment → SHORT
        NEUTRAL → default to LONG (social buzz is usually bullish)
        """
        if sentiment_label == SentimentLabel.BEARISH and avg_sentiment < -0.3:
            return Direction.SHORT
        return Direction.LONG

    def _calculate_position_size(
        self,
        confidence: float,
        bot_risk: float,
        hype_risk: float,
    ) -> float:
        """
        Calculate suggested position size as % of portfolio.

        Higher confidence and lower risk = larger position.
        Scales linearly between min and max position sizes.
        """
        # Start with confidence-based sizing
        confidence_range = self.defaults.max_position_pct - self.defaults.min_position_pct
        base_size = self.defaults.min_position_pct + (confidence * confidence_range)

        # Risk penalties
        risk_factor = 1.0 - (bot_risk * 0.3 + hype_risk * 0.3)
        risk_factor = max(0.3, risk_factor)

        position_pct = base_size * risk_factor
        return round(
            max(self.defaults.min_position_pct, min(self.defaults.max_position_pct, position_pct)),
            2,
        )

    def _calculate_stop_loss(
        self,
        asset_class: AssetClass,
        confidence: float,
    ) -> float:
        """
        Set stop loss based on asset class and confidence.

        Crypto gets wider stops due to higher volatility.
        Higher confidence = tighter stops (we're more sure of direction).
        """
        base_stop = self.defaults.default_stop_loss_pct

        if asset_class == AssetClass.CRYPTO:
            base_stop *= 1.5  # wider stops for crypto
        elif asset_class == AssetClass.ETF:
            base_stop *= 0.7  # tighter stops for ETFs

        # Higher confidence → tighter stops
        confidence_adj = 1.0 - (confidence - 0.5) * 0.3
        return round(base_stop * max(0.5, confidence_adj), 1)

    def _calculate_take_profit(
        self,
        asset_class: AssetClass,
        confidence: float,
        edge_hours_max: float,
    ) -> float:
        """
        Set take profit target based on asset class and edge window.

        Shorter edge = lower take profit target (take what you can get).
        Crypto gets higher targets due to higher volatility.
        """
        base_tp = self.defaults.default_take_profit_pct

        if asset_class == AssetClass.CRYPTO:
            base_tp *= 1.5
        elif asset_class == AssetClass.ETF:
            base_tp *= 0.6

        # Shorter edge → lower target
        if edge_hours_max < 4:
            base_tp *= 0.5  # short edge, take quick profits
        elif edge_hours_max < 12:
            base_tp *= 0.75

        return round(base_tp * confidence, 1)

    def generate(
        self,
        ticker: str,
        validation: CrossValidationResult,
        edge_estimate: EdgeEstimate,
        mentions: list[MentionData],
        avg_sentiment: float,
        sentiment_label: SentimentLabel,
        bot_risk_score: float,
        hype_risk_score: float,
    ) -> SocialArbSignal | None:
        """
        Generate a trade signal from validated data.

        Returns None if the signal doesn't meet minimum quality thresholds.
        """
        if not validation.is_tradeable:
            logger.info(
                f"Signal for {ticker} not tradeable (confidence={validation.final_confidence:.2f})",
                extra={"data": {"ticker": ticker, "confidence": validation.final_confidence}},
            )
            return None

        asset_class = self._determine_asset_class(ticker)
        direction = self._determine_direction(avg_sentiment, sentiment_label)
        position_pct = self._calculate_position_size(
            validation.final_confidence, bot_risk_score, hype_risk_score
        )
        stop_loss = self._calculate_stop_loss(asset_class, validation.final_confidence)
        take_profit = self._calculate_take_profit(
            asset_class, validation.final_confidence, edge_estimate.estimated_edge_hours_max
        )

        # Build source labels for the signal
        source_labels = []
        for src in validation.sources_detected:
            source_labels.append(f"{src.value}_spike")

        # Compile raw data for auditability
        raw_data = {
            "validation": validation.model_dump(),
            "edge_estimate": edge_estimate.model_dump(),
            "mention_count": len(mentions),
            "avg_sentiment": avg_sentiment,
            "sentiment_label": sentiment_label.value,
        }

        signal = SocialArbSignal(
            ticker=ticker,
            asset_class=asset_class,
            direction=direction,
            confidence=round(validation.final_confidence, 3),
            sources=source_labels,
            edge_decay_estimate=edge_estimate.edge_decay_label,
            suggested_position_pct=position_pct,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit,
            raw_data=raw_data,
            bot_risk_score=round(bot_risk_score, 3),
            hype_risk_score=round(hype_risk_score, 3),
            metadata={
                "direction_reasoning": f"Avg sentiment: {avg_sentiment:.2f} ({sentiment_label.value})",
                "position_sizing_reasoning": (
                    f"Confidence {validation.final_confidence:.2f} × "
                    f"risk factor {1.0 - bot_risk_score * 0.3 - hype_risk_score * 0.3:.2f}"
                ),
            },
        )

        logger.info(
            f"Signal generated: {direction.value.upper()} {ticker} "
            f"(confidence={signal.confidence}, position={position_pct}%)",
            extra={"data": signal.model_dump()},
        )

        return signal
