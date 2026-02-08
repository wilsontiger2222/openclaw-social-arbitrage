"""
Signal logger for backtesting and review.

Logs every signal (tradeable and non-tradeable) to a JSONL file
for post-hoc analysis, strategy refinement, and performance tracking.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openclaw.social_arb.config import config
from openclaw.social_arb.signals.signal_schema import SocialArbSignal
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class SignalLog:
    """
    Append-only JSONL log of all signals generated.

    Each line is a complete JSON object representing one signal,
    making it easy to stream, grep, and load into pandas for analysis.
    """

    def __init__(self, log_path: str | None = None):
        self.log_path = Path(log_path or config.storage.signal_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_signal(self, signal: SocialArbSignal) -> None:
        """Append a signal to the log file."""
        try:
            record = signal.model_dump()
            record["_logged_at"] = datetime.now(timezone.utc).isoformat()

            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")

            logger.info(
                f"Logged signal: {signal.ticker} ({signal.direction.value})",
                extra={"data": {"signal_id": signal.signal_id, "ticker": signal.ticker}},
            )
        except IOError as e:
            logger.error(f"Failed to log signal: {e}")

    def log_event(self, event_type: str, data: dict[str, Any]) -> None:
        """Log a non-signal event (anomaly, skipped signal, error, etc.)."""
        try:
            record = {
                "event_type": event_type,
                "_logged_at": datetime.now(timezone.utc).isoformat(),
                **data,
            }
            with open(self.log_path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except IOError as e:
            logger.error(f"Failed to log event: {e}")

    def read_signals(
        self, ticker: str | None = None, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Read recent signals from the log, optionally filtered by ticker."""
        signals: list[dict[str, Any]] = []

        if not self.log_path.exists():
            return signals

        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                        # Only return signal records (not events)
                        if "signal_id" not in record:
                            continue
                        if ticker and record.get("ticker") != ticker:
                            continue
                        signals.append(record)
                    except json.JSONDecodeError:
                        continue
        except IOError as e:
            logger.error(f"Failed to read signal log: {e}")

        # Return most recent signals
        return signals[-limit:]

    def get_signal_count(self) -> int:
        """Count total signals logged."""
        if not self.log_path.exists():
            return 0
        count = 0
        try:
            with open(self.log_path, "r") as f:
                for line in f:
                    if '"signal_id"' in line:
                        count += 1
        except IOError:
            pass
        return count
