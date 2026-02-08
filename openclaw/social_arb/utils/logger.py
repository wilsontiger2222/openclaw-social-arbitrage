"""
Structured JSON logging for the Social Arbitrage pipeline.

Every signal, anomaly, and error is logged in machine-readable format
for post-hoc analysis and backtesting.
"""

import logging
import json
import sys
from datetime import datetime, timezone
from typing import Any


class JSONFormatter(logging.Formatter):
    """Formats log records as single-line JSON for easy parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_data: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }
        # Attach any extra fields passed via `extra={"data": {...}}`
        if hasattr(record, "data"):
            log_data["data"] = record.data
        if record.exc_info and record.exc_info[1]:
            log_data["exception"] = str(record.exc_info[1])
        return json.dumps(log_data, default=str)


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create a structured JSON logger.

    Usage:
        logger = get_logger(__name__)
        logger.info("Signal detected", extra={"data": {"ticker": "AAPL", "z_score": 3.2}})
    """
    logger = logging.getLogger(f"openclaw.social_arb.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger
