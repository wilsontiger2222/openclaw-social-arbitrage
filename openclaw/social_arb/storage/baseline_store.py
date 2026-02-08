"""
Persistent storage for rolling mention baselines.

Uses SQLite for durability — baselines survive process restarts
so we don't lose historical context when the agent restarts.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openclaw.social_arb.config import config
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineStore:
    """
    Stores and retrieves rolling mention baselines per ticker/source pair.

    Uses SQLite for simplicity and durability. Schema:
    - key: (ticker, source) composite
    - values: JSON array of historical counts
    - timestamps: JSON array of corresponding timestamps
    - updated_at: last update time
    """

    def __init__(self, db_path: str | None = None):
        self.db_path = Path(db_path or config.storage.baseline_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the SQLite database and create tables."""
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS baselines (
                ticker TEXT NOT NULL,
                source TEXT NOT NULL,
                values_json TEXT NOT NULL DEFAULT '[]',
                timestamps_json TEXT NOT NULL DEFAULT '[]',
                updated_at TEXT NOT NULL,
                PRIMARY KEY (ticker, source)
            )
        """)
        self._conn.commit()
        logger.info(f"Baseline store initialized at {self.db_path}")

    def get_baseline(
        self, ticker: str, source: str
    ) -> tuple[list[float], list[str]]:
        """
        Retrieve the stored baseline for a ticker/source pair.

        Returns (values, timestamps) lists.
        """
        cursor = self._conn.execute(
            "SELECT values_json, timestamps_json FROM baselines WHERE ticker = ? AND source = ?",
            (ticker, source),
        )
        row = cursor.fetchone()
        if row:
            values = json.loads(row[0])
            timestamps = json.loads(row[1])
            return values, timestamps
        return [], []

    def update_baseline(
        self,
        ticker: str,
        source: str,
        values: list[float],
        timestamps: list[str],
        max_history: int = 500,
    ) -> None:
        """
        Store/update the baseline for a ticker/source pair.

        Trims to max_history to prevent unbounded growth.
        """
        # Trim to max history
        if len(values) > max_history:
            values = values[-max_history:]
            timestamps = timestamps[-max_history:]

        self._conn.execute(
            """
            INSERT INTO baselines (ticker, source, values_json, timestamps_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker, source) DO UPDATE SET
                values_json = excluded.values_json,
                timestamps_json = excluded.timestamps_json,
                updated_at = excluded.updated_at
            """,
            (
                ticker,
                source,
                json.dumps(values),
                json.dumps(timestamps),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def append_value(
        self, ticker: str, source: str, value: float
    ) -> None:
        """Append a single value to an existing baseline."""
        values, timestamps = self.get_baseline(ticker, source)
        values.append(value)
        timestamps.append(datetime.now(timezone.utc).isoformat())
        self.update_baseline(ticker, source, values, timestamps)

    def get_all_tickers(self) -> list[str]:
        """Get all tickers with stored baselines."""
        cursor = self._conn.execute(
            "SELECT DISTINCT ticker FROM baselines"
        )
        return [row[0] for row in cursor.fetchall()]

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None
