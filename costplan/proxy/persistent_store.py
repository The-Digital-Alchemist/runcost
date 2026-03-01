"""Persistent store for proxy call records. Used with --state-db and --budget-window."""

import sqlite3
import time
from pathlib import Path
from typing import Optional


def _init_db(conn: sqlite3.Connection) -> None:
    """Create proxy_calls table if not exists."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS proxy_calls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp REAL NOT NULL,
            model TEXT,
            actual_cost REAL NOT NULL,
            input_tokens INTEGER DEFAULT 0,
            output_tokens INTEGER DEFAULT 0,
            cache_read_tokens INTEGER DEFAULT 0,
            cache_creation_tokens INTEGER DEFAULT 0
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_proxy_calls_timestamp ON proxy_calls(timestamp)"
    )
    conn.commit()


class PersistentCallStore:
    """SQLite-backed store for proxy call records. Thread-safe via per-call connection."""

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            _init_db(conn)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def insert(
        self,
        timestamp: float,
        model: str,
        actual_cost: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Insert a call record."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO proxy_calls
                (timestamp, model, actual_cost, input_tokens, output_tokens,
                 cache_read_tokens, cache_creation_tokens)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    timestamp,
                    model,
                    actual_cost,
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                ),
            )
            conn.commit()

    def spent_in_window(self, window_seconds: float) -> float:
        """Sum actual_cost for records within the last window_seconds."""
        cutoff = time.time() - window_seconds
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(SUM(actual_cost), 0) AS total FROM proxy_calls WHERE timestamp > ?",
                (cutoff,),
            ).fetchone()
            return float(row["total"])

    def prune_older_than(self, seconds: float) -> int:
        """Delete records older than seconds. Returns deleted count."""
        cutoff = time.time() - seconds
        with self._conn() as conn:
            cursor = conn.execute("DELETE FROM proxy_calls WHERE timestamp < ?", (cutoff,))
            conn.commit()
            return cursor.rowcount

    def clear(self) -> None:
        """Delete all records. Used when manually resetting budget."""
        with self._conn() as conn:
            conn.execute("DELETE FROM proxy_calls")
            conn.commit()
