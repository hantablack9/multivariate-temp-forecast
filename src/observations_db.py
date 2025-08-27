"""
# src/observation_db.py
Manages a SQLite database for structured log observations.
Provides a Loguru-compatible sink for writing logs and methods for querying,
analyzing, and exporting them.

Author: Hanish Paturi
Date: 2024-06-20
"""

import csv
import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger


# The Observation dataclass is a perfect fit here. It defines the structure
# of the data that this class is responsible for managing.
@dataclass
class Observation:
    """Represents a single structured log entry for database and export use."""

    text: str
    tags: list[str]
    timestamp: datetime
    section: Optional[str] = None
    level: Optional[str] = "info"

    def to_dict(self) -> dict:
        """Return a dictionary representation of the observation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "section": self.section or "",
            "tags": self.tags or [],
            "text": self.text.strip(),
            "level": self.level or "",
        }


class ObservationDB:
    """
    Manages the SQLite database for structured logs.
    Provides a Loguru-compatible sink for writing logs and a rich set of
    methods for querying, analyzing, and exporting them.
    """

    def __init__(self, db_path: str):
        """
        Initializes the database manager.

        Args:
            db_path (str): The file path for the SQLite database.
        """
        self.db_path = db_path
        self._init_db()

    def get_sink(self):
        """Returns the database sink function for use with Loguru's configuration."""
        return self._db_sink

    def _db_sink(self, message):
        """
        The actual sink function that Loguru will call for each log record.
        It extracts data from the record and writes it to the database.
        """
        record = message.record

        # Extract custom bound data from the 'extra' dict
        tags = record["extra"].get("tags", [])
        section = record["extra"].get("section")
        tags_str = ",".join(tags) if tags else ""

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO observations (timestamp, section, tags, text, level)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    record["time"].isoformat(),
                    section,
                    tags_str,
                    record["message"],
                    record["level"].name.lower(),
                ),
            )

    def _init_db(self):
        """Creates the database and the 'observations' table if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    section TEXT,
                    tags TEXT,
                    text TEXT,
                    level TEXT
                )
                """
            )

    # --- Migrated and Enhanced Analysis/Export Methods ---

    def load_all_logs(self) -> list[Observation]:
        """
        Loads all structured log observations from the SQLite database.
        This is the primary data retrieval method used by all other analysis functions.

        Returns:
            list[Observation]: A list of all Observation objects from the database.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT timestamp, section, tags, text, level FROM observations ORDER BY timestamp"
                )
                return [
                    Observation(
                        timestamp=datetime.fromisoformat(row[0]),
                        section=row[1],
                        tags=row[2].split(",") if row[2] else [],
                        text=row[3],
                        level=row[4],
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            logger.error(f"Failed to load logs from database at {self.db_path}: {e}")
            return []

    def export_to_csv(self, path: str = "observations.csv"):
        """
        Export all logs from the database to a CSV file.

        Args:
            path (str): The path to save the CSV file.
        """
        observations = self.load_all_logs()
        if not observations:
            logger.warning("No logs found in the database to export.")
            return

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "section", "tags", "text", "level"])
            writer.writeheader()
            for obs in observations:
                d = obs.to_dict()
                d["tags"] = ",".join(d["tags"])  # Flatten tags for CSV
                writer.writerow(d)
        logger.info(f"Logs successfully exported to CSV at {path}")

    def export_to_json(self, path: str = "observations.json"):
        """
        Export all logs from the database to a JSON file.

        Args:
            path (str): The path to save the JSON file.
        """
        observations = self.load_all_logs()
        if not observations:
            logger.warning("No logs found in the database to export.")
            return

        entries = [obs.to_dict() for obs in observations]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        logger.info(f"Logs successfully exported to JSON at {path}")

    def filter_by_tag(self, *queries: str) -> list[Observation]:
        """
        Return observations from the database matching all specified tags.

        Args:
            *queries (str): One or more tags to filter by.
        """
        queries_lower = {q.strip().lower() for q in queries}
        return [obs for obs in self.load_all_logs() if queries_lower.issubset(set(obs.tags))]

    def filter_by_section(self, query: str) -> list[Observation]:
        """Filter observations from the database by a substring in the section."""
        query_lower = query.strip().lower()
        return [obs for obs in self.load_all_logs() if obs.section and query_lower in obs.section.lower()]

    def filter_by_level(self, *levels: str) -> list[Observation]:
        """Filter observations from the database by one or more log levels."""
        levels_lower = {lvl.strip().lower() for lvl in levels}
        return [obs for obs in self.load_all_logs() if obs.level and obs.level.lower() in levels_lower]

    def get_last_n_logs(self, n: int = 5) -> list[Observation]:
        """Return the last n observations from the database."""
        # This is more efficient than loading all logs first.
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT timestamp, section, tags, text, level FROM observations ORDER BY timestamp DESC LIMIT ?", (n,)
            )
            # Fetch and reverse the list to maintain chronological order
            return list(
                reversed([
                    Observation(
                        timestamp=datetime.fromisoformat(row[0]),
                        section=row[1],
                        tags=row[2].split(",") if row[2] else [],
                        text=row[3],
                        level=row[4],
                    )
                    for row in cursor.fetchall()
                ])
            )

    def group_by_day(self) -> dict[str, list[Observation]]:
        """Group observations from the database by their log date."""
        grouped = defaultdict(list)
        for obs in self.load_all_logs():
            grouped[obs.timestamp.strftime("%Y-%m-%d")].append(obs)
        return dict(grouped)

    def get_summary_stats(self) -> dict[str, int]:
        """
        Provides a quick summary of the log database, such as total logs
        and counts per level.
        """
        stats = defaultdict(int)
        all_logs = self.load_all_logs()
        stats["total_logs"] = len(all_logs)
        for obs in all_logs:
            if obs.level:
                stats[f"level_{obs.level.lower()}"] += 1
        return dict(stats)

    def reset_db(self):
        """
        Drops and recreates the observations table in the database.
        USE WITH CAUTION: This will delete all existing logs.
        """
        logger.warning(f"Resetting database at {self.db_path}. All log data will be lost.")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS observations")
        self._init_db()  # Recreate the table
        logger.info("Log database has been reset.")
