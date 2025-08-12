"""
A powerful hybrid logger combining real-time console/file logging via Loguru
with structured data persistence in SQLite for advanced analysis and exporting.

Author: Hanish Paturi
"""

import csv
import json
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import ClassVar, Optional, Union

from loguru import logger


class InvalidParametersError(ValueError):
    """Custom exception for invalid parameter values. Used for validation."""

    INVALID_TAGS = "Invalid tags. Use 'update_tags()' to register them."


@dataclass
class Observation:
    """Represents a single structured log entry for database and export use."""

    text: str
    tags: list[str]
    timestamp: datetime
    section: Optional[str] = None
    level: Optional[str] = None

    def to_dict(self) -> dict:
        """Return a dictionary representation of the observation."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "section": self.section or "",
            "tags": self.tags or [],
            "text": self.text.strip(),
            "level": self.level or "",
        }


class ObservationLogger:
    """
    A powerful hybrid logger combining real-time console/file logging via Loguru
    with structured data persistence in SQLite for advanced analysis and exporting.

    This class provides a complete logging solution:
    - Real-time, colored console output.
    - Persistent, rotating text log files.
    - A queryable SQLite database for every log entry.
    - Methods for advanced filtering, grouping, and exporting of logs.

    Attributes:
        db_path (Optional[str]): Path to the SQLite database for structured logs.
        tags (dict): A dictionary of registered tags for validation.
    """

    DEFAULT_TAGS: ClassVar[dict[str, str]] = {
        "meta": "Meta level tags",
        "etl": "Main ETL Process Control",
        "extract": "Data Extraction Step",
        "load": "Data Loading & Cleaning Step",
        "impute": "Imputation Step",
        "filter": "Filtering Step",
        "resample": "Resampling Step",
        "error": "Error Messages",
        "validation": "Input validation",
        "io": "File Input/Output",
    }

    def __init__(
        self,
        log_file: str = "pipeline.log",
        db_path: Optional[str] = "observations.db",
        console_level: str = "INFO",
        file_level: str = "DEBUG",
    ):
        """
        Initializes the logger with console, file, and optional database sinks.

        Args:
            log_file (str): Path for the persistent text log file.
            db_path (Optional[str]): Path for the SQLite database. If None, DB features are disabled.
            console_level (str): Minimum level to show on the console.
            file_level (str): Minimum level to save to the text file.
        """
        # --- Loguru Configuration ---
        logger.remove()  # Start with a clean configuration
        logger.add(
            sys.stderr,
            level=console_level.upper(),
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )
        logger.add(
            log_file,
            level=file_level.upper(),
            rotation="10 MB",
            compression="zip",
        )

        # --- Custom Functionality ---
        self.db_path = db_path
        self.tags = self.DEFAULT_TAGS.copy()
        if self.db_path:
            self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema.

        Creates the 'observations' table if it doesn't exist.
        """
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS observations (
                    timestamp TEXT NOT NULL,
                    section TEXT,
                    tags TEXT,
                    text TEXT,
                    level TEXT
                )
                """
            )

    def update_tags(self, new_tags: dict[str, str]):
        """Update the tag dictionary with new user-defined tags.

        Args:
            new_tags (dict[str, str]): A dictionary of new tags to add.
        """
        self.tags.update({k.lower(): v for k, v in new_tags.items()})

    def log(
        self,
        text: str,
        level: str = "info",
        section: Optional[str] = None,
        tag: Optional[Union[str, list[str]]] = None,
    ):
        """
        Logs a message using Loguru and saves a structured record to the database.

        Args:
            text (str): The log message to be logged.
            level (str): The level of the log message. Default is "info".
            section (Optional[str]): The section of the log message. Default is None.
            tag (Optional[Union[str, list[str]]]): The tag(s) associated with the log message. Default is None.
        """
        # 1. Validate tags
        tag_list = [tag] if isinstance(tag, str) else (tag or [])
        invalid_tags = [t for t in tag_list if t not in self.tags]
        if invalid_tags:
            raise InvalidParametersError(InvalidParametersError.INVALID_TAGS)

        # 2. Use Loguru for real-time logging
        log_message = f"[{', '.join(tag_list).upper()}] {text}" if tag_list else text
        logger.log(level.upper(), log_message)

        # 3. Save to SQLite database if configured
        if self.db_path:
            obs = Observation(
                text=text.strip(),
                level=level.strip().lower(),
                timestamp=datetime.now(),
                section=section.strip() if section else None,
                tags=tag_list,
            )
            with sqlite3.connect(self.db_path) as conn:
                tags_str = ",".join(obs.tags) if obs.tags is not None else ""
                conn.execute(
                    "INSERT INTO observations (timestamp, section, tags, text, level) VALUES (?, ?, ?, ?, ?)",
                    (obs.timestamp.isoformat(), obs.section, tags_str, obs.text, obs.level),
                )

    def load_logs_from_db(self) -> list[Observation]:
        """Loads all structured log observations from the SQLite database.

        Returns:
            List[Observation]: A list of Observation objects representing the loaded logs.
        """
        if not self.db_path:
            logger.warning("Database path not configured. Cannot load logs.")
            return []

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT timestamp, section, tags, text, level FROM observations ORDER BY timestamp")
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

    def reset_db(self):
        """Drops and recreates the observations table in the database.

        Use this method to reset the database if needed.
        """
        if not self.db_path:
            logger.warning("Database path not configured. Cannot reset.")
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DROP TABLE IF EXISTS observations")
        self._init_db()
        logger.info("SQLite log database has been reset.")

    # --- Filtering, Grouping, and Exporting Methods (from ObservationLogger) ---
    # These now use `load_logs_from_db` as their source.

    def export_to_csv(self, path: str = "observations.csv"):
        """Export logs from the database to a CSV file.

        Args:
            path (str): The path to save the CSV file. Default is "observations.csv".
        """
        observations = self.load_logs_from_db()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "section", "tags", "text", "level"])
            writer.writeheader()
            for obs in observations:
                d = obs.to_dict()
                d["tags"] = ",".join(d["tags"])
                writer.writerow(d)
        logger.info(f"Logs exported to CSV at {path}")

    def export_to_json(self, path: str = "observations.json"):
        """Export logs from the database to a JSON file.

        Args:
            path (str): The path to save the JSON file. Default is "observations.json".
        """
        observations = self.load_logs_from_db()
        entries = [obs.to_dict() for obs in observations]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        logger.info(f"Logs exported to JSON at {path}")

    def filter_by_tag(self, *queries: str) -> list[Observation]:
        """Return observations from the database matching all specified tags.

        Args:
            *queries (str): One or more tags to filter by.

        Returns:
            List[Observation]: A list of observations that match all specified tags.
        """
        queries_lower = {q.strip().lower() for q in queries}
        return [obs for obs in self.load_logs_from_db() if queries_lower.issubset(set(obs.tags))]

    def filter_by_section(self, query: str) -> list[Observation]:
        """Filter observations from the database by a substring in the section.

        Args:
            query (str): The substring to search for in the section.

        Returns:
            List[Observation]: A list of observations that contain the specified substring in the section.
        """
        query_lower = query.strip().lower()
        return [obs for obs in self.load_logs_from_db() if obs.section and query_lower in obs.section.lower()]

    def filter_by_level(self, *levels: str) -> list[Observation]:
        """Filter observations from the database by priority levels.

        Args:
            *levels (str): One or more priority levels to filter by.

        Returns:
            List[Observation]: A list of observations that match any of the specified priority levels.
        """
        levels_lower = {lvl.strip().lower() for lvl in levels}
        return [obs for obs in self.load_logs_from_db() if obs.level and obs.level.lower() in levels_lower]

    def get_last_n_logs(self, n: int = 5) -> list[Observation]:
        """Return the last n observations from the database.

        Args:
            n (int): The number of observations to retrieve. Default is 5.

        Returns:
            List[Observation]: A list of the last n observations from the database.
        """
        return self.load_logs_from_db()[-n:]

    def group_by_day(self) -> dict[str, list[Observation]]:
        """Group observations from the database by their log date.

        Returns:
            Dict[str, List[Observation]]: A dictionary where keys are log dates and values are lists of observations for that day.
        """
        grouped = defaultdict(list)
        for obs in self.load_logs_from_db():
            grouped[obs.timestamp.strftime("%Y-%m-%d")].append(obs)
        return dict(grouped)


if __name__ == "__main__":
    # --- Example Usage ---
    print("--- Initializing ObservationLogger ---")
    # Create a logger for a specific project component
    h_logger = ObservationLogger(log_file="project.log", db_path="project.db")

    # Update with custom tags for this project
    h_logger.update_tags({"pipeline": "Data processing pipeline steps", "validation": "Data validation checks"})

    print("\n--- Logging Observations (check console and project.log) ---")
    h_logger.log("Started ETL pipeline.", section="ETL", tag="pipeline", level="info")
    h_logger.log("Raw data contains 10,000 rows.", section="Validation", tag=["data", "validation"], level="debug")
    h_logger.log("API key is missing.", section="Config", tag="error", level="error")

    print("\n--- Filtering Logs from Database ---")
    error_logs = h_logger.filter_by_level("error")
    print(f"Found {len(error_logs)} log(s) with level 'error': {error_logs[0].text}")

    print("\n--- Exporting Logs from Database ---")
    h_logger.export_to_csv("project_logs.csv")
    print("Logs exported to project_logs.csv")

    print("\n--- Resetting Database ---")
    h_logger.reset_db()
    print(f"Logs in DB after reset: {len(h_logger.load_logs_from_db())}")
