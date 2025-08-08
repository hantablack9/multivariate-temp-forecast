"""
# Observation Logger Factory
# src/components/logger_factory.py

This module provides a flexible logging system for tracking observations with optional metadata.
It supports various formats including plain text, SQLite, CSV, JSON, and Markdown.
It is designed to be extensible and can be integrated into larger data processing pipelines.
The logger can be used to log events, errors, and general notes in a structured manner.
It also supports grouping observations by date and filtering by tags, sections, and time ranges.
It is suitable for data science projects, especially those involving time series analysis and
    machine learning.
The logger can be initialized with a log file and an optional SQLite database for persistence.
# It provides methods to reset logs, update tags, and log observations with various metadata.
"""

import csv
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Optional, Union


@dataclass
class Observation:
    """
    Represents a single logged observation with optional metadata.

    Attributes:
        text (str): The content of the observation.
        timestamp (datetime): The time the observation was logged.
        section (Optional[str]): A category or logical grouping.
        tags (Optional[List[str]]): Tags for filtering and categorization.
        level (Optional[str]): Optional priority indicator (e.g., high, medium, low).
    """

    text: str
    timestamp: datetime
    section: Optional[str] = None
    tags: Optional[list[str]] = None
    level: Optional[str] = None

    def format(self) -> str:
        """Return a formatted string representation suitable for writing to file."""
        head = f"[{self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}]"
        if self.section:
            head += f" [Section: {self.section}]"
        if self.tags:
            head += f" [Tags: {', '.join(self.tags)}]"
        if self.level:
            head += f" [Level: {self.level}]"
        return f"{head}\n{self.text.strip()}\n{'-' * 80}\n"

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
    Logger for tracking observations, supporting filtering, persistence, and export.

    Supports logging to plain text, SQLite, CSV, JSON, and Markdown formats.
    """

    DEFAULT_TAGS: ClassVar[dict[str, str]] = {
        "meta": "Meta level tags",
        "data_elt": "Extract-Load-Transform functionality related logs",
        "ml": "Machine Learning related logs",
        "eda": "Exploratory Data Analysis notes",
        "bug": "Bug or issue encountered",
        "note": "General notes",
        "test": "Test run logs",
        "review": "Code/model/data reviews",
        "todo": "Tasks to be completed",
        "data": "Dataset specific notes",
        "config": "Configuration related",
        "init": "Initialization logs",
        "infra": "Infrastructure changes",
        "debug": "Debugging notes",
        "info": "Informational messages",
        "warning": "Warnings or potential issues",
    }

    LEVELS: ClassVar[list[str]] = [
        "debug",
        "info",
        "warning",
    ]

    def __init__(
        self,
        log_file: str = "observations.txt",
        db_path: Optional[str] = None,
        tag_dict: Optional[dict[str, str]] = None,
    ):
        """
        Initialize the logger.

        Args:
            log_file (str): Path to the plain text log file.
            db_path (Optional[str]): Path to SQLite database for persistence.
            tag_dict (Optional[Dict[str, str]]): Optional user-defined tags.
        """
        self.log_file = Path(log_file)
        self.log_file.touch(exist_ok=True)
        self.db_path = db_path
        self.tags = self.DEFAULT_TAGS.copy()
        if tag_dict:
            self.tags.update({k.lower(): v for k, v in tag_dict.items()})
        if db_path:
            self._init_db()

    def _init_db(self):
        """Initialize the SQLite database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS observations (
                    timestamp TEXT NOT NULL,
                    section TEXT,
                    tags TEXT,
                    text TEXT,
                    level TEXT
                )
            """)

    def reset_logs(self, reset_file: bool = True, reset_db: bool = True) -> None:
        """
        Clears all current logs from file and/or database.

        Parameters:
            reset_file (bool): If True, truncates the log file.
            reset_db (bool): If True, drops and recreates the observations table in the database.
        """
        if reset_file and os.path.exists(self.log_file):
            open(self.log_file, "w", encoding="utf-8").close()

        if reset_db and self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DROP TABLE IF EXISTS observations")
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
        """Update the tag dictionary with new user-defined tags."""
        self.tags.update({k.lower(): v for k, v in new_tags.items()})

    def log(
        self,
        text: str,
        section: Optional[str] = None,
        tag: Optional[Union[str, list[str]]] = None,
        level: Optional[str] = None,
    ):
        """
        Add a new observation log entry.

        Args:
            text (str): Observation content.
            section (Optional[str]): Logical section/category.
            tag (Optional[Union[str, List[str]]]): One or more tags.
            level (Optional[str]): Priority label.

        Raises:
            ValueError: If any tag is not registered.
        """
        tag_list = [tag] if isinstance(tag, str) else (tag or [])
        tag_list = [t.strip().lower() for t in tag_list]

        invalid_tags = [t for t in tag_list if t not in self.tags]
        if invalid_tags:
            print(f"Invalid tags: {invalid_tags}. Use 'update_tags()' to register them.")
            raise ValueError

        obs = Observation(
            text=text.strip(),
            level=level.strip().lower() if level in self.LEVELS else None,
            timestamp=datetime.now(),
            section=section.strip() if section else None,
            tags=tag_list,
        )

        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(obs.format())

        if self.db_path:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO observations (timestamp, section, tags, text,level) VALUES (?, ?, ?, ?, ?)",
                    (obs.timestamp.isoformat(), obs.section, ",".join(obs.tags), obs.text, obs.level),
                )

    def load_logs(self, level: Optional[str] = None) -> list[Observation]:
        """
        Load all logs from the text file.

        Returns:
            List[Observation]: Parsed log entries.
        """
        observations = []
        with open(self.log_file, encoding="utf-8") as f:
            content = f.read().strip()
            entries = content.split("-" * 80)

        for entry in entries:
            if not entry.strip():
                continue

            lines = entry.strip().split("\n")
            header = lines[0]
            text = "\n".join(lines[1:]).strip()

            ts = re.search(r"\[\s*(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\s*\]", header, flags=re.IGNORECASE)
            section = re.search(r"\[\s*section\s*:\s*(.*?)\s*\]", header, flags=re.IGNORECASE)
            tags = re.search(r"\[\s*tags\s*:\s*(.*?)\s*\]", header, flags=re.IGNORECASE)
            level = re.search(r"\[\s*priority\s*:\s*(.*?)\s*\]", header, flags=re.IGNORECASE)

            if ts:
                timestamp = datetime.strptime(ts.group(1), "%Y-%m-%d %H:%M:%S")
                observations.append(
                    Observation(
                        text=text,
                        timestamp=timestamp,
                        section=section.group(1).strip() if section else None,
                        tags=[t.strip().lower() for t in tags.group(1).split(",")] if tags else [],
                        level=level.group(1).strip().lower() if level else None,
                    )
                )

        return observations

    def export_to_markdown(self, path: str = "observations.md"):
        """Export log entries to a markdown file."""
        content = self.log_file.read_text(encoding="utf-8")
        Path(path).write_text(f"# Observations Log\n\n```text\n{content}\n```", encoding="utf-8")

    def export_to_csv(self, path: str = "observations.csv"):
        """Export logs to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["timestamp", "section", "tags", "text", "level"])
            writer.writeheader()
            for e in self.load_logs():
                d = e.to_dict()
                d["tags"] = ",".join(d["tags"])
                writer.writerow(d)

    def export_to_json(self, path: str = "observations.json"):
        """Export logs to a JSON file."""
        entries = [obs.to_dict() for obs in self.load_logs()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)

    def filter_by_tag(self, *queries: str) -> list[Observation]:
        """
            Return observations matching all specified tags.

            Args:
        *queries (str): Tags to match (case-insensitive).

            Returns:
                List[Observation]: Matching logs.
        """
        queries = [q.strip().lower() for q in queries]
        return [obs for obs in self.load_logs() if all(q in obs.tags for q in queries)]

    def filter_by_section(self, query: str) -> list[Observation]:
        """
        Filter observations by a substring in the section.

        Args:
            query (str): Substring to match in section (case-insensitive).

        Returns:
            List[Observation]: Matching observations.
        """
        query_lower = query.strip().lower()
        return [obs for obs in self.load_logs() if obs.section and query_lower in obs.section.lower()]

    def filter_by_time_range(self, start: datetime, end: datetime) -> list[Observation]:
        """
        Filter logs between two datetime objects.

        Args:
            start (datetime): Start timestamp.
            end (datetime): End timestamp.

        Returns:
            List[Observation]: Logs within the time range.
        """
        return [obs for obs in self.load_logs() if start <= obs.timestamp <= end]

    def filter_by_priority(self, *levels: list[str]) -> list[Observation]:
        """
        Filter logs by priority levels.

        Args:
            *levels (str): Priority levels to match (e.g., 'high', 'low').

        Returns:
            List[Observation]: Matching logs.
        """
        levels = [lvl.strip().lower() for lvl in levels]
        return [obs for obs in self.load_logs() if obs.priority and obs.priority.lower() in levels]

    def get_last_n_logs(self, n: int = 5) -> list[Observation]:
        """
        Return the last n observations sorted by timestamp.

        Args:
            n (int): Number of entries to return.

        Returns:
            List[Observation]: Most recent observations.
        """
        return sorted(self.load_logs(), key=lambda x: x.timestamp)[-n:]

    def group_by_day(self) -> dict[datetime.date, list[Observation]]:
        """
        Group observations by their log date.

        Returns:
            Dict[datetime.date, List[Observation]]: Date-indexed observations.
        """
        grouped = defaultdict(list)
        for obs in self.load_logs():
            grouped[obs.timestamp.date()].append(obs)
        return dict(grouped)


if __name__ == "__main__":
    # Example usage
    try:
        logger = ObservationLogger(log_file="example_observations.txt", db_path="example_observations.db")
        logger.log("This is a test observation.", section="Testing", tag=["test", "note"], level="high")
        logger.export_to_csv("example_observations.csv")
        print(logger.load_logs())
        print(logger.filter_by_tag("test"))
        print(logger.get_last_n_logs(2))
    except NameError:
        logger = ObservationLogger(log_file="mylog.txt", db_path="mylog.db")
        print("Logger initialized...")
        logger.update_tags({"debug": "Debugging notes", "infra": "Infrastructure changes"})
        logger.log(
            "Started project: Multivariate TS: Weather Forecasting with LSTM.", section="Initial log", tag="meta"
        )
        print(logger.tags)
