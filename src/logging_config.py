"""
# src/logging_config.py
Logging configuration module using Loguru.

Author: Hanish Paturi
Date: 2025-Aug-28
"""

import os
import sys

from loguru import logger

path = os.path.dirname(os.path.abspath(__file__))
if path not in sys.path:
    sys.path.append(path)

from src.observations_db import ObservationDB


def setup_logger(log_file: str, db_path: str):
    """
    Configures the global Loguru logger with file, console, and DB sinks.
    This function should only be called ONCE at the start of the application.
    """
    # 1. Remove default handler and start fresh
    logger.remove()

    # 2. Configure Console Sink
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    )

    # 3. Configure File Sink (to append, not overwrite)
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        compression="zip",
        enqueue=True,  # Process-safe
    )

    # 4. Configure Database Sink
    db_manager = ObservationDB(db_path)
    logger.add(
        db_manager.get_sink(),
        level="DEBUG",
        enqueue=True,  # Process-safe
    )

    logger.info("Logger configured successfully. All logs will be appended.")

    # Return the configured logger instance
    return logger
