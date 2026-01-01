"""Logging helpers.

- Uses Python's stdlib logging with RichHandler for readable console output.
- Writes a file log for full reproducibility.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from rich.logging import RichHandler


def setup_logging(log_path: Path, level: int = logging.INFO) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("layer_time")
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers on re-entry.
    if logger.handlers:
        return logger

    # Console handler (pretty)
    console_handler = RichHandler(rich_tracebacks=True, show_time=True, show_level=True)
    console_handler.setLevel(level)

    # File handler (complete)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(fmt)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
