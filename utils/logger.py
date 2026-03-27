"""
utils/logger.py
~~~~~~~~~~~~~~~
Logging configuration for the LTspice AI pipeline.

Usage example::

    from utils.logger import setup_logger

    logger = setup_logger("pipeline", level="INFO", log_file="results/run.log")
    logger.info("Starting optimisation run")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Format string used for all handlers
# ---------------------------------------------------------------------------

_FORMAT = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Configure and return a named logger with console and optional file output.

    The function is idempotent: calling it twice with the same *name*
    will not add duplicate handlers.

    Parameters
    ----------
    name:
        Logger name (e.g. ``"pipeline"``, ``"core.ltspice_runner"``).
        Use ``"root"`` to configure the root logger.
    level:
        Logging level string: ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        ``"ERROR"``, or ``"CRITICAL"``.  Case-insensitive.
    log_file:
        Optional file path.  If provided, log records are also written to
        this file (in append mode).  Parent directories are created
        automatically.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Notes
    -----
    Console output goes to ``sys.stderr`` (standard Python convention).
    The file handler uses UTF-8 encoding.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    logger_name = None if name == "root" else name
    logger = logging.getLogger(logger_name)

    # Avoid adding duplicate handlers if already configured
    if logger.handlers:
        return logger

    logger.setLevel(numeric_level)
    logger.propagate = False  # don't double-log if root is also configured

    formatter = logging.Formatter(fmt=_FORMAT, datefmt=_DATE_FORMAT)

    # ------------------------------------------------------------------
    # Console handler (stderr)
    # ------------------------------------------------------------------
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ------------------------------------------------------------------
    # File handler (optional)
    # ------------------------------------------------------------------
    if log_file:
        _ensure_parent(log_file)
        try:
            file_handler = logging.FileHandler(
                log_file, mode="a", encoding="utf-8"
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning(
                "Could not create log file %s: %s", log_file, exc
            )

    logger.debug(
        "Logger %r configured: level=%s  file=%s",
        name,
        level.upper(),
        log_file or "(none)",
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """Retrieve a logger by name (without reconfiguring it).

    Parameters
    ----------
    name:
        Logger name.

    Returns
    -------
    logging.Logger
    """
    return logging.getLogger(name)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_parent(path: str) -> None:
    """Create parent directories for *path* if they do not exist."""
    parent = Path(path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
