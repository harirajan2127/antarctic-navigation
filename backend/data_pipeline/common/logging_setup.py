"""Logging configuration for pipeline scripts."""
from __future__ import annotations

import logging
import sys

_LOGGER_INITIALIZED = False


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a configured logger for a pipeline module.

    Pipe logs to stdout with a simple human-readable format. Configured
    once for the whole process.
    """
    global _LOGGER_INITIALIZED
    logger = logging.getLogger(name)
    if not _LOGGER_INITIALIZED:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
        )
        root = logging.getLogger("data_pipeline")
        root.setLevel(level)
        root.addHandler(handler)
        # Prevent duplicate handlers on repeated imports.
        root.propagate = False
        _LOGGER_INITIALIZED = True
        logger = logging.getLogger(name)
        logger.propagate = False
    return logger