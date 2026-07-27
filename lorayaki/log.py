"""Small logging setup shared by all commands."""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure root logger once and return the package logger."""
    global _CONFIGURED
    logger = logging.getLogger("lorayaki")
    if not _CONFIGURED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.propagate = False
        _CONFIGURED = True
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger("lorayaki")
