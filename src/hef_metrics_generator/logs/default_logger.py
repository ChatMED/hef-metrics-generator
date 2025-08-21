"""
Default logging configuration for hef_metrics_generator.

This module provides a helper to configure consistent logging across the package.
It ensures all logs follow the same format and level.
"""

import logging


def configure_logging(level: int = logging.INFO, force: bool = False) -> None:
    """
    Configure the root logger with a standard format.

    Args:
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
                     Defaults to logging.INFO.
        force (bool): If True, clear existing handlers even if already configured.
                      Defaults to False.
    """
    root_logger = logging.getLogger()

    if root_logger.hasHandlers():
        if not force:
            return
        root_logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    ))

    root_logger.setLevel(level)
    root_logger.addHandler(handler)
