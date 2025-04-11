import logging
import sys
from typing import Optional


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Optional string representing the logging level.
                  Defaults to 'INFO' if not provided.
    """
    # Set default log level if not provided
    level = getattr(logging, (log_level or "INFO").upper())

    # Create a formatter that includes timestamp, level, and message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    # Add the console handler
    logger.addHandler(console_handler)

    # Log the initial setup
    logger.info(
        "Logging system initialized with level: %s", logging.getLevelName(level)
    )
