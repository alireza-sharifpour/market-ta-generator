import json
import logging
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        # Add extra fields if present
        if hasattr(record, 'extra_fields') and getattr(record, 'extra_fields', None):
            log_entry.update(getattr(record, 'extra_fields'))
            
        return json.dumps(log_entry)


def setup_logging(log_level: Optional[str] = None) -> None:
    """
    Configure logging for the application.

    Args:
        log_level: Optional string representing the logging level.
                  Defaults to 'INFO' if not provided.
    """
    # Set default log level if not provided
    level = getattr(logging, (log_level or "INFO").upper())

    # Create JSON formatter for structured logging
    json_formatter = JSONFormatter()
    
    # Create standard formatter for fallback
    standard_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    # Use JSON formatter in production, standard in development
    import os
    if os.getenv("ENVIRONMENT") == "production":
        console_handler.setFormatter(json_formatter)
    else:
        console_handler.setFormatter(standard_formatter)

    # Get the root logger
    logger = logging.getLogger()
    logger.setLevel(level)

    # Remove any existing handlers to avoid duplicates
    logger.handlers = []

    # Add the console handler
    logger.addHandler(console_handler)

    # Log the initial setup
    logger.info(
        "Logging system initialized with level: %s", logging._levelToName[level]
    )
