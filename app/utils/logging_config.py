import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Optional


class JSONFormatter(logging.Formatter):
    """
    Custom JSON formatter for structured logging
    """

    def format(self, record):
        try:
            # Get the log message and ensure it's JSON-safe
            message = record.getMessage()

            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": message,
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
            }

            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.formatException(record.exc_info)

            # Add extra fields if present
            if hasattr(record, "extra_fields") and getattr(
                record, "extra_fields", None
            ):
                log_entry.update(getattr(record, "extra_fields"))

            return json.dumps(log_entry, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as e:
            # Fallback to simple format if JSON serialization fails
            fallback_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": str(record.getMessage()).replace('"', '\\"'),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "json_error": str(e),
            }
            return json.dumps(fallback_entry, ensure_ascii=False, default=str)


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

    # Add file handler for persistent logging
    try:
        # Create logs directory if it doesn't exist
        log_dir = "/app/logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Test file creation first
        test_file = os.path.join(log_dir, "test_write.log")
        with open(test_file, 'w') as f:
            f.write("test\n")
        os.remove(test_file)

        # Import and create rotating file handler
        from logging.handlers import RotatingFileHandler

        log_file_path = os.path.join(log_dir, "app.log")
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Use JSON formatter for file as well
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(level)
        
        # Add to logger
        logger.addHandler(file_handler)

        # Force a log message to create the file
        logger.info("File logging enabled: /app/logs/app.log")
        
        # Verify file was created
        if os.path.exists(log_file_path):
            logger.info(f"Log file confirmed created: {log_file_path}")
        else:
            logger.error(f"Log file was not created: {log_file_path}")
            
    except ImportError as e:
        print(f"IMPORT ERROR: Could not import RotatingFileHandler: {e}")
        logger.error(f"Could not import RotatingFileHandler: {e}")
    except PermissionError as e:
        print(f"PERMISSION ERROR: {e}")
        logger.error(f"Permission error setting up file logging: {e}")
    except Exception as e:
        print(f"GENERAL ERROR: {e}")
        logger.error(f"Could not setup file logging: {e}")
        import traceback
        print(f"TRACEBACK: {traceback.format_exc()}")
        logger.error(f"Full traceback: {traceback.format_exc()}")

    # Log the initial setup
    logger.info(
        "Logging system initialized with level: %s", logging._levelToName[level]
    )
