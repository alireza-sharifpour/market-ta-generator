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
    # First log that we're attempting file logging setup
    logger.info("Attempting to set up persistent file logging...")
    
    try:
        # Create logs directory if it doesn't exist
        log_dir = "/app/logs"
        logger.info(f"Creating logs directory: {log_dir}")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"Logs directory created successfully: {log_dir}")
        
        # Test file creation first
        test_file = os.path.join(log_dir, "test_write.log")
        logger.info(f"Testing file creation: {test_file}")
        with open(test_file, 'w') as f:
            f.write("test\n")
        os.remove(test_file)
        logger.info("File creation test successful")

        # Import and create rotating file handler
        logger.info("Importing RotatingFileHandler...")
        from logging.handlers import RotatingFileHandler
        logger.info("RotatingFileHandler imported successfully")

        log_file_path = os.path.join(log_dir, "app.log")
        logger.info(f"Creating RotatingFileHandler for: {log_file_path}")
        # Get log file settings from environment or use defaults
        max_bytes = int(os.getenv("LOG_FILE_MAX_MB", "100")) * 1024 * 1024  # Default 100MB
        backup_count = int(os.getenv("LOG_FILE_BACKUP_COUNT", "10"))  # Default 10 backups
        
        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        logger.info("RotatingFileHandler created successfully")
        
        # Use JSON formatter for file as well
        file_handler.setFormatter(json_formatter)
        file_handler.setLevel(level)
        logger.info("File handler formatter and level set")
        
        # Add to logger
        logger.addHandler(file_handler)
        logger.info("File handler added to logger")

        # Force a log message to create the file
        logger.info("File logging enabled: /app/logs/app.log")
        
        # Verify file was created
        if os.path.exists(log_file_path):
            logger.info(f"Log file confirmed created: {log_file_path}")
        else:
            logger.error(f"Log file was not created: {log_file_path}")
            
    except ImportError as e:
        error_msg = f"Could not import RotatingFileHandler: {e}"
        print(f"IMPORT ERROR: {error_msg}")
        logger.error(error_msg)
    except PermissionError as e:
        error_msg = f"Permission error setting up file logging: {e}"
        print(f"PERMISSION ERROR: {error_msg}")
        logger.error(error_msg)
    except Exception as e:
        error_msg = f"Could not setup file logging: {e}"
        print(f"GENERAL ERROR: {error_msg}")
        logger.error(error_msg)
        import traceback
        traceback_msg = traceback.format_exc()
        print(f"TRACEBACK: {traceback_msg}")
        logger.error(f"Full traceback: {traceback_msg}")

    # Log the initial setup
    logger.info(
        "Logging system initialized with level: %s", logging._levelToName[level]
    )
