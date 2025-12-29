"""Centralized logging configuration for WheatVision2."""

import logging
import sys
from pathlib import Path
from typing import Optional

_LOGGER_NAME = "wheatvision"
_initialized = False


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Set up the WheatVision2 logging system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional file path for logging output.
        console: Whether to also log to console.
        
    Returns:
        Configured logger instance.
    """
    global _initialized
    
    logger = logging.getLogger(_LOGGER_NAME)
    
    if _initialized:
        return logger
    
    logger.setLevel(level)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    _initialized = True
    logger.info(f"WheatVision2 logging initialized (level={logging.getLevelName(level)})")
    
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for a specific module.
    
    Args:
        name: Optional submodule name (e.g., "engines.sam").
              If None, returns the root WheatVision logger.
              
    Returns:
        Logger instance for the specified module.
    """
    global _initialized
    
    if not _initialized:
        setup_logging()
    
    if name:
        return logging.getLogger(f"{_LOGGER_NAME}.{name}")
    return logging.getLogger(_LOGGER_NAME)


class LogContext:
    """Context manager for logging operation start/end with timing."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.INFO):
        """
        Initialize the logging context.
        
        Args:
            logger: Logger instance to use.
            operation: Description of the operation being performed.
            level: Logging level for messages.
        """
        self._logger = logger
        self._operation = operation
        self._level = level
        self._start_time: float = 0.0

    def __enter__(self) -> "LogContext":
        """Log operation start."""
        import time
        self._start_time = time.perf_counter()
        self._logger.log(self._level, f"Starting: {self._operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Log operation end with timing."""
        import time
        elapsed_ms = (time.perf_counter() - self._start_time) * 1000
        
        if exc_type is not None:
            self._logger.error(f"Failed: {self._operation} ({elapsed_ms:.1f}ms) - {exc_val}")
        else:
            self._logger.log(self._level, f"Completed: {self._operation} ({elapsed_ms:.1f}ms)")


def log_progress(
    logger: logging.Logger,
    current: int,
    total: int,
    operation: str,
    interval: int = 1,
) -> None:
    """
    Log progress for iterative operations.
    
    Args:
        logger: Logger instance.
        current: Current iteration (1-indexed).
        total: Total iterations.
        operation: Description of the operation.
        interval: Log every N iterations.
    """
    if current % interval == 0 or current == total:
        percentage = (current / total) * 100
        logger.info(f"{operation}: {current}/{total} ({percentage:.1f}%)")
