"""Logging configuration for the esperanto package."""
import logging
import sys


def setup_logging(level=logging.INFO):
    """Configure logging for the esperanto package.
    
    Args:
        level: The logging level to use. Defaults to INFO.
    """
    logger = logging.getLogger("esperanto")
    logger.setLevel(level)

    # Create console handler with a higher log level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    # Create formatter and add it to the handler
    formatter = logging.Formatter(
        "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(console_handler)

    return logger


# Create and configure the default logger
logger = setup_logging()
