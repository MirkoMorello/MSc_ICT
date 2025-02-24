import logging
from ..config import LOGGING_LEVEL


def get_logger(name):
    """
    Initializes a logger with pre-set formatting.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # prevent duplicate handlers
        logger.setLevel(LOGGING_LEVEL)
        formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S"
        )
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger