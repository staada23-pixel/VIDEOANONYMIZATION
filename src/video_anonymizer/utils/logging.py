"""Logging setup pro video_anonymizer."""
import logging
import sys


def setup_logging(level="INFO"):
    """Nastaví root logger s konzistentním formátem."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )
    return logging.getLogger("video_anonymizer")


def get_logger(name):
    return logging.getLogger(f"video_anonymizer.{name}")
