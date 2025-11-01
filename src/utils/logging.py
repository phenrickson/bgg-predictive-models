import logging
import warnings
from pathlib import Path
from typing import Optional


def filter_lightgbm_warnings(record):
    """Filter out specific LightGBM warnings."""
    return not (
        record.levelname == "WARNING"
        and "No further splits with positive gain" in str(record.msg)
    )


def setup_logging(log_file: Optional[Path] = None) -> logging.Logger:
    """Configure logging for the training process."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(__name__)

    # Filter LightGBM warnings
    logging.getLogger("lightgbm").addFilter(filter_lightgbm_warnings)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        )
        logger.addHandler(file_handler)

    return logger
