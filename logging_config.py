# File: mlb_betting_system/logging_config.py

import logging


def configure_logging(level=logging.INFO):
    """Configures basic logging for the betting system."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Silence chatty loggers if necessary
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("polars").setLevel(logging.INFO)
