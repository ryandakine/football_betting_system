import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


def get_env_var(var_name: str, default: Any, cast_type: Callable[[Any], Any] = str) -> Any:
    """Safely retrieve environment variable with type casting."""
    value = os.getenv(var_name, default)
    try:
        return cast_type(value)
    except ValueError as e:
        logger.error(f"Invalid environment variable {var_name}: {e}. Using default: {default}")
        return cast_type(default)


@dataclass
class SystemConfig:
    """Configuration for the MLB betting system."""

    data_dir: str = field(default_factory=lambda: get_env_var("DATA_DIR", "data"))
    output_dir: str = field(default_factory=lambda: get_env_var("OUTPUT_DIR", "output"))
    model_dir: str = field(default_factory=lambda: get_env_var("MODEL_DIR", "models"))
    odds_api_key: str = field(
        default_factory=lambda: get_env_var("ODDS_API_KEY", "YOUR_ODDS_API_KEY")
    )
    min_confidence: float = field(default_factory=lambda: get_env_var("MIN_CONFIDENCE", 0.5, float))
    bankroll: float = field(default_factory=lambda: get_env_var("BANKROLL", 1000.0, float))
    max_retries: int = field(default_factory=lambda: get_env_var("MAX_RETRIES", 3, int))
    api_timeout_seconds: int = field(
        default_factory=lambda: get_env_var("API_TIMEOUT_SECONDS", 15, int)
    )
    retrain_interval_days: int = field(
        default_factory=lambda: get_env_var("RETRAIN_INTERVAL_DAYS", 7, int)
    )
    test_mode: bool = field(default_factory=lambda: get_env_var("TEST_MODE", "False", bool))

    def validate(self) -> None:
        """Validate configuration settings."""
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory {self.data_dir} does not exist. Creating it.")
            os.makedirs(self.data_dir, exist_ok=True)
        if not os.path.exists(self.output_dir):
            logger.warning(f"Output directory {self.output_dir} does not exist. Creating it.")
            os.makedirs(self.output_dir, exist_ok=True)
        if not os.path.exists(self.model_dir):
            logger.warning(f"Model directory {self.model_dir} does not exist. Creating it.")
            os.makedirs(self.model_dir, exist_ok=True)
        if self.odds_api_key == "YOUR_ODDS_API_KEY":
            logger.warning("Odds API key not set. Configure ODDS_API_KEY environment variable.")
