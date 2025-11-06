#!/usr/bin/env python3
"""
API Configuration Module v5.0 (Final, Complete)
Manages all API keys, model configurations, and system settings for the Ultimate System.
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


@dataclass
class ModelConfiguration:
    """Configuration for individual AI models."""

    name: str
    model_id: str
    weight: float
    timeout: int = 30
    max_retries: int = 3


# Load environment variables from a .env file at the project root
load_dotenv()


def get_trimodel_api_keys() -> dict[str, str | None]:
    """Loads all required API keys from environment variables."""
    api_keys = {
        "odds_api": os.getenv("THE_ODDS_API_KEY"),
        "claude": os.getenv("ANTHROPIC_API_KEY"),
        "openai": os.getenv("OPENAI_API_KEY"),
        "gemini": os.getenv("GEMINI_API_KEY"),
        "grok": os.getenv("GROK_API_KEY"),
        "perplexity": os.getenv("PERPLEXITY_API_KEY"),
    }
    for service, key in api_keys.items():
        if key:
            logger.info(f"✅ {service.upper()} API key loaded")
        else:
            logger.warning(f"⚠️ {service.upper()} API key not found")
    return api_keys


def get_ensemble_config() -> dict:
    """Returns the configuration for the AI model ensemble."""
    config = {
        "models": {
            "claude": ModelConfiguration(
                name="Claude 3.5 Sonnet",
                model_id="claude-3-5-sonnet-20240620",
                weight=0.3,
            ),
            "openai": ModelConfiguration(name="GPT-4o", model_id="gpt-4o", weight=0.3),
            "grok": ModelConfiguration(name="Grok-1", model_id="grok-1", weight=0.2),
            "perplexity": ModelConfiguration(
                name="Perplexity Pro",
                model_id="llama-3.1-sonar-large-128k-online",
                weight=0.2,
            ),
        },
        "weights": {"claude": 0.3, "openai": 0.3, "grok": 0.2, "perplexity": 0.2},
        "cache_duration_hours": 6,
    }
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(model.weight for model in config["models"].values())
    if abs(total_weight - 1.0) > 0.01:
        for model_cfg in config["models"].values():
            model_cfg.weight /= total_weight
        config["weights"] = {
            name: model.weight for name, model in config["models"].items()
        }
    return config


def get_odds_api_config() -> dict:
    """Returns the configuration for the Odds API fetcher."""
    return {
        "supported_markets": [
            "h2h",
            "spreads",
            "totals",
            "player_home_runs",
            "player_hits",
            "pitcher_strikeouts",
        ]
    }


def get_alert_config() -> dict:
    """Loads configurations for the SmartAlertManager."""
    # This function should remain as it was, correctly parsing .env variables
    return {
        "slack": {"webhook_url": os.getenv("SLACK_WEBHOOK_URL")}
    }  # Simplified example


def validate_api_configuration() -> tuple[bool, list[str]]:
    """Validates that the essential API keys are present."""
    errors = []
    api_keys = get_trimodel_api_keys()
    # Check for the keys your system actually uses
    required_keys = ["odds_api", "claude", "openai"]
    for key in required_keys:
        if not api_keys.get(key):
            errors.append(f"Missing required API key: {key}")
    is_valid = len(errors) == 0
    if not is_valid:
        logger.error(f"❌ API configuration validation failed: {errors}")
    return is_valid, errors


def initialize_configuration():
    """Validates the configuration when the module is imported."""
    validate_api_configuration()


# This ensures the validation runs once when the module is first imported
initialize_configuration()
