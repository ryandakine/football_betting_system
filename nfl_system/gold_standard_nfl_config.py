#!/usr/bin/env python3
"""
Gold Standard NFL Configuration System
====================================

Advanced Pydantic-based configuration for NFL betting system,
matching college system sophistication with NFL-specific settings.
"""

import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NFLBankrollConfig(BaseModel):
    """NFL-specific bankroll configuration."""

    bankroll: float = Field(
        default=1000.0,
        ge=100,
        le=100000,
        description="Starting bankroll for NFL betting",
    )
    unit_size: float = Field(
        default=10.0,
        ge=1,
        le=500,
        description="Base unit size for bets",
    )
    max_exposure: float = Field(
        default=0.12,
        ge=0.01,
        le=0.25,
        description="Maximum exposure per game",
    )
    kelly_criterion: bool = Field(
        default=True,
        description="Use Kelly Criterion for stake sizing",
    )


class NFLThresholdsConfig(BaseModel):
    """NFL-specific thresholds for edges and confidence."""

    min_edge_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum edge for consideration",
    )
    holy_grail_threshold: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Holy Grail game threshold",
    )
    confidence_threshold: float = Field(
        default=0.60,
        ge=0.50,
        le=0.95,
        description="Minimum confidence for bets",
    )
    obscure_bonus_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.15,
        description="Bonus for obscure games",
    )


class NFLPerformanceConfig(BaseModel):
    """NFL-specific performance settings."""

    workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers",
    )
    batch_size: int = Field(
        default=15,
        ge=5,
        le=50,
        description="Batch size for processing",
    )
    cache_size: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Cache size for performance",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=10,
        le=120,
        description="API timeout in seconds",
    )


class NFLAlertConfig(BaseModel):
    """NFL-specific alert settings."""

    enable_email_alerts: bool = Field(
        default=True,
        description="Enable email notifications",
    )
    enable_sms_alerts: bool = Field(
        default=False,
        description="Enable SMS notifications",
    )
    holy_grail_alerts: bool = Field(
        default=True,
        description="Alert for holy grail games",
    )
    edge_drop_alerts: bool = Field(
        default=True,
        description="Alert for edge drops",
    )
    alert_frequency: str = Field(
        default="immediate",
        description="Alert frequency (immediate/daily/summary)",
    )


class NFLAPIKeys(BaseModel):
    """NFL API key management."""

    odds_api: Optional[str] = Field(
        default_factory=lambda: os.getenv("NFL_ODDS_API_KEY")
    )
    anthropic: Optional[str] = Field(
        default_factory=lambda: os.getenv("NFL_ANTHROPIC_API_KEY")
    )
    openai: Optional[str] = Field(
        default_factory=lambda: os.getenv("NFL_OPENAI_API_KEY")
    )
    grok: Optional[str] = Field(
        default_factory=lambda: os.getenv("NFL_GROK_API_KEY")
    )
    weather: Optional[str] = Field(
        default_factory=lambda: os.getenv("NFL_WEATHER_API_KEY")
    )


class NFLPatternWeights(BaseModel):
    """NFL-specific pattern weights for games."""

    thursday: float = Field(
        default=0.720,
        ge=0.0,
        le=1.0,
        description="Thursday Night Football weight",
    )
    sunday_early: float = Field(
        default=0.650,
        ge=0.0,
        le=1.0,
        description="Sunday early games weight",
    )
    sunday_late: float = Field(
        default=0.620,
        ge=0.0,
        le=1.0,
        description="Sunday late games weight",
    )
    sunday_night: float = Field(
        default=0.550,
        ge=0.0,
        le=1.0,
        description="Sunday Night Football weight",
    )
    monday: float = Field(
        default=0.580,
        ge=0.0,
        le=1.0,
        description="Monday Night Football weight",
    )
    weather_bonus: float = Field(
        default=0.080,
        ge=0.0,
        le=0.20,
        description="Weather impact bonus",
    )
    holy_grail_threshold: float = Field(
        default=0.900,
        ge=0.50,
        le=0.99,
        description="Holy Grail threshold",
    )
    obscure_bonus: float = Field(
        default=0.050,
        ge=0.01,
        le=0.15,
        description="Obscure game bonus",
    )


class NFLConferenceClassifications(BaseModel):
    """NFL conference classifications for weighting."""

    power_conferences: List[str] = Field(default_factory=lambda: ["AFC", "NFC"])
    obscure_conferences: List[str] = Field(
        default_factory=lambda: ["AFC North", "NFC South"]
    )
    hidden_gems: List[str] = Field(
        default_factory=lambda: ["AFC East", "NFC West"]
    )


class NFLGoldStandardConfig(BaseModel):
    """Complete NFL configuration system."""

    bankroll: NFLBankrollConfig = Field(default_factory=NFLBankrollConfig)
    thresholds: NFLThresholdsConfig = Field(default_factory=NFLThresholdsConfig)
    performance: NFLPerformanceConfig = Field(default_factory=NFLPerformanceConfig)
    alerts: NFLAlertConfig = Field(default_factory=NFLAlertConfig)
    api_keys: NFLAPIKeys = Field(default_factory=NFLAPIKeys)
    pattern_weights: NFLPatternWeights = Field(default_factory=NFLPatternWeights)
    conferences: NFLConferenceClassifications = Field(
        default_factory=NFLConferenceClassifications
    )

    class Config:
        validate_assignment = True


def get_nfl_config() -> NFLGoldStandardConfig:
    """Get NFL configuration with environment variable support."""
    return NFLGoldStandardConfig()


def setup_nfl_environment() -> None:
    """Set up NFL environment variables."""
    # Set defaults if not provided
    os.environ.setdefault("NFL_ODDS_API_KEY", "")
    os.environ.setdefault("NFL_BANKROLL", "1000.0")
    os.environ.setdefault("NFL_UNIT_SIZE", "10.0")
    os.environ.setdefault("ZEPHYR_BASE_URL", "")
    os.environ.setdefault("ZEPHYR_API_KEY", "")
    os.environ.setdefault("ZEPHYR_MODEL", "zephyr-7b-beta")
    os.environ.setdefault("ZEPHYR_TIMEOUT", "20")
    os.environ.setdefault("ZEPHYR_TEMPERATURE", "0.25")
    os.environ.setdefault("ZEPHYR_MAX_TOKENS", "640")


# Test function
def test_nfl_config():
    """Test NFL configuration system."""
    print("🧪 Testing Gold Standard NFL Configuration...")

    config = get_nfl_config()
    _issues = (
        config.validate_configuration()
        if hasattr(config, "validate_configuration")
        else []
    )

    print("🏈 NFL Configuration Summary:")
    print(f"   Bankroll: ${config.bankroll.bankroll}")
    print(f"   Unit Size: ${config.bankroll.unit_size}")
    print(f"   Min Edge: {config.thresholds.min_edge_threshold:.1%}")
    print(f"   Holy Grail: {config.thresholds.holy_grail_threshold:.1%}")
    print(f"   Workers: {config.performance.workers}")
    print(
        f"   Alerts: {'Enabled' if config.alerts.enable_email_alerts else 'Disabled'}"
    )

    print("✅ NFL configuration test complete!")


if __name__ == "__main__":
    test_nfl_config()
