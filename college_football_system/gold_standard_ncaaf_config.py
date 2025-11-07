#!/usr/bin/env python3
"""
Gold Standard NCAA/College Football Configuration System
=========================================================

Advanced Pydantic-based configuration for NCAA betting system,
providing type safety, validation, and centralized settings management.
"""

import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class NCAABankrollConfig(BaseModel):
    """NCAA-specific bankroll configuration."""

    bankroll: float = Field(
        default=1000.0,
        ge=100,
        le=100000,
        description="Starting bankroll for NCAA betting",
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
    parlay_allocation: float = Field(
        default=0.05,
        ge=0.01,
        le=0.15,
        description="Percentage of bankroll for parlays",
    )


class NCAAThresholdsConfig(BaseModel):
    """NCAA-specific thresholds for edges and confidence."""

    min_edge_threshold: float = Field(
        default=0.03,
        ge=0.01,
        le=0.20,
        description="Minimum edge for consideration (lower than NFL due to more opportunities)",
    )
    holy_grail_threshold: float = Field(
        default=0.88,
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
        default=0.08,
        ge=0.01,
        le=0.20,
        description="Bonus for Group of Five games (higher than NFL)",
    )
    rivalry_bonus: float = Field(
        default=0.05,
        ge=0.01,
        le=0.15,
        description="Bonus for rivalry games",
    )


class NCAAPerformanceConfig(BaseModel):
    """NCAA-specific performance settings."""

    workers: int = Field(
        default=6,
        ge=1,
        le=16,
        description="Number of parallel workers (more games than NFL)",
    )
    batch_size: int = Field(
        default=25,
        ge=5,
        le=75,
        description="Batch size for processing (larger than NFL)",
    )
    cache_size: int = Field(
        default=1000,
        ge=100,
        le=5000,
        description="Cache size for performance (more teams than NFL)",
    )
    timeout_seconds: int = Field(
        default=30,
        ge=10,
        le=120,
        description="API timeout in seconds",
    )


class NCAAAlertConfig(BaseModel):
    """NCAA-specific alert settings."""

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
    rivalry_alerts: bool = Field(
        default=True,
        description="Alert for rivalry games",
    )
    upset_alerts: bool = Field(
        default=True,
        description="Alert for upset potential",
    )
    alert_frequency: str = Field(
        default="immediate",
        description="Alert frequency (immediate/daily/summary)",
    )


class NCAAAPIKeys(BaseModel):
    """NCAA API key management."""

    odds_api: Optional[str] = Field(
        default_factory=lambda: os.getenv("NCAA_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    anthropic: Optional[str] = Field(
        default_factory=lambda: os.getenv("NCAA_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )
    openai: Optional[str] = Field(
        default_factory=lambda: os.getenv("NCAA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    grok: Optional[str] = Field(
        default_factory=lambda: os.getenv("NCAA_GROK_API_KEY") or os.getenv("GROK_API_KEY")
    )
    weather: Optional[str] = Field(
        default_factory=lambda: os.getenv("NCAA_WEATHER_API_KEY") or os.getenv("WEATHER_API_KEY")
    )
    zephyr_base_url: Optional[str] = Field(
        default_factory=lambda: os.getenv("ZEPHYR_BASE_URL")
    )
    zephyr_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("ZEPHYR_API_KEY")
    )


class NCAAPatternWeights(BaseModel):
    """NCAA-specific pattern weights for games."""

    # Day of week weights
    tuesday: float = Field(
        default=0.850,
        ge=0.0,
        le=1.0,
        description="Tuesday MACtion weight",
    )
    wednesday: float = Field(
        default=0.830,
        ge=0.0,
        le=1.0,
        description="Wednesday MACtion weight",
    )
    thursday: float = Field(
        default=0.780,
        ge=0.0,
        le=1.0,
        description="Thursday night games weight",
    )
    friday: float = Field(
        default=0.750,
        ge=0.0,
        le=1.0,
        description="Friday night games weight",
    )
    saturday_early: float = Field(
        default=0.650,
        ge=0.0,
        le=1.0,
        description="Saturday early games weight",
    )
    saturday_afternoon: float = Field(
        default=0.620,
        ge=0.0,
        le=1.0,
        description="Saturday afternoon games weight",
    )
    saturday_night: float = Field(
        default=0.580,
        ge=0.0,
        le=1.0,
        description="Saturday night games weight",
    )

    # Situational weights
    weather_bonus: float = Field(
        default=0.120,
        ge=0.0,
        le=0.30,
        description="Weather impact bonus (higher than NFL)",
    )
    holy_grail_threshold: float = Field(
        default=0.880,
        ge=0.50,
        le=0.99,
        description="Holy Grail threshold",
    )
    obscure_bonus: float = Field(
        default=0.080,
        ge=0.01,
        le=0.20,
        description="Group of Five game bonus",
    )
    rivalry_bonus: float = Field(
        default=0.060,
        ge=0.01,
        le=0.15,
        description="Rivalry game bonus",
    )
    conference_championship_weight: float = Field(
        default=0.550,
        ge=0.0,
        le=1.0,
        description="Conference championship game weight",
    )
    bowl_game_weight: float = Field(
        default=0.620,
        ge=0.0,
        le=1.0,
        description="Bowl game weight",
    )


class NCAAConferenceClassifications(BaseModel):
    """NCAA conference classifications for weighting."""

    power_conferences: List[str] = Field(
        default_factory=lambda: ["SEC", "Big Ten", "Big 12", "ACC", "Pac-12"]
    )
    group_of_five: List[str] = Field(
        default_factory=lambda: ["American", "MAC", "Mountain West", "Sun Belt", "C-USA"]
    )
    hidden_gems: List[str] = Field(
        default_factory=lambda: ["MAC", "Sun Belt", "Mountain West"]
    )
    high_scoring: List[str] = Field(
        default_factory=lambda: ["Big 12", "Pac-12", "American"]
    )
    defensive: List[str] = Field(
        default_factory=lambda: ["SEC", "Big Ten"]
    )


class NCAAGoldStandardConfig(BaseModel):
    """Complete NCAA configuration system."""

    bankroll: NCAABankrollConfig = Field(default_factory=NCAABankrollConfig)
    thresholds: NCAAThresholdsConfig = Field(default_factory=NCAAThresholdsConfig)
    performance: NCAAPerformanceConfig = Field(default_factory=NCAAPerformanceConfig)
    alerts: NCAAAlertConfig = Field(default_factory=NCAAAlertConfig)
    api_keys: NCAAAPIKeys = Field(default_factory=NCAAAPIKeys)
    pattern_weights: NCAAPatternWeights = Field(default_factory=NCAAPatternWeights)
    conferences: NCAAConferenceClassifications = Field(
        default_factory=NCAAConferenceClassifications
    )

    class Config:
        validate_assignment = True

    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []

        # Check API keys
        if not self.api_keys.odds_api:
            issues.append("Missing ODDS_API_KEY - required for fetching odds data")

        # Check bankroll settings
        if self.bankroll.unit_size > self.bankroll.bankroll * 0.1:
            issues.append(
                f"Unit size (${self.bankroll.unit_size}) > 10% of bankroll "
                f"(${self.bankroll.bankroll}) - high risk"
            )

        # Check thresholds
        if self.thresholds.min_edge_threshold > 0.10:
            issues.append(
                f"Min edge threshold ({self.thresholds.min_edge_threshold:.1%}) "
                f"is very high - may miss opportunities"
            )

        # Check performance settings
        if self.performance.workers > 12:
            issues.append(
                f"High worker count ({self.performance.workers}) may cause rate limiting"
            )

        return issues


def get_ncaa_config() -> NCAAGoldStandardConfig:
    """Get NCAA configuration with environment variable support."""
    return NCAAGoldStandardConfig()


def setup_ncaa_environment() -> None:
    """Set up NCAA environment variables."""
    # Set defaults if not provided
    os.environ.setdefault("NCAA_ODDS_API_KEY", os.getenv("ODDS_API_KEY", ""))
    os.environ.setdefault("NCAA_BANKROLL", "1000.0")
    os.environ.setdefault("NCAA_UNIT_SIZE", "10.0")
    os.environ.setdefault("ZEPHYR_BASE_URL", "")
    os.environ.setdefault("ZEPHYR_API_KEY", "")
    os.environ.setdefault("ZEPHYR_MODEL", "zephyr-7b-beta")
    os.environ.setdefault("ZEPHYR_TIMEOUT", "20")
    os.environ.setdefault("ZEPHYR_TEMPERATURE", "0.25")
    os.environ.setdefault("ZEPHYR_MAX_TOKENS", "640")

    logger.info("✅ NCAA environment configured")


# Test function
def test_ncaa_config():
    """Test NCAA configuration system."""
    print("🧪 Testing Gold Standard NCAA Configuration...")

    config = get_ncaa_config()
    issues = config.validate_configuration()

    print("🏈 NCAA Configuration Summary:")
    print(f"   Bankroll: ${config.bankroll.bankroll}")
    print(f"   Unit Size: ${config.bankroll.unit_size}")
    print(f"   Min Edge: {config.thresholds.min_edge_threshold:.1%}")
    print(f"   Holy Grail: {config.thresholds.holy_grail_threshold:.1%}")
    print(f"   Workers: {config.performance.workers}")
    print(f"   Batch Size: {config.performance.batch_size}")
    print(
        f"   Alerts: {'Enabled' if config.alerts.enable_email_alerts else 'Disabled'}"
    )

    print("\n📊 Conference Classifications:")
    print(f"   Power 5: {', '.join(config.conferences.power_conferences)}")
    print(f"   Group of 5: {', '.join(config.conferences.group_of_five)}")
    print(f"   Hidden Gems: {', '.join(config.conferences.hidden_gems)}")

    if issues:
        print("\n⚠️  Configuration Issues:")
        for issue in issues:
            print(f"   • {issue}")
    else:
        print("\n✅ No configuration issues detected")

    print("\n✅ NCAA configuration test complete!")
    return config


if __name__ == "__main__":
    test_ncaa_config()
