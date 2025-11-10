#!/usr/bin/env python3
"""
WNBA Configuration System
========================

Pydantic-based configuration for WNBA betting system.
"""

import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class WNBABankrollConfig(BaseModel):
    """WNBA-specific bankroll configuration."""

    bankroll: float = Field(
        default=1000.0,
        ge=100,
        le=100000,
        description="Starting bankroll for WNBA betting",
    )
    unit_size: float = Field(
        default=10.0,
        ge=1,
        le=500,
        description="Base unit size for bets",
    )
    max_exposure: float = Field(
        default=0.08,
        ge=0.01,
        le=0.20,
        description="Maximum exposure per game (conservative for WNBA)",
    )
    kelly_criterion: bool = Field(
        default=True,
        description="Use Kelly Criterion for stake sizing",
    )


class WNBAThresholdsConfig(BaseModel):
    """WNBA-specific thresholds for edges and confidence."""

    min_edge_threshold: float = Field(
        default=0.06,
        ge=0.01,
        le=0.20,
        description="Minimum edge for consideration (higher than college)",
    )
    holy_grail_threshold: float = Field(
        default=0.90,
        ge=0.50,
        le=0.99,
        description="Holy Grail game threshold",
    )
    confidence_threshold: float = Field(
        default=0.62,
        ge=0.50,
        le=0.95,
        description="Minimum confidence for bets",
    )


class WNBAPerformanceConfig(BaseModel):
    """WNBA-specific performance settings."""

    workers: int = Field(
        default=3,
        ge=1,
        le=8,
        description="Number of parallel workers (fewer games)",
    )
    batch_size: int = Field(
        default=6,
        ge=3,
        le=15,
        description="Batch size for processing (smaller slates)",
    )
    cache_size: int = Field(
        default=300,
        ge=100,
        le=1000,
        description="Cache size for performance",
    )
    timeout_seconds: int = Field(
        default=25,
        ge=10,
        le=120,
        description="API timeout in seconds",
    )


class WNBAAlertConfig(BaseModel):
    """WNBA-specific alert settings."""

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
    player_news_alerts: bool = Field(
        default=True,
        description="Alert for player injury/news",
    )
    alert_frequency: str = Field(
        default="immediate",
        description="Alert frequency (immediate/daily/summary)",
    )


class WNBAAPIKeys(BaseModel):
    """WNBA API key management."""

    odds_api: Optional[str] = Field(
        default_factory=lambda: os.getenv("WNBA_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    anthropic: Optional[str] = Field(
        default_factory=lambda: os.getenv("WNBA_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )
    openai: Optional[str] = Field(
        default_factory=lambda: os.getenv("WNBA_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    grok: Optional[str] = Field(
        default_factory=lambda: os.getenv("WNBA_GROK_API_KEY") or os.getenv("GROK_API_KEY")
    )


class WNBAPatternWeights(BaseModel):
    """WNBA-specific pattern weights for games."""

    weekday: float = Field(
        default=0.68,
        ge=0.0,
        le=1.0,
        description="Weekday game weight",
    )
    weekend: float = Field(
        default=0.78,
        ge=0.0,
        le=1.0,
        description="Weekend game weight (higher attendance)",
    )
    playoffs: float = Field(
        default=0.92,
        ge=0.0,
        le=1.0,
        description="Playoff game weight",
    )
    commissioners_cup: float = Field(
        default=0.82,
        ge=0.0,
        le=1.0,
        description="Commissioner's Cup game weight",
    )
    rivalry: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Rivalry game weight",
    )
    holy_grail_threshold: float = Field(
        default=0.900,
        ge=0.50,
        le=0.99,
        description="Holy Grail threshold",
    )


class WNBATeamClassifications(BaseModel):
    """WNBA team classifications for weighting."""

    elite_teams: List[str] = Field(
        default_factory=lambda: ["Las Vegas Aces", "New York Liberty", "Minnesota Lynx", "Seattle Storm"]
    )
    playoff_teams: List[str] = Field(
        default_factory=lambda: ["Connecticut Sun", "Phoenix Mercury", "Chicago Sky", "Washington Mystics"]
    )
    developing_teams: List[str] = Field(
        default_factory=lambda: ["Dallas Wings", "Atlanta Dream", "Indiana Fever", "Los Angeles Sparks"]
    )


class WNBAGoldStandardConfig(BaseModel):
    """Complete WNBA configuration system."""

    bankroll: WNBABankrollConfig = Field(default_factory=WNBABankrollConfig)
    thresholds: WNBAThresholdsConfig = Field(default_factory=WNBAThresholdsConfig)
    performance: WNBAPerformanceConfig = Field(default_factory=WNBAPerformanceConfig)
    alerts: WNBAAlertConfig = Field(default_factory=WNBAAlertConfig)
    api_keys: WNBAAPIKeys = Field(default_factory=WNBAAPIKeys)
    pattern_weights: WNBAPatternWeights = Field(default_factory=WNBAPatternWeights)
    teams: WNBATeamClassifications = Field(default_factory=WNBATeamClassifications)

    class Config:
        validate_assignment = True


def get_wnba_config() -> WNBAGoldStandardConfig:
    """Get WNBA configuration with environment variable support."""
    return WNBAGoldStandardConfig()


def setup_wnba_environment() -> None:
    """Set up WNBA environment variables."""
    # Set defaults if not provided
    os.environ.setdefault("WNBA_ODDS_API_KEY", "")
    os.environ.setdefault("WNBA_BANKROLL", "1000.0")
    os.environ.setdefault("WNBA_UNIT_SIZE", "10.0")


# Test function
def test_wnba_config():
    """Test WNBA configuration system."""
    print("🧪 Testing WNBA Configuration...")

    config = get_wnba_config()

    print("🏀 WNBA Configuration Summary:")
    print(f"   Bankroll: ${config.bankroll.bankroll}")
    print(f"   Unit Size: ${config.bankroll.unit_size}")
    print(f"   Min Edge: {config.thresholds.min_edge_threshold:.1%}")
    print(f"   Holy Grail: {config.thresholds.holy_grail_threshold:.1%}")
    print(f"   Workers: {config.performance.workers}")
    print(f"   Batch Size: {config.performance.batch_size}")
    print(
        f"   Alerts: {'Enabled' if config.alerts.enable_email_alerts else 'Disabled'}"
    )

    print("✅ WNBA configuration test complete!")


if __name__ == "__main__":
    test_wnba_config()
