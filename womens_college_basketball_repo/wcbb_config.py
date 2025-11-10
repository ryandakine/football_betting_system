#!/usr/bin/env python3
"""
Women's College Basketball Configuration System
==============================================

Pydantic-based configuration for WCBB betting system.
"""

import os
import logging
from typing import List, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


class WCBBBankrollConfig(BaseModel):
    """WCBB-specific bankroll configuration."""

    bankroll: float = Field(
        default=1000.0,
        ge=100,
        le=100000,
        description="Starting bankroll for WCBB betting",
    )
    unit_size: float = Field(
        default=10.0,
        ge=1,
        le=500,
        description="Base unit size for bets",
    )
    max_exposure: float = Field(
        default=0.10,
        ge=0.01,
        le=0.25,
        description="Maximum exposure per game",
    )
    kelly_criterion: bool = Field(
        default=True,
        description="Use Kelly Criterion for stake sizing",
    )


class WCBBThresholdsConfig(BaseModel):
    """WCBB-specific thresholds for edges and confidence."""

    min_edge_threshold: float = Field(
        default=0.05,
        ge=0.01,
        le=0.20,
        description="Minimum edge for consideration",
    )
    holy_grail_threshold: float = Field(
        default=0.88,
        ge=0.50,
        le=0.99,
        description="Holy Grail game threshold",
    )
    confidence_threshold: float = Field(
        default=0.58,
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


class WCBBPerformanceConfig(BaseModel):
    """WCBB-specific performance settings."""

    workers: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Number of parallel workers",
    )
    batch_size: int = Field(
        default=12,
        ge=5,
        le=50,
        description="Batch size for processing",
    )
    cache_size: int = Field(
        default=400,
        ge=100,
        le=2000,
        description="Cache size for performance",
    )
    timeout_seconds: int = Field(
        default=25,
        ge=10,
        le=120,
        description="API timeout in seconds",
    )


class WCBBAlertConfig(BaseModel):
    """WCBB-specific alert settings."""

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


class WCBBAPIKeys(BaseModel):
    """WCBB API key management."""

    odds_api: Optional[str] = Field(
        default_factory=lambda: os.getenv("WCBB_ODDS_API_KEY") or os.getenv("ODDS_API_KEY")
    )
    anthropic: Optional[str] = Field(
        default_factory=lambda: os.getenv("WCBB_ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    )
    openai: Optional[str] = Field(
        default_factory=lambda: os.getenv("WCBB_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    )
    grok: Optional[str] = Field(
        default_factory=lambda: os.getenv("WCBB_GROK_API_KEY") or os.getenv("GROK_API_KEY")
    )


class WCBBPatternWeights(BaseModel):
    """WCBB-specific pattern weights for games."""

    weekday: float = Field(
        default=0.65,
        ge=0.0,
        le=1.0,
        description="Weekday game weight",
    )
    weekend: float = Field(
        default=0.72,
        ge=0.0,
        le=1.0,
        description="Weekend game weight",
    )
    tournament: float = Field(
        default=0.88,
        ge=0.0,
        le=1.0,
        description="Tournament game weight",
    )
    rivalry: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Rivalry game weight",
    )
    holy_grail_threshold: float = Field(
        default=0.880,
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


class WCBBConferenceClassifications(BaseModel):
    """WCBB conference classifications for weighting."""

    power_conferences: List[str] = Field(
        default_factory=lambda: ["Big Ten", "SEC", "ACC", "Pac-12", "Big 12", "Big East"]
    )
    mid_major_conferences: List[str] = Field(
        default_factory=lambda: ["American", "Atlantic 10", "WCC", "Mountain West"]
    )
    hidden_gems: List[str] = Field(
        default_factory=lambda: ["Summit League", "Horizon League", "MAC"]
    )


class WCBBGoldStandardConfig(BaseModel):
    """Complete WCBB configuration system."""

    bankroll: WCBBBankrollConfig = Field(default_factory=WCBBBankrollConfig)
    thresholds: WCBBThresholdsConfig = Field(default_factory=WCBBThresholdsConfig)
    performance: WCBBPerformanceConfig = Field(default_factory=WCBBPerformanceConfig)
    alerts: WCBBAlertConfig = Field(default_factory=WCBBAlertConfig)
    api_keys: WCBBAPIKeys = Field(default_factory=WCBBAPIKeys)
    pattern_weights: WCBBPatternWeights = Field(default_factory=WCBBPatternWeights)
    conferences: WCBBConferenceClassifications = Field(
        default_factory=WCBBConferenceClassifications
    )

    class Config:
        validate_assignment = True


def get_wcbb_config() -> WCBBGoldStandardConfig:
    """Get WCBB configuration with environment variable support."""
    return WCBBGoldStandardConfig()


def setup_wcbb_environment() -> None:
    """Set up WCBB environment variables."""
    # Set defaults if not provided
    os.environ.setdefault("WCBB_ODDS_API_KEY", "")
    os.environ.setdefault("WCBB_BANKROLL", "1000.0")
    os.environ.setdefault("WCBB_UNIT_SIZE", "10.0")


# Test function
def test_wcbb_config():
    """Test WCBB configuration system."""
    print("🧪 Testing Women's College Basketball Configuration...")

    config = get_wcbb_config()

    print("🏀 WCBB Configuration Summary:")
    print(f"   Bankroll: ${config.bankroll.bankroll}")
    print(f"   Unit Size: ${config.bankroll.unit_size}")
    print(f"   Min Edge: {config.thresholds.min_edge_threshold:.1%}")
    print(f"   Holy Grail: {config.thresholds.holy_grail_threshold:.1%}")
    print(f"   Workers: {config.performance.workers}")
    print(
        f"   Alerts: {'Enabled' if config.alerts.enable_email_alerts else 'Disabled'}"
    )

    print("✅ WCBB configuration test complete!")


if __name__ == "__main__":
    test_wcbb_config()
