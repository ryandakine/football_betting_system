from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import pytest

from betting_types import (
    NarrativeContext,
    NarrativeType,
    RiskLevel,
    SpreadPrediction,
    SentimentContext,
    RefereeProfile,
)
from unified_betting_intelligence import (
    NarrativeIntegratedAICouncil,
    BettingConfig,
    UnifiedPrediction,
)


@pytest.fixture
def config(tmp_path) -> BettingConfig:
    cfg = BettingConfig(
        version="test",
        thresholds={
            "contrarian": {"trigger": 0.70, "strong_trigger": 0.85},
            "narrative": {"trap_strength_min": 0.7, "impact_cap": 0.1},
            "referee": {"ot_specialist_threshold": 0.08, "high_penalty_threshold": 1.2},
            "confidence": {"high": 0.75, "medium": 0.60, "low": 0.50},
        },
        model_weights={"spread_expert": 1.0},
        risk_limits={"max_bet_fraction": 0.05, "max_daily_exposure": 0.15, "kelly_fraction": 0.25},
        feature_flags={"use_narratives": True, "use_sentiment": True, "use_referee": True, "use_crew": True, "use_parlays": True},
        model_checksums={},
    )
    return cfg


@pytest.fixture
def council(config) -> NarrativeIntegratedAICouncil:
    return NarrativeIntegratedAICouncil(config=config)


def test_validation_marks_degraded_mode(council: NarrativeIntegratedAICouncil) -> None:
    game: Dict[str, Any] = {"home_team": "A"}
    result = council.make_unified_prediction(game)  # type: ignore[arg-type]
    assert isinstance(result, UnifiedPrediction)
    assert result.degraded
    assert any("missing field" in reason for reason in result.degraded_reasons)


def test_narrative_adjustment_capped(config: BettingConfig) -> None:
    council = NarrativeIntegratedAICouncil(config=config)
    base = SpreadPrediction(pick="home", adjusted_line=-7.0, confidence=0.6, edge=0.0, american_odds=-110)
    narrative = NarrativeContext(
        storyline="Huge revenge",
        type=NarrativeType.REVENGE_GAME,
        strength=1.0,
        impact_home_team=0.5,
        expected_performance_delta=1.5,
        confidence=0.8,
    )
    sentiment = SentimentContext(0.0, 0.5, 0.0, 0.0, 0.0, 0.0)
    referee = RefereeProfile(
        name="Test",
        avg_margin=0.0,
        penalty_rate=6.0,
        overtime_frequency=0.05,
        home_advantage=0.0,
        classification="unknown",
        games_reffed=50,
    )
    adjusted = council._apply_adjustments(base, narrative, sentiment, referee)
    max_adjustment = config.thresholds["narrative"]["impact_cap"]
    assert abs(adjusted.adjusted_line - base.adjusted_line) <= max_adjustment + 1e-6


@pytest.mark.parametrize(
    "confidence,signals,expected",
    [
        (0.8, 4, RiskLevel.LOW),
        (0.65, 2, RiskLevel.MEDIUM),
        (0.55, 1, RiskLevel.HIGH),
    ],
)
def test_risk_assessment(confidence: float, signals: int, expected: RiskLevel, council: NarrativeIntegratedAICouncil) -> None:
    assert council._assess_risk(confidence, signals) == expected
