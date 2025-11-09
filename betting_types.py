from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Literal, Optional, TypedDict


class NarrativeType(Enum):
    REVENGE_GAME = "revenge_game"
    TRAP_GAME = "trap_game"
    HOMECOMING = "homecoming_narrative"
    DIVISIONAL = "divisional_rivalry"
    PRIMETIME = "primetime_pressure"
    INJURY_REVENGE = "injury_revenge"
    NEUTRAL = "neutral"


class BetSize(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Market(Enum):
    SPREAD = "spread"
    TOTAL = "total"
    MONEYLINE = "moneyline"
    FIRST_HALF_SPREAD = "first_half_spread"
    TEAM_TOTAL_HOME = "team_total_home"
    TEAM_TOTAL_AWAY = "team_total_away"


@dataclass
class NarrativeContext:
    storyline: str
    type: NarrativeType
    strength: float
    impact_home_team: float
    expected_performance_delta: float
    confidence: float


@dataclass
class SentimentContext:
    reddit_lean: float
    expert_consensus: float
    sharp_vs_public_ml: float
    sharp_vs_public_total: float
    contrarian_score: float
    crowd_roar_signal: float


@dataclass
class SpreadPrediction:
    pick: Literal["home", "away"]
    adjusted_line: float
    confidence: float
    edge: float
    american_odds: int


@dataclass
class TotalPrediction:
    pick: Literal["over", "under"]
    adjusted_line: float
    confidence: float
    edge: float
    american_odds: int


@dataclass
class MoneylinePrediction:
    pick: Literal["home", "away"]
    confidence: float
    edge: float
    home_odds: int
    away_odds: int


@dataclass
class FirstHalfSpreadPrediction:
    """First half spread prediction."""
    pick: Literal["home", "away"]
    adjusted_line: float
    confidence: float
    edge: float
    american_odds: int


@dataclass
class TeamTotalPrediction:
    """Individual team total prediction."""
    team: Literal["home", "away"]
    pick: Literal["over", "under"]
    adjusted_line: float
    confidence: float
    edge: float
    american_odds: int


@dataclass
class RefereeProfile:
    name: str
    avg_margin: float
    penalty_rate: float
    overtime_frequency: float
    home_advantage: float
    classification: str
    games_reffed: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.overtime_frequency <= 1.0:
            raise ValueError(f"overtime_frequency must be between 0 and 1, got {self.overtime_frequency}")
        if self.penalty_rate < 0:
            raise ValueError("penalty_rate must be non-negative")
        if self.games_reffed < 0:
            raise ValueError("games_reffed must be non-negative")


class GameData(TypedDict, total=False):
    game_id: str
    home_team: str
    away_team: str
    referee: str
    spread: float
    total: float
    home_ml_odds: int
    away_ml_odds: int
    timestamp: datetime
    days_since_last_monday_night: Optional[int]
    last_meeting_result: Optional[str]
    is_trap_game: Optional[bool]
    injuries: Optional[List[str]]
    injury_report: Optional[dict]
    reddit_lean: Optional[float]
    expert_pct_home: Optional[float]
    sharp_public_ml: Optional[float]
    sharp_public_total: Optional[float]
    contrarian_opportunity: Optional[float]
    crowd_roar: Optional[float]
    spread_model_home_pct: Optional[float]
    total_model_over_pct: Optional[float]
    home_advantage_pct: Optional[float]
    first_half_spread_home_pct: Optional[float]
    home_team_total_over_pct: Optional[float]
    away_team_total_over_pct: Optional[float]
    xgboost_model_pct: Optional[float]
    neural_net_model_pct: Optional[float]
    stacking_model_pct: Optional[float]
    latitude: Optional[float]
    longitude: Optional[float]
    subreddit: Optional[str]
    division: Optional[str]
    conference: Optional[str]
    kickoff_window: Optional[str]
    weather_tag: Optional[str]
    travel_profile: Optional[str]


@dataclass
class UnifiedPrediction:
    game_id: str
    home_team: str
    away_team: str
    timestamp: datetime
    spread_prediction: SpreadPrediction
    total_prediction: TotalPrediction
    moneyline_prediction: MoneylinePrediction
    narrative: NarrativeContext
    sentiment: SentimentContext
    referee: RefereeProfile
    recommendation: dict
    confidence: float
    risk_level: RiskLevel
    edge_signals: List[str]
    degraded: bool = False
    degraded_reasons: List[str] = field(default_factory=list)
    version: str = "1.0.0"
    # Enhanced predictions (optional)
    first_half_spread_prediction: Optional[FirstHalfSpreadPrediction] = None
    home_team_total_prediction: Optional[TeamTotalPrediction] = None
    away_team_total_prediction: Optional[TeamTotalPrediction] = None
    ensemble_metadata: dict = field(default_factory=dict)
