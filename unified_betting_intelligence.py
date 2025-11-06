from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from betting_types import (
    BetSize,
    GameData,
    MoneylinePrediction,
    NarrativeContext,
    NarrativeType,
    RefereeProfile,
    RiskLevel,
    SentimentContext,
    SpreadPrediction,
    TotalPrediction,
    UnifiedPrediction,
)
from config_loader import BettingConfig
from ai_council_narrative_unified import american_to_implied_prob, remove_vig


logger = logging.getLogger(__name__)


class SentimentAnalyzer(Protocol):
    """Protocol describing sentiment extraction engines."""

    def analyze(self, game_data: GameData) -> SentimentContext:
        ...


class ModelRegistry(Protocol):
    """Protocol describing access to trained model artifacts."""

    def get_model(self, name: str) -> Any:
        ...


@dataclass
class ValidationResult:
    is_valid: bool
    errors: List[str]


class DefaultSentimentAnalyzer:
    """Fallback sentiment analyzer."""

    def analyze(self, game_data: GameData) -> SentimentContext:
        reddit = float(game_data.get("reddit_lean", 0.0) or 0.0)
        expert = float(game_data.get("expert_pct_home", 0.5) or 0.5)
        sharp_ml = float(game_data.get("sharp_public_ml", 0.0) or 0.0)
        sharp_total = float(game_data.get("sharp_public_total", 0.0) or 0.0)
        contrarian = float(game_data.get("contrarian_opportunity", 0.0) or 0.0)
        crowd_roar = float(game_data.get("crowd_roar", 0.0) or 0.0)
        return SentimentContext(
            reddit_lean=reddit,
            expert_consensus=expert,
            sharp_vs_public_ml=sharp_ml,
            sharp_vs_public_total=sharp_total,
            contrarian_score=contrarian,
            crowd_roar_signal=crowd_roar,
        )


class InMemoryModelRegistry:
    """Simple registry that stores model objects in memory."""

    def __init__(self, models: Optional[Dict[str, Any]] = None) -> None:
        self._models: Dict[str, Any] = models or {}

    def register(self, name: str, model: Any) -> None:
        self._models[name] = model

    def get_model(self, name: str) -> Any:
        return self._models.get(name)


class NarrativeIntegratedAICouncil:
    """Production-ready betting intelligence system."""

    VERSION = "1.2.0"

    def __init__(
        self,
        config: Optional[BettingConfig] = None,
        model_registry: Optional[ModelRegistry] = None,
        sentiment_analyzer: Optional[SentimentAnalyzer] = None,
    ) -> None:
        self.config = config or BettingConfig.from_yaml()
        self.model_registry = model_registry or InMemoryModelRegistry()
        self.sentiment_analyzer = sentiment_analyzer or DefaultSentimentAnalyzer()
        self.degraded_mode = False
        self.degraded_reasons: List[str] = []
        logger.info(
            "Initialized %s v%s", self.__class__.__name__, self.VERSION
        )

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def make_unified_prediction(self, game_data: GameData) -> UnifiedPrediction:
        """Main prediction method with type safety and graceful degradation."""
        self.degraded_mode = False
        self.degraded_reasons.clear()

        validation = self._validate_game_data(game_data)
        if not validation.is_valid:
            self.degraded_mode = True
            self.degraded_reasons.extend(validation.errors)
            logger.warning("Invalid game data: %s", validation.errors)

        narrative = self._extract_narrative_safe(game_data)
        sentiment = self._extract_sentiment_safe(game_data)
        referee = self._extract_referee_safe(game_data)

        spread = self._get_spread_prediction(game_data, narrative, referee)
        total = self._get_total_prediction(game_data, narrative, referee)
        moneyline = self._get_moneyline_prediction(game_data, narrative, referee)

        if not self.degraded_mode:
            spread = self._apply_adjustments(spread, narrative, sentiment, referee)

        recommendation = self._build_recommendation(
            spread, total, moneyline, narrative, sentiment, referee
        )
        edge_signals = recommendation.get("edge_signals", [])
        confidence = self._calculate_calibrated_confidence(spread, sentiment, referee)
        risk = self._assess_risk(confidence, len(edge_signals))

        timestamp = self._coerce_timestamp(game_data.get("timestamp"))

        return UnifiedPrediction(
            game_id=game_data.get("game_id", "UNKNOWN"),
            home_team=game_data.get("home_team", "HOME"),
            away_team=game_data.get("away_team", "AWAY"),
            timestamp=timestamp,
            spread_prediction=spread,
            total_prediction=total,
            moneyline_prediction=moneyline,
            narrative=narrative,
            sentiment=sentiment,
            referee=referee,
            recommendation=recommendation,
            confidence=confidence,
            risk_level=risk,
            edge_signals=edge_signals,
            degraded=self.degraded_mode,
            degraded_reasons=list(self.degraded_reasons),
            version=self.VERSION,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _validate_game_data(self, game_data: GameData) -> ValidationResult:
        errors: List[str] = []
        required_fields = ["game_id", "home_team", "away_team", "spread", "total"]
        for field in required_fields:
            if field not in game_data:
                errors.append(f"missing field {field}")
        return ValidationResult(is_valid=not errors, errors=errors)

    def _extract_narrative_safe(self, game_data: GameData) -> NarrativeContext:
        try:
            return self._extract_narrative(game_data)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Narrative extraction failed: %s", exc, exc_info=True)
            self.degraded_mode = True
            self.degraded_reasons.append("narrative failure")
            return NarrativeContext(
                storyline="Standard matchup",
                type=NarrativeType.NEUTRAL,
                strength=0.0,
                impact_home_team=0.0,
                expected_performance_delta=0.0,
                confidence=0.5,
            )

    def _extract_sentiment_safe(self, game_data: GameData) -> SentimentContext:
        try:
            if self.config.is_feature_enabled("sentiment"):
                return self.sentiment_analyzer.analyze(game_data)
            return DefaultSentimentAnalyzer().analyze(game_data)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Sentiment extraction failed: %s", exc, exc_info=True)
            self.degraded_mode = True
            self.degraded_reasons.append("sentiment failure")
            return DefaultSentimentAnalyzer().analyze({})

    def _extract_referee_safe(self, game_data: GameData) -> RefereeProfile:
        name = game_data.get("referee", "Unknown")
        # A real implementation would pull from a database; we return neutral defaults.
        return RefereeProfile(
            name=name,
            avg_margin=0.0,
            penalty_rate=6.1,
            overtime_frequency=0.06,
            home_advantage=0.0,
            classification="unknown",
            games_reffed=0,
        )

    def _extract_narrative(self, game_data: GameData) -> NarrativeContext:
        narratives: List[NarrativeContext] = []

        days_since_mnf = game_data.get("days_since_last_monday_night")
        if days_since_mnf and days_since_mnf > 365 * 5:
            narratives.append(
                NarrativeContext(
                    storyline=f"First Monday night in {days_since_mnf // 365} years",
                    type=NarrativeType.HOMECOMING,
                    strength=0.7,
                    impact_home_team=-0.03,
                    expected_performance_delta=-0.09,
                    confidence=0.8,
                )
            )

        last_meeting = game_data.get("last_meeting_result")
        if last_meeting:
            narratives.append(
                NarrativeContext(
                    storyline=f"Rematch after {last_meeting}",
                    type=NarrativeType.REVENGE_GAME,
                    strength=0.6,
                    impact_home_team=0.02,
                    expected_performance_delta=0.06,
                    confidence=0.7,
                )
            )

        if game_data.get("is_trap_game"):
            narratives.append(
                NarrativeContext(
                    storyline="Potential trap after big win",
                    type=NarrativeType.TRAP_GAME,
                    strength=0.8,
                    impact_home_team=-0.05,
                    expected_performance_delta=-0.15,
                    confidence=0.75,
                )
            )

        if game_data.get("injuries") or game_data.get("injury_report"):
            narratives.append(
                NarrativeContext(
                    storyline="Injury revenge angle for returning starters",
                    type=NarrativeType.INJURY_REVENGE,
                    strength=0.7,
                    impact_home_team=0.05,
                    expected_performance_delta=0.15,
                    confidence=0.75,
                )
            )

        if narratives:
            return max(narratives, key=lambda ctx: ctx.strength)

        return NarrativeContext(
            storyline="Standard matchup",
            type=NarrativeType.NEUTRAL,
            strength=0.0,
            impact_home_team=0.0,
            expected_performance_delta=0.0,
            confidence=0.5,
        )

    def _get_spread_prediction(
        self,
        game_data: GameData,
        narrative: NarrativeContext,
        referee: RefereeProfile,
    ) -> SpreadPrediction:
        home_pct = float(game_data.get("spread_model_home_pct", 0.5) or 0.5)
        pick = "home" if home_pct > 0.5 else "away"
        confidence = max(0.0, min(1.0, abs(home_pct - 0.5) * 2))
        line = float(game_data.get("spread", 0.0))
        odds = int(game_data.get("spread_odds", -110))

        narrative_adjustment = narrative.expected_performance_delta
        referee_adjustment = referee.avg_margin * 0.5
        total_adjustment = narrative_adjustment + referee_adjustment

        return SpreadPrediction(
            pick=pick,
            adjusted_line=line + total_adjustment,
            confidence=confidence,
            edge=total_adjustment,
            american_odds=odds,
        )

    def _get_total_prediction(
        self,
        game_data: GameData,
        narrative: NarrativeContext,
        referee: RefereeProfile,
    ) -> TotalPrediction:
        over_pct = float(game_data.get("total_model_over_pct", 0.5) or 0.5)
        pick = "over" if over_pct > 0.5 else "under"
        confidence = max(0.0, min(1.0, abs(over_pct - 0.5) * 2))
        line = float(game_data.get("total", 44.5))
        odds = int(game_data.get("total_odds", -110))

        trap_threshold = self.config.thresholds["narrative"]["trap_strength_min"]
        narrative_adjustment = -0.5 if narrative.type == NarrativeType.TRAP_GAME and narrative.strength > trap_threshold else 0.0
        ot_threshold = self.config.thresholds["referee"]["ot_specialist_threshold"]
        referee_adjustment = 0.5 if referee.overtime_frequency > ot_threshold else 0.0

        total_adjustment = narrative_adjustment + referee_adjustment
        return TotalPrediction(
            pick=pick,
            adjusted_line=line + total_adjustment,
            confidence=confidence,
            edge=total_adjustment,
            american_odds=odds,
        )

    def _get_moneyline_prediction(
        self,
        game_data: GameData,
        narrative: NarrativeContext,
        referee: RefereeProfile,
    ) -> MoneylinePrediction:
        home_odds = int(game_data.get("home_ml_odds", -110))
        away_odds = int(game_data.get("away_ml_odds", 110))
        implied_home = american_to_implied_prob(home_odds)
        implied_away = american_to_implied_prob(away_odds)
        implied_home, implied_away = remove_vig(implied_home, implied_away)

        model_bias = game_data.get("home_advantage_pct")
        if model_bias is not None:
            pick = "home" if model_bias > 0.5 else "away"
            model_confidence = abs(float(model_bias) - 0.5) * 2
        else:
            pick = "home" if implied_home >= implied_away else "away"
            model_confidence = abs(implied_home - implied_away)

        baseline_prob = implied_home if pick == "home" else implied_away
        confidence = max(model_confidence, baseline_prob)
        edge = baseline_prob - 0.5

        return MoneylinePrediction(
            pick=pick,
            confidence=max(0.0, min(1.0, confidence)),
            edge=edge,
            home_odds=home_odds,
            away_odds=away_odds,
        )

    def _apply_adjustments(
        self,
        prediction: SpreadPrediction,
        narrative: NarrativeContext,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> SpreadPrediction:
        adjusted = SpreadPrediction(
            pick=prediction.pick,
            adjusted_line=prediction.adjusted_line,
            confidence=prediction.confidence,
            edge=prediction.edge,
            american_odds=prediction.american_odds,
        )

        max_adjustment = self.config.thresholds["narrative"]["impact_cap"]
        adjustment = max(-max_adjustment, min(max_adjustment, narrative.impact_home_team))
        adjusted.adjusted_line += adjustment * narrative.strength
        adjusted.edge += adjustment * narrative.strength

        ref_adjustment = referee.avg_margin * 0.1
        adjusted.adjusted_line += ref_adjustment
        adjusted.edge += ref_adjustment

        return adjusted

    def _build_recommendation(
        self,
        spread: SpreadPrediction,
        total: TotalPrediction,
        moneyline: MoneylinePrediction,
        narrative: NarrativeContext,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> Dict[str, Any]:
        recommendation: Dict[str, Any] = {
            "primary_play": None,
            "size": BetSize.SMALL.value,
            "reasoning": [],
            "secondary_plays": [],
        }

        edge_signals: List[str] = []
        contrarian_cfg = self.config.thresholds["contrarian"]

        if sentiment.contrarian_score > contrarian_cfg["trigger"]:
            recommendation["primary_play"] = "CONTRARIAN: Fade public"
            recommendation["size"] = (
                BetSize.LARGE.value
                if sentiment.contrarian_score > contrarian_cfg["strong_trigger"]
                else BetSize.MEDIUM.value
            )
            recommendation["reasoning"].append(
                f"Sharp/public divergence {sentiment.contrarian_score:.1%}."
            )
            edge_signals.append("CONTRARIAN_EDGE")

        elif narrative.type == NarrativeType.TRAP_GAME and narrative.strength > self.config.thresholds["narrative"]["trap_strength_min"]:
            recommendation["primary_play"] = f"FADE {narrative.storyline}"
            recommendation["size"] = BetSize.MEDIUM.value
            recommendation["reasoning"].append("Trap game detection.")
            edge_signals.append("NARRATIVE_TRAP")

        if referee.overtime_frequency > self.config.thresholds["referee"]["ot_specialist_threshold"]:
            recommendation["secondary_plays"].append(
                {
                    "play": "Overtime props",
                    "reason": f"{referee.name} OT rate {referee.overtime_frequency:.1%}",
                }
            )
            edge_signals.append("OT_SPECIALIST")

        recommendation["edge_signals"] = edge_signals
        return recommendation

    def _calculate_calibrated_confidence(
        self,
        spread: SpreadPrediction,
        sentiment: SentimentContext,
        referee: RefereeProfile,
    ) -> float:
        base_confidence = 0.5
        base_confidence += spread.confidence * 0.3
        base_confidence += min(0.2, sentiment.contrarian_score * 0.3)
        if referee.classification and referee.classification != "unknown":
            base_confidence += 0.1
        return min(1.0, max(0.0, base_confidence))

    def _assess_risk(self, confidence: float, signal_count: int) -> RiskLevel:
        if signal_count > 3:
            return RiskLevel.LOW
        if signal_count > 1 and confidence >= self.config.thresholds["confidence"]["medium"]:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def _coerce_timestamp(self, ts: Optional[Any]) -> datetime:
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, str):
            try:
                return datetime.fromisoformat(ts)
            except ValueError:
                logger.warning("Invalid timestamp string '%s', using now()", ts)
        return datetime.now()


# ============================================================================ #
# AWS Lambda Handler
# ============================================================================ #
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for unified betting intelligence."""
    try:
        council = NarrativeIntegratedAICouncil()
        
        # Extract game data from Lambda event
        game_data = event.get("game_data", {})
        
        if not game_data:
            return {
                "statusCode": 400,
                "body": {"error": "Missing game_data in event"},
            }
        
        # Generate prediction
        prediction = council.make_unified_prediction(game_data)
        
        # Convert to dict for JSON serialization
        result = {
            "statusCode": 200,
            "prediction": {
                "game_id": prediction.game_id,
                "home_team": prediction.home_team,
                "away_team": prediction.away_team,
                "timestamp": prediction.timestamp.isoformat(),
                "confidence": prediction.confidence,
                "risk_level": prediction.risk_level.value,
                "edge_signals": prediction.edge_signals,
                "degraded": prediction.degraded,
                "version": prediction.version,
            },
        }
        
        if prediction.spread_prediction:
            result["prediction"]["spread"] = {
                "pick": prediction.spread_prediction.pick,
                "adjusted_line": prediction.spread_prediction.adjusted_line,
                "confidence": prediction.spread_prediction.confidence,
            }
        
        if prediction.total_prediction:
            result["prediction"]["total"] = {
                "pick": prediction.total_prediction.pick,
                "adjusted_line": prediction.total_prediction.adjusted_line,
                "confidence": prediction.total_prediction.confidence,
            }
        
        if prediction.moneyline_prediction:
            result["prediction"]["moneyline"] = {
                "pick": prediction.moneyline_prediction.pick,
                "confidence": prediction.moneyline_prediction.confidence,
            }
        
        return result
    
    except Exception as exc:
        logger.error("Handler error: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(exc)},
        }
