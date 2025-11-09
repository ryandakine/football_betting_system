#!/usr/bin/env python3
"""
Enhanced AI Council - 10 Model Super Intelligence
==================================================
Extends the base AI Council with 10 coordinated models:

Models 1-3: Base Ensembles (Spread, Total, Moneyline)
Models 4-6: New Prediction Targets (First Half, Team Totals)
Models 7-8: Algorithm Variants (XGBoost, Neural Net)
Model 9: Stacking Meta-Learner
Model 10: Situational Specialist
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List
from datetime import datetime

from betting_types import (
    GameData,
    UnifiedPrediction,
    SpreadPrediction,
    TotalPrediction,
    MoneylinePrediction,
    FirstHalfSpreadPrediction,
    TeamTotalPrediction,
    NarrativeContext,
    SentimentContext,
    RefereeProfile,
    RiskLevel,
    BetSize,
    NarrativeType,
)
from config_loader import BettingConfig
from unified_betting_intelligence import (
    NarrativeIntegratedAICouncil,
    DefaultSentimentAnalyzer,
)
from enhanced_model_architectures import (
    EnhancedModelRegistry,
    SituationalSpecialist,
)

logger = logging.getLogger(__name__)


class EnhancedAICouncil(NarrativeIntegratedAICouncil):
    """
    Enhanced AI Council with 10 coordinated models.
    Inherits base functionality and adds new model predictions.
    """

    VERSION = "2.0.0-ENHANCED"

    def __init__(
        self,
        config: Optional[BettingConfig] = None,
        model_registry: Optional[EnhancedModelRegistry] = None,
        sentiment_analyzer: Optional[Any] = None,
    ):
        # Initialize base council
        super().__init__(config, None, sentiment_analyzer)

        # Enhanced model registry with all 10 models
        self.enhanced_registry = model_registry or EnhancedModelRegistry()

        # Situational specialist for context adjustments
        self.specialist = SituationalSpecialist()

        # Model weights for ensemble (can be tuned based on performance)
        self.model_weights = {
            'spread_ensemble': 1.0,
            'total_ensemble': 1.0,
            'moneyline_ensemble': 1.0,
            'first_half_spread': 0.8,
            'home_team_total': 0.7,
            'away_team_total': 0.7,
            'xgboost_ensemble': 1.2,  # Often performs well
            'neural_net_deep': 0.9,
            'stacking_meta': 1.5,  # Meta-learner gets highest weight
        }

        logger.info(
            "Initialized Enhanced AI Council v%s with 10 models",
            self.VERSION
        )

    # ================================================================
    # Enhanced Prediction Methods
    # ================================================================
    def make_unified_prediction(self, game_data: GameData) -> UnifiedPrediction:
        """
        Enhanced prediction using all 10 models.
        Extends base prediction with additional model outputs.
        """
        # Get base predictions (Models 1-3)
        base_prediction = super().make_unified_prediction(game_data)

        # Add enhanced predictions (Models 4-10)
        first_half = self._get_first_half_prediction(game_data)
        home_total = self._get_team_total_prediction(game_data, "home")
        away_total = self._get_team_total_prediction(game_data, "away")

        # Get ensemble metadata with all model outputs
        ensemble_meta = self._get_ensemble_metadata(game_data)

        # Enhanced confidence using all 10 models
        enhanced_confidence = self._calculate_enhanced_confidence(
            base_prediction,
            first_half,
            home_total,
            away_total,
            ensemble_meta,
        )

        # Enhanced edge signals from all models
        enhanced_signals = self._detect_enhanced_edge_signals(
            base_prediction,
            first_half,
            home_total,
            away_total,
            ensemble_meta,
        )

        # Build enhanced prediction
        return UnifiedPrediction(
            game_id=base_prediction.game_id,
            home_team=base_prediction.home_team,
            away_team=base_prediction.away_team,
            timestamp=base_prediction.timestamp,
            spread_prediction=base_prediction.spread_prediction,
            total_prediction=base_prediction.total_prediction,
            moneyline_prediction=base_prediction.moneyline_prediction,
            narrative=base_prediction.narrative,
            sentiment=base_prediction.sentiment,
            referee=base_prediction.referee,
            recommendation=self._build_enhanced_recommendation(
                base_prediction,
                first_half,
                home_total,
                away_total,
                enhanced_signals,
            ),
            confidence=enhanced_confidence,
            risk_level=self._assess_risk(enhanced_confidence, len(enhanced_signals)),
            edge_signals=enhanced_signals,
            degraded=base_prediction.degraded,
            degraded_reasons=base_prediction.degraded_reasons,
            version=self.VERSION,
            first_half_spread_prediction=first_half,
            home_team_total_prediction=home_total,
            away_team_total_prediction=away_total,
            ensemble_metadata=ensemble_meta,
        )

    def _get_first_half_prediction(
        self, game_data: GameData
    ) -> Optional[FirstHalfSpreadPrediction]:
        """Get first half spread prediction from Model 4."""
        try:
            home_pct = float(game_data.get("first_half_spread_home_pct", 0.5) or 0.5)
            pick = "home" if home_pct > 0.5 else "away"
            confidence = max(0.0, min(1.0, abs(home_pct - 0.5) * 2))

            # Estimate first half line (usually ~half of full game)
            full_line = float(game_data.get("spread", 0.0))
            estimated_line = full_line * 0.55  # First half slightly more than half

            return FirstHalfSpreadPrediction(
                pick=pick,
                adjusted_line=estimated_line,
                confidence=confidence,
                edge=abs(home_pct - 0.5) * 2,
                american_odds=-110,
            )
        except Exception as e:
            logger.error(f"First half prediction failed: {e}")
            return None

    def _get_team_total_prediction(
        self, game_data: GameData, team: str
    ) -> Optional[TeamTotalPrediction]:
        """Get team total prediction from Models 5-6."""
        try:
            field_name = f"{team}_team_total_over_pct"
            over_pct = float(game_data.get(field_name, 0.5) or 0.5)
            pick = "over" if over_pct > 0.5 else "under"
            confidence = max(0.0, min(1.0, abs(over_pct - 0.5) * 2))

            # Estimate team total (usually ~half of game total)
            game_total = float(game_data.get("total", 44.5))
            estimated_team_total = game_total / 2.0

            # Home teams typically score slightly more
            if team == "home":
                estimated_team_total += 0.5

            return TeamTotalPrediction(
                team=team,  # type: ignore
                pick=pick,  # type: ignore
                adjusted_line=estimated_team_total,
                confidence=confidence,
                edge=abs(over_pct - 0.5) * 2,
                american_odds=-110,
            )
        except Exception as e:
            logger.error(f"Team total prediction failed for {team}: {e}")
            return None

    def _get_ensemble_metadata(self, game_data: GameData) -> Dict[str, Any]:
        """
        Collect all model outputs into metadata dict.
        Includes Models 7-9 (XGBoost, Neural Net, Stacking).
        """
        metadata = {
            "model_count": 10,
            "individual_probabilities": {},
            "situational_adjustments": [],
            "algorithm_variants": {},
        }

        # Base model probabilities (Models 1-3)
        metadata["individual_probabilities"]["spread_ensemble"] = float(
            game_data.get("spread_model_home_pct", 0.5) or 0.5
        )
        metadata["individual_probabilities"]["total_ensemble"] = float(
            game_data.get("total_model_over_pct", 0.5) or 0.5
        )
        metadata["individual_probabilities"]["moneyline_ensemble"] = float(
            game_data.get("home_advantage_pct", 0.5) or 0.5
        )

        # Enhanced model probabilities (Models 4-6)
        metadata["individual_probabilities"]["first_half_spread"] = float(
            game_data.get("first_half_spread_home_pct", 0.5) or 0.5
        )
        metadata["individual_probabilities"]["home_team_total"] = float(
            game_data.get("home_team_total_over_pct", 0.5) or 0.5
        )
        metadata["individual_probabilities"]["away_team_total"] = float(
            game_data.get("away_team_total_over_pct", 0.5) or 0.5
        )

        # Algorithm variant probabilities (Models 7-9)
        metadata["algorithm_variants"]["xgboost"] = float(
            game_data.get("xgboost_model_pct", 0.5) or 0.5
        )
        metadata["algorithm_variants"]["neural_net"] = float(
            game_data.get("neural_net_model_pct", 0.5) or 0.5
        )
        metadata["algorithm_variants"]["stacking_meta"] = float(
            game_data.get("stacking_model_pct", 0.5) or 0.5
        )

        # Situational context (Model 10)
        game_context = {
            'is_primetime': game_data.get('kickoff_window') in ['SNF', 'MNF', 'TNF'],
            'is_divisional': game_data.get('division') is not None,
            'weather': game_data.get('weather_tag', 'clear'),
            'rest_differential': 0,  # Would need to calculate from schedule
        }

        # Apply situational adjustments
        base_prob = metadata["individual_probabilities"]["spread_ensemble"]
        adjusted_prob, reasons = self.specialist.adjust_prediction(
            base_prob, game_context
        )
        metadata["situational_adjustments"] = reasons
        metadata["situational_boost"] = adjusted_prob - base_prob

        return metadata

    def _calculate_enhanced_confidence(
        self,
        base_prediction: UnifiedPrediction,
        first_half: Optional[FirstHalfSpreadPrediction],
        home_total: Optional[TeamTotalPrediction],
        away_total: Optional[TeamTotalPrediction],
        ensemble_meta: Dict[str, Any],
    ) -> float:
        """
        Calculate confidence using all 10 models.
        Weighted average with dynamic weighting based on agreement.
        """
        confidences = []
        weights = []

        # Base models (Models 1-3)
        confidences.append(base_prediction.spread_prediction.confidence)
        weights.append(self.model_weights['spread_ensemble'])

        confidences.append(base_prediction.total_prediction.confidence)
        weights.append(self.model_weights['total_ensemble'])

        confidences.append(base_prediction.moneyline_prediction.confidence)
        weights.append(self.model_weights['moneyline_ensemble'])

        # Enhanced predictions (Models 4-6)
        if first_half:
            confidences.append(first_half.confidence)
            weights.append(self.model_weights['first_half_spread'])

        if home_total:
            confidences.append(home_total.confidence)
            weights.append(self.model_weights['home_team_total'])

        if away_total:
            confidences.append(away_total.confidence)
            weights.append(self.model_weights['away_team_total'])

        # Algorithm variants (Models 7-9)
        xgb_conf = abs(ensemble_meta["algorithm_variants"]["xgboost"] - 0.5) * 2
        nn_conf = abs(ensemble_meta["algorithm_variants"]["neural_net"] - 0.5) * 2
        stack_conf = abs(ensemble_meta["algorithm_variants"]["stacking_meta"] - 0.5) * 2

        confidences.extend([xgb_conf, nn_conf, stack_conf])
        weights.extend([
            self.model_weights['xgboost_ensemble'],
            self.model_weights['neural_net_deep'],
            self.model_weights['stacking_meta'],
        ])

        # Weighted average
        total_weight = sum(weights)
        weighted_sum = sum(c * w for c, w in zip(confidences, weights))
        base_confidence = weighted_sum / total_weight if total_weight > 0 else 0.5

        # Boost for model agreement (all models pointing same direction)
        model_probs = list(ensemble_meta["individual_probabilities"].values())
        if len(model_probs) >= 6:
            agreement = 1.0 - np.std(model_probs)  # Low std = high agreement
            base_confidence += agreement * 0.15  # Up to +15% for perfect agreement

        # Situational boost (Model 10)
        situational_boost = ensemble_meta.get("situational_boost", 0.0)
        base_confidence += situational_boost * 0.5  # Weight situational adjustments

        # Sentiment and narrative boosts from base system
        base_confidence += min(0.1, base_prediction.sentiment.contrarian_score * 0.2)
        base_confidence += min(0.05, base_prediction.sentiment.crowd_roar_signal * 0.1)

        return min(1.0, max(0.0, base_confidence))

    def _detect_enhanced_edge_signals(
        self,
        base_prediction: UnifiedPrediction,
        first_half: Optional[FirstHalfSpreadPrediction],
        home_total: Optional[TeamTotalPrediction],
        away_total: Optional[TeamTotalPrediction],
        ensemble_meta: Dict[str, Any],
    ) -> List[str]:
        """Detect edge signals from all 10 models."""
        signals = list(base_prediction.edge_signals)

        # First half edge
        if first_half and first_half.confidence > 0.70:
            signals.append("FIRST_HALF_EDGE")

        # Team total edges
        if home_total and home_total.confidence > 0.70:
            signals.append(f"HOME_TOTAL_{home_total.pick.upper()}_EDGE")
        if away_total and away_total.confidence > 0.70:
            signals.append(f"AWAY_TOTAL_{away_total.pick.upper()}_EDGE")

        # Algorithm consensus
        algo_probs = ensemble_meta["algorithm_variants"]
        if all(p > 0.60 for p in algo_probs.values()):
            signals.append("ALGO_CONSENSUS_STRONG")

        # Model agreement across all 10
        all_probs = list(ensemble_meta["individual_probabilities"].values())
        if len(all_probs) >= 8:
            std_dev = np.std(all_probs)
            if std_dev < 0.10:  # Very tight agreement
                signals.append("UNANIMOUS_10_MODEL_EDGE")
            elif std_dev < 0.15:
                signals.append("STRONG_MODEL_AGREEMENT")

        # Situational edges
        signals.extend(ensemble_meta.get("situational_adjustments", []))

        return signals

    def _build_enhanced_recommendation(
        self,
        base_prediction: UnifiedPrediction,
        first_half: Optional[FirstHalfSpreadPrediction],
        home_total: Optional[TeamTotalPrediction],
        away_total: Optional[TeamTotalPrediction],
        enhanced_signals: List[str],
    ) -> Dict[str, Any]:
        """Build recommendation with all model insights."""
        recommendation = dict(base_prediction.recommendation)

        # Add secondary plays from enhanced models
        secondary_plays = recommendation.get("secondary_plays", [])

        if first_half and first_half.confidence > 0.65:
            secondary_plays.append({
                "play": f"1H {first_half.pick.upper()} {abs(first_half.adjusted_line)}",
                "reason": f"First half model {first_half.confidence:.0%} confident",
                "confidence": first_half.confidence,
            })

        if home_total and home_total.confidence > 0.65:
            secondary_plays.append({
                "play": f"{base_prediction.home_team} {home_total.pick.upper()} {home_total.adjusted_line:.1f}",
                "reason": f"Team total model {home_total.confidence:.0%} confident",
                "confidence": home_total.confidence,
            })

        if away_total and away_total.confidence > 0.65:
            secondary_plays.append({
                "play": f"{base_prediction.away_team} {away_total.pick.upper()} {away_total.adjusted_line:.1f}",
                "reason": f"Team total model {away_total.confidence:.0%} confident",
                "confidence": away_total.confidence,
            })

        recommendation["secondary_plays"] = secondary_plays

        # Upgrade bet size if we have unanimous agreement
        if "UNANIMOUS_10_MODEL_EDGE" in enhanced_signals:
            recommendation["size"] = BetSize.LARGE.value
            recommendation["reasoning"].append("UNANIMOUS agreement across all 10 models!")

        return recommendation


# ========================================================================
# AWS Lambda Handler for Enhanced Council
# ========================================================================
def handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda entry point for enhanced 10-model intelligence."""
    try:
        council = EnhancedAICouncil()

        game_data = event.get("game_data", {})

        if not game_data:
            return {
                "statusCode": 400,
                "body": {"error": "Missing game_data in event"},
            }

        # Generate enhanced prediction
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
                "model_count": prediction.ensemble_metadata.get("model_count", 10),
            },
        }

        # Add all predictions
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

        if prediction.first_half_spread_prediction:
            result["prediction"]["first_half_spread"] = {
                "pick": prediction.first_half_spread_prediction.pick,
                "adjusted_line": prediction.first_half_spread_prediction.adjusted_line,
                "confidence": prediction.first_half_spread_prediction.confidence,
            }

        if prediction.home_team_total_prediction:
            result["prediction"]["home_team_total"] = {
                "pick": prediction.home_team_total_prediction.pick,
                "adjusted_line": prediction.home_team_total_prediction.adjusted_line,
                "confidence": prediction.home_team_total_prediction.confidence,
            }

        if prediction.away_team_total_prediction:
            result["prediction"]["away_team_total"] = {
                "pick": prediction.away_team_total_prediction.pick,
                "adjusted_line": prediction.away_team_total_prediction.adjusted_line,
                "confidence": prediction.away_team_total_prediction.confidence,
            }

        # Add ensemble metadata
        result["prediction"]["ensemble_metadata"] = prediction.ensemble_metadata

        return result

    except Exception as exc:
        logger.error("Enhanced handler error: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "body": {"error": str(exc)},
        }
