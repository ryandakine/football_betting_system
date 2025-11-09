#!/usr/bin/env python3
"""
Super Intelligence Predictor - Uses 10 models for predictions
Enhanced version of prediction agent
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


class SuperIntelligencePredictor:
    """
    Advanced predictor using 10-model super intelligence system
    Replaces basic SP+ predictions with ML ensemble
    """

    def __init__(self, config):
        self.config = config
        self.orchestrator = None
        self.feature_engineer = None
        self._load_models()

    def _load_models(self):
        """Load trained super intelligence models"""
        try:
            from ncaa_models.feature_engineering import NCAAFeatureEngineer
            from ncaa_models.super_intelligence import SuperIntelligenceOrchestrator

            self.feature_engineer = NCAAFeatureEngineer(
                data_dir=str(self.config.data_dir)
            )

            self.orchestrator = SuperIntelligenceOrchestrator(
                models_dir="models/ncaa"
            )

            # Try to load existing models
            self.orchestrator.load_all()

            # Check if any models are trained
            trained_count = sum(1 for m in self.orchestrator.models.values() if m.is_trained)

            if trained_count > 0:
                logger.info(f"Loaded {trained_count}/10 trained models")
            else:
                logger.warning("No trained models found. Run train_super_intelligence.py first!")

        except Exception as e:
            logger.error(f"Failed to load super intelligence: {e}")
            self.orchestrator = None

    async def generate_picks(self, days_ahead=7, min_edge=0.03, min_confidence=0.60, current_bankroll=10000):
        """
        Generate picks using super intelligence models
        """
        logger.info(f"Generating super intelligence picks for next {days_ahead} days...")

        try:
            # Get upcoming games
            from ncaa_agents.data_collector import DataCollectorAgent
            collector = DataCollectorAgent(self.config)
            upcoming_games = await collector.get_upcoming_games(days_ahead)

            if not upcoming_games:
                logger.info("No upcoming games found")
                return []

            # Load current season data for feature engineering
            season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
            self.feature_engineer.load_season_data(season)

            # Generate predictions for each game
            picks = []

            for game in upcoming_games:
                pick = await self._evaluate_game_with_super_intelligence(
                    game, season, min_edge, min_confidence, current_bankroll
                )
                if pick:
                    picks.append(pick)

            # Sort by edge (highest first)
            picks.sort(key=lambda x: x['edge'], reverse=True)

            logger.info(f"Generated {len(picks)} picks from {len(upcoming_games)} games")
            return picks

        except Exception as e:
            logger.error(f"Pick generation failed: {e}")
            return []

    async def _evaluate_game_with_super_intelligence(self, game, season, min_edge, min_confidence, bankroll):
        """
        Evaluate game using all 10 models
        """
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        if not home_team or not away_team:
            return None

        # Generate features for this game
        try:
            features_dict = self.feature_engineer.engineer_features(game, season)
        except Exception as e:
            logger.warning(f"Feature engineering failed for {home_team} vs {away_team}: {e}")
            return None

        # Convert to DataFrame for model input
        import pandas as pd
        features_df = pd.DataFrame([features_dict])

        # Get predictions from all models
        if self.orchestrator:
            try:
                all_predictions = self.orchestrator.predict(features_df)
                consensus = self.orchestrator.get_consensus_prediction(all_predictions)
            except Exception as e:
                logger.warning(f"Super intelligence prediction failed: {e}")
                return None
        else:
            return None

        # Extract key predictions
        predicted_spread = consensus.get('spread', [0])[0] if 'spread' in consensus else 0
        win_prob = consensus.get('win_probability', [0.5])[0] if 'win_probability' in consensus else 0.5
        spread_confidence = consensus.get('spread_confidence', [0.5])[0] if 'spread_confidence' in consensus else 0.5

        # Determine pick
        if predicted_spread > 0:
            pick_team = home_team
            pick_side = 'home'
            actual_win_prob = win_prob
        else:
            pick_team = away_team
            pick_side = 'away'
            actual_win_prob = 1 - win_prob

        # Calculate edge (probability above market implied)
        market_implied_prob = 0.5238  # -110 odds
        edge = actual_win_prob - market_implied_prob
        edge = max(0, edge)

        # Use spread_confidence as overall confidence
        confidence = spread_confidence

        # Check thresholds
        if edge < min_edge or confidence < min_confidence:
            return None

        # Calculate recommended stake (Kelly Criterion)
        b = 0.909  # For -110 odds
        kelly_fraction = (actual_win_prob * b - (1 - actual_win_prob)) / b
        kelly_fraction = max(0, min(0.25, kelly_fraction))

        recommended_stake = bankroll * kelly_fraction
        recommended_stake = max(recommended_stake, 10)  # Min $10
        recommended_stake = min(recommended_stake, 200)  # Max $200

        # Generate detailed reasoning
        reasoning = self._generate_super_intelligence_reasoning(
            all_predictions, consensus, home_team, away_team, predicted_spread
        )

        return {
            'game_id': game.get('id'),
            'home_team': home_team,
            'away_team': away_team,
            'date': game.get('start_date', ''),
            'pick': pick_side,
            'predicted_winner': pick_team,
            'edge': edge,
            'confidence': confidence,
            'predicted_spread': predicted_spread,
            'predicted_total': consensus.get('total', [50])[0] if 'total' in consensus else 50,
            'win_prob': actual_win_prob,
            'recommended_stake': round(recommended_stake, 2),
            'reasoning': reasoning,
            'model_predictions': {
                'spread': float(predicted_spread),
                'total': float(consensus.get('total', [50])[0]) if 'total' in consensus else 50,
                '1h_spread': float(consensus.get('1h_spread', [0])[0]) if '1h_spread' in consensus else 0,
                'home_total': float(consensus.get('home_total', [25])[0]) if 'home_total' in consensus else 25,
                'away_total': float(consensus.get('away_total', [25])[0]) if 'away_total' in consensus else 25,
            }
        }

    def _generate_super_intelligence_reasoning(self, all_predictions, consensus, home_team, away_team, predicted_spread):
        """
        Generate detailed reasoning from multi-model analysis
        """
        reasoning_parts = []

        # Model consensus
        spread_models = ['spread_ensemble', 'xgboost_super', 'neural_net_deep', 'stacking_meta']
        spread_preds = [all_predictions.get(m, [0])[0] for m in spread_models if m in all_predictions]

        if spread_preds:
            avg_spread = np.mean(spread_preds)
            std_spread = np.std(spread_preds)

            stronger_team = home_team if avg_spread > 0 else away_team

            reasoning_parts.append(
                f"10-Model Consensus: {stronger_team} favored by {abs(avg_spread):.1f} points "
                f"(confidence: ±{std_spread:.1f})"
            )

        # Total prediction
        if 'total' in consensus:
            total_pred = consensus['total'][0]
            reasoning_parts.append(f"Projected total: {total_pred:.1f} points")

        # Team totals
        if 'home_total' in consensus and 'away_total' in consensus:
            home_t = consensus['home_total'][0]
            away_t = consensus['away_total'][0]
            reasoning_parts.append(f"Team totals: {home_team} {home_t:.1f}, {away_team} {away_t:.1f}")

        # Model agreement
        if len(spread_preds) > 1:
            agreement = 1 - (std_spread / (abs(avg_spread) + 1))
            if agreement > 0.8:
                reasoning_parts.append("Strong model agreement ✓")
            elif agreement > 0.6:
                reasoning_parts.append("Moderate model agreement")
            else:
                reasoning_parts.append("Models show some disagreement")

        return " | ".join(reasoning_parts)
