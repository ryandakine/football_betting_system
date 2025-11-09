#!/usr/bin/env python3
"""
Prediction Agent - Generates betting picks for upcoming games
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class PredictionAgent:
    """Agent responsible for generating betting predictions"""

    def __init__(self, config):
        self.config = config

    async def generate_picks(self, days_ahead=7, min_edge=0.03, min_confidence=0.60, current_bankroll=10000):
        """Generate picks for upcoming games"""
        logger.info(f"Generating picks for next {days_ahead} days...")

        try:
            # Get upcoming games
            from ncaa_agents.data_collector import DataCollectorAgent
            collector = DataCollectorAgent(self.config)
            upcoming_games = await collector.get_upcoming_games(days_ahead)

            if not upcoming_games:
                logger.info("No upcoming games found")
                return []

            # Load SP+ ratings
            season = datetime.now().year if datetime.now().month >= 8 else datetime.now().year - 1
            sp_ratings = self._load_sp_ratings(season)

            if not sp_ratings:
                logger.warning("No SP+ ratings available")
                return []

            # Generate predictions for each game
            picks = []
            for game in upcoming_games:
                pick = self._evaluate_game(game, sp_ratings, min_edge, min_confidence, current_bankroll)
                if pick:
                    picks.append(pick)

            # Sort by edge (highest first)
            picks.sort(key=lambda x: x['edge'], reverse=True)

            logger.info(f"Generated {len(picks)} picks from {len(upcoming_games)} games")
            return picks

        except Exception as e:
            logger.error(f"Pick generation failed: {e}")
            return []

    def _load_sp_ratings(self, year):
        """Load SP+ ratings for season"""
        sp_file = self.config.data_dir / f"ncaaf_{year}_sp_ratings.json"

        if not sp_file.exists():
            return {}

        with open(sp_file) as f:
            sp_data = json.load(f)

        # Create lookup dict
        ratings = {}
        for team in sp_data:
            team_name = team.get('team', '')
            rating = team.get('rating', 0)
            if team_name and rating:
                ratings[team_name] = rating

        return ratings

    def _evaluate_game(self, game, sp_ratings, min_edge, min_confidence, current_bankroll):
        """Evaluate a game and generate pick if criteria met"""
        home_team = game.get('home_team', '')
        away_team = game.get('away_team', '')

        if not home_team or not away_team:
            return None

        # Get SP+ ratings
        home_sp = sp_ratings.get(home_team, 0)
        away_sp = sp_ratings.get(away_team, 0)

        if not home_sp or not away_sp:
            return None

        # Calculate prediction
        home_advantage = 3.0
        predicted_margin = (home_sp - away_sp) + home_advantage

        # Convert to win probability
        win_prob = 1 / (1 + np.exp(-predicted_margin / 14))

        # Calculate edge
        market_implied_prob = 0.5238
        if win_prob > 0.5:
            edge = win_prob - market_implied_prob
            pick_team = home_team
            pick_side = 'home'
        else:
            edge = (1 - win_prob) - market_implied_prob
            pick_team = away_team
            pick_side = 'away'

        edge = max(0, edge)

        # Confidence based on SP+ differential
        sp_diff = abs(home_sp - away_sp)
        confidence = min(0.95, 0.5 + (sp_diff / 40))

        # Check thresholds
        if edge < min_edge or confidence < min_confidence:
            return None

        # Calculate recommended stake (Kelly Criterion)
        b = 0.909  # For -110 odds
        kelly_fraction = (win_prob * b - (1 - win_prob)) / b
        kelly_fraction = max(0, min(0.25, kelly_fraction))

        recommended_stake = current_bankroll * kelly_fraction
        recommended_stake = max(recommended_stake, 10)  # Min $10
        recommended_stake = min(recommended_stake, 200)  # Max $200 per bet

        # Generate reasoning
        reasoning = self._generate_reasoning(home_team, away_team, home_sp, away_sp, predicted_margin, pick_side)

        return {
            'game_id': game.get('id'),
            'home_team': home_team,
            'away_team': away_team,
            'date': game.get('start_date', ''),
            'pick': pick_side,
            'predicted_winner': pick_team,
            'edge': edge,
            'confidence': confidence,
            'predicted_margin': predicted_margin,
            'win_prob': win_prob,
            'recommended_stake': round(recommended_stake, 2),
            'home_sp': home_sp,
            'away_sp': away_sp,
            'reasoning': reasoning
        }

    def _generate_reasoning(self, home_team, away_team, home_sp, away_sp, margin, pick_side):
        """Generate human-readable reasoning for pick"""
        sp_diff = abs(home_sp - away_sp)

        if pick_side == 'home':
            stronger_team = home_team
            stronger_sp = home_sp
            weaker_sp = away_sp
        else:
            stronger_team = away_team
            stronger_sp = away_sp
            weaker_sp = home_sp

        reasoning = f"{stronger_team} has significant SP+ advantage ({stronger_sp:.1f} vs {weaker_sp:.1f}, diff: {sp_diff:.1f}). "

        if sp_diff > 20:
            reasoning += "Large talent gap suggests high win probability."
        elif sp_diff > 10:
            reasoning += "Moderate edge based on SP+ differential."
        else:
            reasoning += "Small but measurable edge."

        if pick_side == 'home':
            reasoning += " Home field advantage included."

        return reasoning
