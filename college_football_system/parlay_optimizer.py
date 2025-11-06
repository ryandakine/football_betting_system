#!/usr/bin/env python3
"""
College Football Parlay Optimization Engine
==========================================

Matches MLB system sophistication with intelligent parlay building,
correlation analysis, and stake optimization for college football.
"""

import json
import itertools
import logging
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ParlayBet:
    """Represents a parlay bet."""

    games: List[Dict[str, Any]]
    combined_odds: float
    expected_value: float
    confidence: float
    correlation_factor: float
    recommended_stake: float
    risk_score: float
    reasoning: List[str]


class ParlayOptimizer:
    """Advanced parlay optimization for college football."""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._get_default_config()
        self.correlation_matrix = self._build_correlation_matrix()

        logger.info("🎰 Parlay Optimizer initialized")

    def _get_default_config(self) -> Dict[str, Any]:
        """Default parlay configuration."""
        return {
            "max_parlay_size": 4,
            "min_confidence_threshold": 0.65,
            "min_edge_threshold": 0.03,
            "max_correlation_penalty": 0.15,
            "bankroll_allocation": 0.05,  # 5% of bankroll per parlay
            "risk_tolerance": "medium",
            "conference_correlations": {
                "same_conference": 0.3,  # 30% correlation penalty
                "rival_games": 0.2,
                "neutral_site": -0.1,  # Slight boost for neutral sites
            },
        }

    def _build_correlation_matrix(self) -> Dict[str, float]:
        """Build correlation matrix for college football games."""
        return {
            "same_conference": 0.25,
            "rival_teams": 0.20,
            "conference_championship": 0.35,
            "neutral_site": -0.05,
            "weather_impact": 0.15,
            "injury_impact": 0.10,
        }

    def optimize_parlays(self, games: List[Dict[str, Any]], bankroll: float) -> List[ParlayBet]:
        """Generate optimized parlays from high-confidence games."""
        # Filter high-confidence games
        eligible_games = [
            game
            for game in games
            if game.get("confidence", 0) >= self.config["min_confidence_threshold"]
            and game.get("edge_value", 0) >= self.config["min_edge_threshold"]
        ]

        if len(eligible_games) < 2:
            logger.info("Not enough eligible games for parlays")
            return []

        optimized_parlays: List[ParlayBet] = []

        # Generate parlays of different sizes
        for size in range(
            2, min(self.config["max_parlay_size"] + 1, len(eligible_games) + 1)
        ):
            size_parlays = self._generate_size_parlays(eligible_games, size)
            optimized_parlays.extend(size_parlays)

        # Sort by expected value
        optimized_parlays.sort(key=lambda x: x.expected_value, reverse=True)

        # Allocate stakes based on bankroll
        for parlay in optimized_parlays[:5]:  # Top 5 parlays
            parlay.recommended_stake = bankroll * self.config["bankroll_allocation"]
            parlay.risk_score = self._calculate_risk_score(parlay)

        logger.info("🎰 Generated %d optimized parlays", len(optimized_parlays))
        return optimized_parlays

    def _generate_size_parlays(self, games: List[Dict[str, Any]], size: int) -> List[ParlayBet]:
        """Generate all possible parlays of given size."""
        parlays = []

        for game_combination in itertools.combinations(games, size):
            parlay = self._evaluate_parlay(list(game_combination))
            if parlay:
                parlays.append(parlay)

        return parlays

    def _evaluate_parlay(self, games: List[Dict[str, Any]]) -> Optional[ParlayBet]:
        """Evaluate a potential parlay."""
        if not games:
            return None

        # Calculate combined odds (American odds)
        combined_odds = 100.0  # Neutral starting point
        for game in games:
            odds = game.get("odds", -110)
            if odds > 0:
                combined_odds *= odds / 100
            else:
                combined_odds *= 100 / abs(odds)

        # Calculate expected value
        avg_edge = sum(game.get("edge_value", 0) for game in games) / len(games)
        avg_confidence = sum(game.get("confidence", 0) for game in games) / len(games)
        correlation_penalty = self._calculate_correlation_penalty(games)

        expected_value = avg_edge - correlation_penalty
        if expected_value <= 0:
            return None  # Not profitable

        # Generate reasoning
        reasoning = [
            f"Average edge: {avg_edge:.2%}",
            f"Average confidence: {avg_confidence:.1%}",
            f"Correlation penalty: {correlation_penalty:.2%}",
        ]

        return ParlayBet(
            games=games,
            combined_odds=combined_odds,
            expected_value=expected_value,
            confidence=avg_confidence,
            correlation_factor=correlation_penalty,
            recommended_stake=0,  # Set later
            risk_score=0,  # Set later
            reasoning=reasoning,
        )

    def _calculate_correlation_penalty(self, games: List[Dict[str, Any]]) -> float:
        """Calculate correlation penalty for parlay."""
        total_penalty = 0.0

        # Same conference penalty
        conferences = [game.get("conference", "Unknown") for game in games]
        if len(set(conferences)) == 1:
            total_penalty += self.config["conference_correlations"]["same_conference"]

        # Rival teams penalty (mock logic)
        team_names = [game.get("home_team", "") + game.get("away_team", "") for game in games]
        if len(set(team_names)) < len(team_names):
            total_penalty += self.config["conference_correlations"]["rival_games"]

        return min(total_penalty, self.config["max_correlation_penalty"])

    def _calculate_risk_score(self, parlay: ParlayBet) -> float:
        """Calculate risk score for parlay (0-1, higher = riskier)."""
        # Risk based on size and correlation
        size_risk = (len(parlay.games) - 1) * 0.2  # Each additional game adds 20% risk
        correlation_risk = parlay.correlation_factor * 0.5
        confidence_risk = (1 - parlay.confidence) * 0.3

        total_risk = size_risk + correlation_risk + confidence_risk
        return min(total_risk, 1.0)

    def display_parlay_recommendations(self, parlays: List[ParlayBet]) -> None:
        """Display top parlay recommendations."""
        print("\n🎰 TOP PARLAY RECOMMENDATIONS")
        print("=" * 60)

        for i, parlay in enumerate(parlays[:3], 1):  # Top 3
            print(f"\n{i}. PARLAY ({len(parlay.games)} games)")
            print(f"   Expected Value: {parlay.expected_value:.2%}")
            print(f"   Combined Odds: +{parlay.combined_odds:.0f}")
            print(f"   Confidence: {parlay.confidence:.1%}")
            print(f"   Risk Score: {parlay.risk_score:.1%}")
            print(f"   Recommended Stake: ${parlay.recommended_stake:.0f}")

            print("   Games:")
            for game in parlay.games:
                print(
                    f"     • {game.get('away_team')} @ {game.get('home_team')} "
                    f"({game.get('conference')}) - Edge: {game.get('edge_value', 0):.1%}"
                )

            print(f"   Reasoning: {'; '.join(parlay.reasoning)}")


def test_parlay_optimizer() -> None:
    """Test the parlay optimizer."""
    print("🧪 Testing College Football Parlay Optimizer...")

    optimizer = ParlayOptimizer()

    # Mock high-confidence games
    mock_games = [
        {
            "home_team": "Alabama",
            "away_team": "Georgia",
            "conference": "SEC",
            "edge_value": 0.08,
            "confidence": 0.75,
            "odds": -110,
        },
        {
            "home_team": "Ohio State",
            "away_team": "Michigan",
            "conference": "Big Ten",
            "edge_value": 0.06,
            "confidence": 0.70,
            "odds": -115,
        },
        {
            "home_team": "Clemson",
            "away_team": "Florida State",
            "conference": "ACC",
            "edge_value": 0.05,
            "confidence": 0.68,
            "odds": -120,
        },
        {
            "home_team": "App State",
            "away_team": "Georgia Southern",
            "conference": "Sun Belt",
            "edge_value": 0.12,
            "confidence": 0.65,
            "odds": -105,
        },
    ]

    parlays = optimizer.optimize_parlays(mock_games, 10000)
    optimizer.display_parlay_recommendations(parlays)

    print("✅ Parlay optimizer test complete!")


if __name__ == "__main__":
    test_parlay_optimizer()
