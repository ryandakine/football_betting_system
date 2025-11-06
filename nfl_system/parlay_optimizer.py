#!/usr/bin/env python3
"""
NFL Parlay Optimization Engine
==============================

Generates intelligent NFL parlays using correlation-aware heuristics
and the gold-standard NFL configuration thresholds.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

from nfl_system.gold_standard_nfl_config import NFLGoldStandardConfig, get_nfl_config

logger = logging.getLogger(__name__)


@dataclass
class NFLParlayBet:
    games: List[Dict[str, Any]]
    combined_odds: float
    expected_value: float
    confidence: float
    correlation_factor: float
    recommended_stake: float
    risk_score: float
    reasoning: List[str]


class NFLParlayOptimizer:
    """Parlay builder tuned for professional football."""

    def __init__(self, *, config: Optional[NFLGoldStandardConfig] = None) -> None:
        self.config = config or get_nfl_config()
        thresholds = self.config.thresholds
        bankroll = self.config.bankroll

        self.max_parlay_size = 4
        self.min_confidence = max(thresholds.confidence_threshold, 0.6)
        self.min_edge = max(thresholds.min_edge_threshold, 0.03)
        self.max_correlation_penalty = 0.18
        self.bankroll_allocation = min(bankroll.max_exposure, 0.08)
        self.risk_tolerance = "medium"

        self.correlation_weights = {
            "same_division": 0.18,
            "same_conference": 0.12,
            "prime_time": 0.08,
            "travel_overlap": 0.10,
            "weather_overlap": 0.12,
            "indoor_mix": -0.05,
        }

        logger.info("🎰 NFL Parlay Optimizer initialized")

    def optimize_parlays(self, games: Iterable[Dict[str, Any]], bankroll: float) -> List[NFLParlayBet]:
        """Generate top parlays from eligible games."""
        high_conf_games = [
            game
            for game in games
            if game.get("confidence", 0.0) >= self.min_confidence
            and game.get("edge", game.get("edge_value", 0.0)) >= self.min_edge
        ]

        if len(high_conf_games) < 2:
            logger.info("Not enough NFL games meet parlay thresholds")
            return []

        parlays: List[NFLParlayBet] = []
        for size in range(2, min(self.max_parlay_size + 1, len(high_conf_games) + 1)):
            size_parlays = self._generate_size_parlays(high_conf_games, size)
            parlays.extend(size_parlays)

        parlays.sort(key=lambda entry: entry.expected_value, reverse=True)

        for parlay in parlays[:5]:
            parlay.recommended_stake = bankroll * self.bankroll_allocation
            parlay.risk_score = self._calculate_risk_score(parlay)

        return parlays

    def _generate_size_parlays(self, games: List[Dict[str, Any]], size: int) -> List[NFLParlayBet]:
        results: List[NFLParlayBet] = []
        for combo in itertools.combinations(games, size):
            parlay = self._evaluate_parlay(list(combo))
            if parlay:
                results.append(parlay)
        return results

    def _evaluate_parlay(self, games: List[Dict[str, Any]]) -> Optional[NFLParlayBet]:
        if not games:
            return None

        combined_multiplier = 1.0
        for game in games:
            odds = int(game.get("odds", -110))
            if odds >= 0:
                combined_multiplier *= 1 + odds / 100.0
            else:
                combined_multiplier *= 1 + 100.0 / abs(odds)

        avg_edge = sum(game.get("edge", game.get("edge_value", 0.0)) for game in games) / len(games)
        avg_confidence = sum(game.get("confidence", 0.0) for game in games) / len(games)
        correlation_penalty = self._correlation_penalty(games)

        expected_value = avg_edge - correlation_penalty
        if expected_value <= 0:
            return None

        reasoning = [
            f"Average edge {avg_edge:.2%}",
            f"Average confidence {avg_confidence:.1%}",
            f"Correlation penalty {correlation_penalty:.2%}",
        ]

        return NFLParlayBet(
            games=games,
            combined_odds=(combined_multiplier - 1) * 100,
            expected_value=expected_value,
            confidence=avg_confidence,
            correlation_factor=correlation_penalty,
            recommended_stake=0.0,
            risk_score=0.0,
            reasoning=reasoning,
        )

    def _correlation_penalty(self, games: List[Dict[str, Any]]) -> float:
        penalty = 0.0

        divisions = [str(game.get("division", "")).lower() for game in games if game.get("division")]
        if divisions and len(set(divisions)) < len(divisions):
            penalty += self.correlation_weights["same_division"]

        conferences = [str(game.get("conference", "")).lower() for game in games if game.get("conference")]
        if conferences and len(set(conferences)) < len(conferences):
            penalty += self.correlation_weights["same_conference"]

        kickoff_windows = [self._bucket(game) for game in games]
        if kickoff_windows.count("prime_time") >= 2:
            penalty += self.correlation_weights["prime_time"]

        weather_tags = [str(game.get("weather_tag", "")).lower() for game in games]
        if weather_tags and len(set(weather_tags)) == 1 and weather_tags[0] not in {"", "indoor"}:
            penalty += self.correlation_weights["weather_overlap"]
        if weather_tags.count("indoor") == len(weather_tags):
            penalty += self.correlation_weights["indoor_mix"]

        travel_tags = [str(game.get("travel_profile", "")).lower() for game in games]
        if travel_tags and len(set(travel_tags)) == 1 and travel_tags[0]:
            penalty += self.correlation_weights["travel_overlap"]

        return min(penalty, self.max_correlation_penalty)

    def _bucket(self, game: Dict[str, Any]) -> str:
        kickoff = str(game.get("kickoff_window", "")).lower()
        if kickoff in {"thursday night", "sunday night", "monday night"}:
            return "prime_time"
        return kickoff

    def _calculate_risk_score(self, parlay: NFLParlayBet) -> float:
        size_risk = (len(parlay.games) - 1) * 0.18
        correlation_risk = parlay.correlation_factor * 0.6
        confidence_risk = max(0.0, 1 - parlay.confidence) * 0.25
        return min(size_risk + correlation_risk + confidence_risk, 1.0)

    def display_recommendations(self, parlays: List[NFLParlayBet]) -> None:
        print("\n🎰 NFL PARLAY RECOMMENDATIONS")
        print("=" * 60)
        for idx, parlay in enumerate(parlays[:3], 1):
            print(f"\n{idx}. Parlay ({len(parlay.games)} games)")
            print(f"   Expected Value: {parlay.expected_value:.2%}")
            print(f"   Combined Odds: +{parlay.combined_odds:.0f}")
            print(f"   Confidence: {parlay.confidence:.1%}")
            print(f"   Risk Score: {parlay.risk_score:.1%}")
            print(f"   Recommended Stake: ${parlay.recommended_stake:.0f}")
            for reason in parlay.reasoning:
                print(f"   • {reason}")
            print("   Games:")
            for game in parlay.games:
                print(
                    f"     - {game.get('away_team')} @ {game.get('home_team')} "
                    f"(edge {game.get('edge', game.get('edge_value', 0)):.1%}, "
                    f"confidence {game.get('confidence', 0):.1%})"
                )


__all__ = ["NFLParlayOptimizer", "NFLParlayBet"]
