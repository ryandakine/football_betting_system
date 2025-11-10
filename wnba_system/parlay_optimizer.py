#!/usr/bin/env python3
"""
WNBA Parlay Optimizer
====================

Optimizes parlay construction for WNBA betting.
"""

import logging
from typing import List, Dict, Any
from itertools import combinations

logger = logging.getLogger(__name__)


class ParlayOptimizer:
    """
    Optimize parlay construction for WNBA.
    Balances correlation, variance, and expected value.
    """

    def __init__(
        self,
        min_legs: int = 2,
        max_legs: int = 3,  # WNBA: fewer legs due to smaller slate
        min_edge_per_leg: float = 0.06,
        correlation_penalty: float = 0.12,  # Lower penalty for WNBA
    ):
        self.min_legs = min_legs
        self.max_legs = max_legs
        self.min_edge_per_leg = min_edge_per_leg
        self.correlation_penalty = correlation_penalty
        logger.info("🎰 WNBA Parlay Optimizer initialized")

    def optimize_parlays(
        self,
        games: List[Dict[str, Any]],
        bankroll: float,
        max_parlays: int = 3,  # Fewer games = fewer parlays
    ) -> List[Dict[str, Any]]:
        """
        Generate optimized parlays from analyzed games.

        Args:
            games: List of analyzed game results
            bankroll: Available bankroll
            max_parlays: Maximum number of parlays to generate

        Returns:
            List of optimized parlay recommendations
        """
        if not games:
            return []

        # Filter games with sufficient edge
        qualified_games = [
            g for g in games
            if g.get('total_edge', 0) >= self.min_edge_per_leg
        ]

        if len(qualified_games) < self.min_legs:
            logger.info("Insufficient qualified games for parlays")
            return []

        parlays = []

        # Generate 2-leg parlays
        for combo in combinations(qualified_games, 2):
            parlay = self._build_parlay(combo, bankroll)
            if parlay:
                parlays.append(parlay)

        # Generate 3-leg parlays if we have enough games
        if len(qualified_games) >= 3:
            for combo in combinations(qualified_games, 3):
                parlay = self._build_parlay(combo, bankroll)
                if parlay:
                    parlays.append(parlay)

        # Sort by expected value and return top N
        parlays.sort(key=lambda p: p['expected_value'], reverse=True)
        return parlays[:max_parlays]

    def _build_parlay(
        self,
        games: tuple,
        bankroll: float,
    ) -> Dict[str, Any]:
        """Build a single parlay from a combination of games."""
        try:
            # Calculate combined edge (with correlation penalty)
            edges = [g.get('total_edge', 0) for g in games]
            confidences = [g.get('confidence', 0.5) for g in games]

            # Apply correlation penalty
            combined_edge = sum(edges) * (1.0 - self.correlation_penalty)
            combined_confidence = sum(confidences) / len(confidences)

            # Calculate parlay odds (assuming -110 per leg)
            parlay_odds = 1.0
            for _ in games:
                parlay_odds *= 1.909  # -110 decimal odds

            # Calculate stake (Kelly criterion with safety factor)
            edge_fraction = combined_edge
            kelly_fraction = edge_fraction / (parlay_odds - 1)
            safe_kelly = kelly_fraction * 0.20  # Conservative for WNBA
            stake = min(bankroll * safe_kelly, bankroll * 0.025)  # Cap at 2.5% of bankroll

            expected_value = stake * combined_edge

            return {
                'legs': [
                    {
                        'game_id': g.get('game_id'),
                        'matchup': f"{g.get('away_team')} @ {g.get('home_team')}",
                        'pick': g.get('unified_recommendation', {}).get('final_action', 'PASS'),
                        'edge': g.get('total_edge'),
                    }
                    for g in games
                ],
                'num_legs': len(games),
                'combined_edge': combined_edge,
                'combined_confidence': combined_confidence,
                'parlay_odds': parlay_odds,
                'stake': stake,
                'expected_value': expected_value,
                'potential_payout': stake * parlay_odds,
            }

        except Exception as exc:
            logger.debug(f"Failed to build parlay: {exc}")
            return None


__all__ = ["ParlayOptimizer"]
