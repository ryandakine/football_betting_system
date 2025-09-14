# recommendation_engine.py

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

# We will import the data structures this engine needs to work with.
# Let's assume they are in a shared 'models.py' or defined in 'odds_fetcher.py'
from odds_fetcher import H2HBet, StructuredOdds

logger = logging.getLogger(__name__)


# --- Configuration for the Betting Strategy ---
@dataclass
class BettingConfig:
    """Holds all parameters for the betting strategy."""

    base_unit_size: float = 5.0  # The dollar value of 1 unit
    min_edge_threshold: float = 0.03  # Minimum 3% edge to consider a bet
    min_confidence: float = 0.60  # Minimum 60% AI confidence
    unit_rules: list[tuple[float, float, int]] = field(
        default_factory=lambda: [
            # (min_confidence, min_edge, units)
            (0.90, 0.25, 5),
            (0.85, 0.20, 4),
            (0.80, 0.15, 4),
            (0.75, 0.12, 3),
            (0.70, 0.10, 3),
            (0.65, 0.07, 2),
            (0.60, 0.05, 2),
            (0.60, 0.03, 1),
        ]
    )


# --- Structured Bet Recommendation ---
@dataclass
class FinalBet:
    """Represents a single, final bet recommendation after all analysis."""

    game_id: str
    selection: str
    odds: float
    bookmaker: str
    units: int
    stake: float
    confidence: float
    edge: float
    expected_value: float
    priority_score: float


class RecommendationEngine:
    """
    Analyzes odds and AI insights to generate quantifiable betting recommendations.
    """

    def __init__(self, config: BettingConfig = BettingConfig()):
        self.config = config
        logger.info("💡 RecommendationEngine initialized.")

    def _calculate_edge(self, true_probability: float, decimal_odds: float) -> float:
        if true_probability <= 0 or decimal_odds <= 1:
            return 0.0
        return (true_probability * decimal_odds) - 1

    def _determine_units(self, confidence: float, edge: float) -> int:
        if (
            confidence < self.config.min_confidence
            or edge < self.config.min_edge_threshold
        ):
            return 0
        for min_conf, min_edge, units in self.config.unit_rules:
            if confidence >= min_conf and edge >= min_edge:
                return units
        return 0

    def generate_recommendations(
        self, odds_data: StructuredOdds, ai_analysis: dict[str, Any]
    ) -> list[FinalBet]:
        """Processes structured odds and AI analysis to create a list of potential bets."""
        potential_bets = []

        # Process H2H (Moneyline) Bets
        for h2h_bet in odds_data.h2h_bets:
            analysis = ai_analysis.get(h2h_bet.game_id)
            if not analysis:
                continue

            consensus_confidence = analysis.get("consensus_confidence", 0.5)
            # This logic assumes the confidence score is for the favored team.
            # A more advanced version could have the AI specify which team it favors.

            edge = self._calculate_edge(consensus_confidence, h2h_bet.odds)
            units = self._determine_units(consensus_confidence, edge)

            if units > 0:
                stake = self.config.base_unit_size * units
                potential_bets.append(
                    FinalBet(
                        game_id=h2h_bet.game_id,
                        selection=h2h_bet.team,
                        odds=h2h_bet.odds,
                        bookmaker=h2h_bet.bookmaker,
                        units=units,
                        stake=stake,
                        confidence=consensus_confidence,
                        edge=edge,
                        expected_value=(stake * edge),
                        priority_score=(consensus_confidence * edge * units),
                    )
                )

        # TODO: Add logic here to process PlayerPropBet objects from odds_data.prop_bets

        logger.info(f"Generated {len(potential_bets)} potential H2H recommendations.")
        return potential_bets
