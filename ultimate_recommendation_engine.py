#!/usr/bin/env python3
"""
Ultimate Recommendation Engine v4.1
"""
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


@dataclass
class FinalBet:
    """A final, actionable bet ready for execution or tracking."""

    game_id: str
    selection: str
    odds: float
    bookmaker: str
    stake: float
    units: int
    edge: float
    ai_confidence: float
    primary_signal: str
    supporting_ais: list[str] = field(default_factory=list)


class UltimateRecommendationEngine:
    """
    Takes consensus AI analysis and live market odds to generate a final,
    optimized betting portfolio.
    """

    def __init__(self, unit_size: float = 5.0, confidence_threshold: float = 0.65):
        self.unit_size = unit_size
        self.confidence_threshold = confidence_threshold
        logger.info("💡 Ultimate Recommendation Engine Initialized.")

    def _get_best_odds(self, game_id: str, team: str, odds_data: list[dict]) -> dict:
        """Finds the best available odds for a specific team in a game's H2H market."""
        best_odds, best_bookmaker = 0, "N/A"
        for bet in odds_data:
            if (
                bet.get("game_id") == game_id
                and bet.get("team") == team
                and bet.get("odds", 0) > best_odds
            ):
                best_odds = bet["odds"]
                best_bookmaker = bet.get("bookmaker", "N/A")
        return {"odds": best_odds, "bookmaker": best_bookmaker}

    def _calculate_units(self, confidence: float) -> int:
        """Determines bet size in units based on AI confidence."""
        if confidence >= 0.78:
            return 3
        if confidence >= 0.72:
            return 2
        if confidence >= self.confidence_threshold:
            return 1
        return 0

    def generate_portfolio(
        self, analyses: list[dict], odds_data: dict
    ) -> list[FinalBet]:
        """Generates the final list of bets to be placed."""
        portfolio = []
        h2h_odds = odds_data.get("h2h", [])

        for analysis in analyses:
            if "error" in analysis or not analysis.get("favored_team"):
                continue

            confidence = analysis.get("consensus_confidence", 0.5)
            units = self._calculate_units(confidence)

            if units > 0:
                favored_team = analysis["favored_team"]
                market_info = self._get_best_odds(
                    analysis["game_id"], favored_team, h2h_odds
                )

                if market_info["odds"] > 1.0:
                    edge = (confidence * market_info["odds"]) - 1
                    if edge > 0.03:
                        stake = self.unit_size * units
                        portfolio.append(
                            FinalBet(
                                game_id=analysis["game_id"],
                                selection=favored_team,
                                odds=market_info["odds"],
                                bookmaker=market_info["bookmaker"],
                                stake=stake,
                                units=units,
                                edge=edge,
                                ai_confidence=confidence,
                                primary_signal=analysis.get(
                                    "primary_signal", "Consensus"
                                ),
                                supporting_ais=[
                                    res.get("provider")
                                    for res in analysis.get("individual_analyses", [])
                                    if "error" not in res
                                ],
                            )
                        )

        logger.info(f"✅ Generated a portfolio of {len(portfolio)} bets.")
        return sorted(portfolio, key=lambda x: x.ai_confidence * x.edge, reverse=True)
