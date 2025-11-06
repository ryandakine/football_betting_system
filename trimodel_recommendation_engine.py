# trimodel_recommendation_engine.py
import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from advanced_components import RiskManager

# We import the structured data objects to ensure type safety
from enhanced_odds_fetcher import StructuredOdds

logger = logging.getLogger(__name__)


@dataclass
class EnhancedTriModelBet:
    """A final, detailed bet object ready for execution and tracking."""

    bet_id: str
    game_id: str
    selection: str
    odds: float
    bookmaker: str
    stake: float
    units: int
    edge: float
    expected_value: float
    ai_confidence: float
    quality_score: float
    risk_level: str
    supporting_ais: list[str] = field(default_factory=list)


class AdvancedTriModelRecommendationEngine:
    """
    Takes consensus AI analysis and live market odds to generate a final,
    optimized betting portfolio.
    """

    def __init__(self, bankroll: float, config: dict):
        self.bankroll = bankroll
        self.config = config
        self.risk_manager = RiskManager(bankroll=self.bankroll)
        self.unit_size = config.get("base_unit_size", 5.0)
        logger.info("💡 Advanced Recommendation Engine Initialized with Kelly Sizing.")

    def _calculate_edge(self, true_probability: float, odds: float) -> float:
        if true_probability <= 0 or odds <= 1:
            return 0.0
        return (true_probability * odds) - 1

    def _determine_risk_level(self, odds: float, quality_score: float) -> str:
        if odds > 2.8 or quality_score < 0.65:
            return "high"
        if odds > 2.2 or quality_score < 0.75:
            return "medium"
        return "low"

    def generate_comprehensive_recommendations(
        self, odds_data: StructuredOdds, ai_analysis: dict
    ) -> list[EnhancedTriModelBet]:
        recommendations = []

        for game_id, analysis in ai_analysis.items():
            if "error" in analysis or not analysis.get("favored_team"):
                continue

            confidence = analysis.get("consensus_confidence", 0.5)
            quality_score = analysis.get("quality_score", 0.5)

            best_odds_info = self._get_best_odds(
                game_id, analysis["favored_team"], odds_data
            )
            if not best_odds_info:
                continue

            edge = self._calculate_edge(confidence, best_odds_info["odds"])

            min_conf = self.config.get("min_consensus_confidence", 0.65)
            min_qual = self.config.get("min_quality_score", 0.70)
            min_edge = self.config.get("min_edge_percentage", 0.03)

            if confidence < min_conf or quality_score < min_qual or edge < min_edge:
                continue

            # Use Kelly Criterion for scientific position sizing
            stake_fraction = self.risk_manager.calculate_kelly_criterion(
                win_prob=confidence,
                odds=best_odds_info["odds"],
            )

            # Convert stake fraction to a dollar amount
            stake_amount = self.bankroll * stake_fraction

            if stake_amount >= self.unit_size:  # Only consider bets of at least 1 unit
                units = max(1, round(stake_amount / self.unit_size))
                final_stake = units * self.unit_size

                recommendations.append(
                    EnhancedTriModelBet(
                        bet_id=f"{game_id}_{analysis['favored_team']}",
                        game_id=game_id,
                        selection=analysis["favored_team"],
                        odds=best_odds_info["odds"],
                        bookmaker=best_odds_info["bookmaker"],
                        stake=final_stake,
                        units=units,
                        edge=edge,
                        expected_value=(final_stake * edge),
                        ai_confidence=confidence,
                        quality_score=quality_score,
                        risk_level=self._determine_risk_level(
                            best_odds_info["odds"], quality_score
                        ),
                        supporting_ais=[
                            res.get("source")
                            for res in analysis.get("individual_analyses", [])
                        ],
                    )
                )

        logger.info(
            f"Generated {len(recommendations)} high-quality, Kelly-sized recommendations."
        )
        return sorted(recommendations, key=lambda x: x.expected_value, reverse=True)

    def _get_best_odds(
        self, game_id: str, team: str, odds_data: StructuredOdds
    ) -> dict | None:
        best_odds, best_bookmaker = 0, None
        for bet in odds_data.h2h_bets:
            if bet.game_id == game_id and bet.team == team and bet.odds > best_odds:
                best_odds = bet.odds
                best_bookmaker = bet.bookmaker

        if best_bookmaker:
            return {"odds": best_odds, "bookmaker": best_bookmaker}
        return None
