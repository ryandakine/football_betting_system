#!/usr/bin/env python3
"""
Improved MLB Betting System using Dataclasses for robust configuration and data handling.
$100 bankroll with confidence-based unit sizing (1-5 units @ $5 each).
"""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# --- Data Structures ---
# Using dataclasses makes our data structures explicit, typed, and easier to work with.


@dataclass
class UnitRule:
    """Defines a single rule for determining bet units."""

    min_confidence: float
    min_edge: float
    units: int
    label: str


@dataclass
class BetRecommendation:
    """Holds all information for a single, structured bet recommendation."""

    team: str
    game: str
    odds: float
    bookmaker: str
    units: int
    bet_amount: float
    confidence_score: float
    confidence_label: str
    edge: float
    expected_value: float
    roi_percentage: float
    risk_level: str
    priority_score: float  # For sorting bets


@dataclass
class BettingConfig:
    """Groups all system configuration into a single object."""

    bankroll: float = 100.00
    base_unit: float = 5.00
    max_units: int = 5
    max_total_exposure: float = 50.00
    max_bets_per_day: int = 10
    min_edge_threshold: float = 0.03
    min_confidence: float = 0.60

    # Use a default_factory to correctly initialize the mutable list
    unit_matrix: list[UnitRule] = field(
        default_factory=lambda: [
            UnitRule(0.90, 0.25, 5, "EXTREME"),
            UnitRule(0.85, 0.20, 4, "VERY HIGH"),
            UnitRule(0.80, 0.15, 4, "HIGH"),
            UnitRule(0.75, 0.12, 3, "MEDIUM-HIGH"),
            UnitRule(0.70, 0.10, 3, "MEDIUM"),
            UnitRule(0.65, 0.07, 2, "LOW-MEDIUM"),
            UnitRule(0.60, 0.05, 2, "LOW"),
            UnitRule(0.60, 0.03, 1, "MINIMAL"),
        ]
    )


# --- Core Betting Logic ---


class TestingBettingSystem:
    """
    Testing-friendly betting system that operates on a given configuration.
    This class is now decoupled from the configuration itself.
    """

    def __init__(self, config: BettingConfig):
        """
        Initializes the betting system with a specific configuration.

        Args:
            config: A BettingConfig object containing all system parameters.
        """
        self.config = config

        logger.info("🎯 Testing System Initialized:")
        logger.info(f"  Bankroll: ${self.config.bankroll}")
        logger.info(f"  Base Unit: ${self.config.base_unit}")
        logger.info(f"  Max Exposure: ${self.config.max_total_exposure}")
        unit_range_display = f"${self.config.base_unit * 1:.2f}-${self.config.max_units * self.config.base_unit:.2f}"
        logger.info(
            f"  Unit Range: 1-{self.config.max_units} units ({unit_range_display})"
        )

    def calculate_edge(self, true_probability: float, decimal_odds: float) -> float:
        """
        Calculates the betting edge, which represents the expected profit margin.
        Formula: (True Probability * Decimal Odds) - 1
        """
        if true_probability <= 0 or decimal_odds <= 1:
            return 0.0

        expected_value = (true_probability * decimal_odds) - 1
        return max(0.0, expected_value)

    def determine_units(self, confidence: float, edge: float) -> tuple[int, str]:
        """
        Determines unit size by finding the first matching rule in the unit matrix.

        Returns:
            A tuple of (units, confidence_label).
        """
        if (
            confidence < self.config.min_confidence
            or edge < self.config.min_edge_threshold
        ):
            return 0, "NO BET"

        # Iterate through the structured rules for better readability
        for rule in self.config.unit_matrix:
            if confidence >= rule.min_confidence and edge >= rule.min_edge:
                return rule.units, rule.label

        return 0, "NO BET"

    def estimate_realistic_probability(
        self, market_odds: float, ai_insight: dict
    ) -> float:
        """
        Estimates a realistic win probability by blending market data with AI insights.
        This is a placeholder for a more complex model.
        """
        market_implied = 1.0 / market_odds

        if not ai_insight:
            # Fallback if no AI insight is available: assume a small, fixed edge.
            return market_implied * 1.05

        confidence = ai_insight.get("confidence", 0.65)

        # Determine how much to adjust market probability based on AI confidence
        if confidence >= 0.80:
            adjustment_factor = 1.4  # High confidence allows significant deviation
        elif confidence >= 0.70:
            adjustment_factor = 1.25  # Medium confidence, moderate deviation
        else:
            adjustment_factor = 1.1  # Low confidence, minimal deviation

        ai_probability = market_implied * adjustment_factor

        # Create a weighted blend between market and AI probability.
        # A higher confidence score gives more weight to the AI's adjusted probability.
        # This scales confidence from [0.5, 1.0] to a weight of [0, 1.0].
        blend_weight = max(0, (confidence - 0.5) * 2)

        final_probability = (
            market_implied * (1 - blend_weight) + ai_probability * blend_weight
        )

        # Clamp the result to reasonable probability bounds (e.g., 10% to 90%)
        return max(0.1, min(0.9, final_probability))

    def create_bet_recommendation(
        self,
        team: str,
        odds: float,
        confidence: float,
        edge: float,
        game_info: str,
        bookmaker: str,
    ) -> BetRecommendation | None:
        """Creates a structured BetRecommendation object if the bet meets criteria."""

        units, confidence_label = self.determine_units(confidence, edge)
        if units == 0:
            return None

        bet_amount = units * self.config.base_unit

        # The edge is the direct Return on Investment (ROI)
        roi_percentage = edge * 100
        expected_value = edge * bet_amount

        return BetRecommendation(
            team=team,
            game=game_info,
            odds=odds,
            bookmaker=bookmaker,
            units=units,
            bet_amount=bet_amount,
            confidence_score=confidence,
            confidence_label=confidence_label,
            edge=edge,
            expected_value=expected_value,
            roi_percentage=roi_percentage,
            risk_level=self._get_risk_level(units),
            priority_score=(confidence * edge * units),  # Key metric for sorting
        )

    def _get_risk_level(self, units: int) -> str:
        """Determines a human-readable risk level based on unit size."""
        if units >= 5:
            return "MAXIMUM"
        if units >= 4:
            return "HIGH"
        if units >= 3:
            return "MEDIUM"
        if units >= 2:
            return "LOW"
        return "MINIMAL"

    def process_odds_data(
        self, odds_data: list[dict], ai_insights: dict
    ) -> list[BetRecommendation]:
        """
        Processes a list of odds and generates potential bet recommendations.
        """
        recommendations = []
        logger.info(f"🔍 Processing {len(odds_data)} odds entries...")

        for odds_entry in odds_data:
            try:
                team = odds_entry.get("team_name") or odds_entry.get("team")
                odds = odds_entry.get("odds")
                game_id = odds_entry.get("game_id")

                if not all([team, odds, game_id]):
                    logger.warning(f"Skipping incomplete odds entry: {odds_entry}")
                    continue

                ai_insight = ai_insights.get(game_id, {})
                confidence = ai_insight.get("confidence", 0.0)

                true_prob = self.estimate_realistic_probability(odds, ai_insight)
                edge = self.calculate_edge(true_prob, odds)

                recommendation = self.create_bet_recommendation(
                    team=team,
                    odds=odds,
                    confidence=confidence,
                    edge=edge,
                    game_info=odds_entry.get("game_info", f"{team} Game"),
                    bookmaker=odds_entry.get("bookmaker", "Unknown"),
                )

                if recommendation:
                    recommendations.append(recommendation)
                    logger.debug(
                        f"✅ {team}: {recommendation.units} units, {recommendation.confidence_label}"
                    )

            except Exception as e:
                logger.error(f"❌ Error processing odds entry {odds_entry}: {e}")
                continue

        logger.info(f"📊 Generated {len(recommendations)} potential recommendations")
        return recommendations

    def apply_portfolio_management(
        self, recommendations: list[BetRecommendation]
    ) -> list[BetRecommendation]:
        """Applies bankroll management rules to a list of potential bets."""

        if not recommendations:
            return []

        # Sort by priority score to consider the highest value bets first
        sorted_bets = sorted(
            recommendations, key=lambda x: x.priority_score, reverse=True
        )

        final_bets = []
        total_exposure = 0.0

        logger.info("💼 Applying Portfolio Management...")

        for bet in sorted_bets:
            if (
                len(final_bets) < self.config.max_bets_per_day
                and total_exposure + bet.bet_amount <= self.config.max_total_exposure
            ):

                final_bets.append(bet)
                total_exposure += bet.bet_amount
                logger.debug(
                    f"✅ Added: {bet.team} ({bet.units} units) - New Exposure: ${total_exposure:.2f}"
                )
            else:
                logger.debug(f"❌ Skipped: {bet.team} - Would exceed portfolio limits.")

        logger.info("📋 Final Portfolio Summary:")
        logger.info(f"  Bets: {len(final_bets)}")
        logger.info(f"  Total Units: {sum(b.units for b in final_bets)}")
        logger.info(f"  Total Exposure: ${total_exposure:.2f}")

        return final_bets


# --- Output and Execution ---


def format_recommendations_for_display(
    recommendations: list[BetRecommendation], config: BettingConfig
) -> str:
    """Formats the final recommendations into a human-readable string."""

    if not recommendations:
        return "❌ No betting opportunities found meeting criteria."

    total_bet = sum(bet.bet_amount for bet in recommendations)
    total_ev = sum(bet.expected_value for bet in recommendations)
    avg_edge = sum(bet.edge for bet in recommendations) / len(recommendations)
    bankroll_usage = (total_bet / config.bankroll) * 100

    output = ["🎯 MLB BETTING RECOMMENDATIONS", "=" * 50]

    for i, bet in enumerate(recommendations, 1):
        output.extend(
            [
                f"\n{i}. {bet.team}",
                f"   Game: {bet.game}",
                f"   Bet: {bet.units} units (${bet.bet_amount:.2f}) @ {bet.odds:.2f}",
                f"   Confidence: {bet.confidence_label} ({bet.confidence_score:.1%})",
                f"   Edge: {bet.edge:.1%} | EV: +${bet.expected_value:.2f}",
                f"   Risk: {bet.risk_level} | ROI: {bet.roi_percentage:.1f}%",
                f"   Bookmaker: {bet.bookmaker}",
            ]
        )

    output.extend(
        [
            "\n" + "=" * 50,
            "📊 PORTFOLIO SUMMARY",
            f"Total Bets: {len(recommendations)}",
            f"Total Investment: ${total_bet:.2f} / ${config.max_total_exposure:.2f}",
            f"Total Expected Value (EV): +${total_ev:.2f}",
            f"Average Edge: {avg_edge:.1%}",
            f"Bankroll Usage: {bankroll_usage:.1f}%",
        ]
    )

    return "\n".join(output)


def save_recommendations_to_file(
    recommendations: list[BetRecommendation],
    config: BettingConfig,
    filename: str = None,
):
    """Saves the final recommendations and system config to a JSON file."""
    if not filename:
        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"testing_recommendations_{date_str}.json"

    log_dir = "testing_logs"
    os.makedirs(log_dir, exist_ok=True)
    filepath = os.path.join(log_dir, filename)

    # Use asdict to convert dataclasses to dictionaries for JSON serialization
    save_data = {
        "date": datetime.now().isoformat(),
        "system_config": asdict(config),
        "recommendations": [asdict(rec) for rec in recommendations],
        "summary": {
            "total_bets": len(recommendations),
            "total_exposure": sum(bet.bet_amount for bet in recommendations),
            "total_expected_value": sum(bet.expected_value for bet in recommendations),
            "average_edge": (
                (sum(r.edge for r in recommendations) / len(recommendations))
                if recommendations
                else 0
            ),
        },
    }

    with open(filepath, "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"💾 Recommendations saved to: {filepath}")


def run_testing_system(
    odds_data: list[dict], ai_insights: dict
) -> list[BetRecommendation]:
    """
    Main function to initialize and run the betting system.

    This function now sets up the configuration and orchestrates the process.
    """
    logger.info("🚀 Starting Improved MLB Betting System...")

    # 1. Initialize the configuration
    config = BettingConfig()

    # 2. Initialize the system with the configuration
    betting_system = TestingBettingSystem(config)

    # 3. Process data to get potential bets
    potential_recommendations = betting_system.process_odds_data(odds_data, ai_insights)

    # 4. Apply portfolio management to get final bets
    final_recommendations = betting_system.apply_portfolio_management(
        potential_recommendations
    )

    # 5. Display and save results
    formatted_output = format_recommendations_for_display(final_recommendations, config)
    print(formatted_output)

    save_recommendations_to_file(final_recommendations, config)

    logger.info("✅ System run complete!")
    return final_recommendations


# Example usage with your existing data structure
if __name__ == "__main__":

    # Example data (replace with your actual data)
    sample_odds_data = [
        {
            "team_name": "Los Angeles Angels",
            "odds": 2.70,
            "game_id": "f8b977a9a45a2c67182143d2a0294726",
            "bookmaker": "williamhill_us",
            "game_info": "Los Angeles Angels @ New York Yankees",
        },
        {
            "team_name": "Pittsburgh Pirates",
            "odds": 2.80,
            "game_id": "66057d172d66d1e51aa7e2b418502262",
            "bookmaker": "lowvig",
            "game_info": "Pittsburgh Pirates @ Detroit Tigers",
        },
        {  # Add another bet to test portfolio management
            "team_name": "Arizona Diamondbacks",
            "odds": 3.50,
            "game_id": "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6",
            "bookmaker": "draftkings",
            "game_info": "Arizona Diamondbacks @ San Francisco Giants",
        },
    ]

    sample_ai_insights = {
        "f8b977a9a45a2c67182143d2a0294726": {"confidence": 0.85},
        "66057d172d66d1e51aa7e2b418502262": {"confidence": 0.82},
        "a1b2c3d4e5f6g7h8i9j0k1l2m3n4o5p6": {"confidence": 0.91},
    }

    # Run the system
    final_bets = run_testing_system(sample_odds_data, sample_ai_insights)
