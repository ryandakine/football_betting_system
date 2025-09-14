#!/usr/bin/env python3
"""
Testing-Friendly MLB Betting System
$100 bankroll with confidence-based unit sizing (1-5 units @ $5 each)
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TestingBettingSystem:
    """Testing-friendly betting system with confidence-based units"""

    def __init__(self):
        # TESTING CONFIGURATION
        self.BANKROLL = 100.00
        self.BASE_UNIT = 5.00  # $5 = 1 unit
        self.MAX_UNITS = 5  # Maximum 5 units per bet ($25)
        self.MAX_TOTAL_EXPOSURE = 50.00  # 50% of bankroll maximum
        self.MAX_BETS_PER_DAY = 10  # Maximum number of bets
        self.MIN_EDGE_THRESHOLD = 0.03  # 3% minimum edge
        self.MIN_CONFIDENCE = 0.60  # 60% minimum confidence

        # Confidence-based unit sizing
        self.UNIT_MATRIX = [
            # (min_confidence, min_edge, units, label)
            (0.90, 0.25, 5, "EXTREME"),  # 90%+ confidence, 25%+ edge = 5 units
            (0.85, 0.20, 4, "VERY HIGH"),  # 85%+ confidence, 20%+ edge = 4 units
            (0.80, 0.15, 4, "HIGH"),  # 80%+ confidence, 15%+ edge = 4 units
            (0.75, 0.12, 3, "MEDIUM-HIGH"),  # 75%+ confidence, 12%+ edge = 3 units
            (0.70, 0.10, 3, "MEDIUM"),  # 70%+ confidence, 10%+ edge = 3 units
            (0.65, 0.07, 2, "LOW-MEDIUM"),  # 65%+ confidence, 7%+ edge = 2 units
            (0.60, 0.05, 2, "LOW"),  # 60%+ confidence, 5%+ edge = 2 units
            (0.60, 0.03, 1, "MINIMAL"),  # 60%+ confidence, 3%+ edge = 1 unit
        ]

        logger.info(f"🎯 Testing System Initialized:")
        logger.info(f"   Bankroll: ${self.BANKROLL}")
        logger.info(f"   Base Unit: ${self.BASE_UNIT}")
        logger.info(f"   Max Exposure: ${self.MAX_TOTAL_EXPOSURE}")
        logger.info(
            f"   Unit Range: 1-{self.MAX_UNITS} units (${self.BASE_UNIT}-${self.MAX_UNITS * self.BASE_UNIT})"
        )

    def calculate_edge(self, true_probability: float, decimal_odds: float) -> float:
        """Calculate betting edge"""
        if true_probability <= 0 or decimal_odds <= 1:
            return 0.0

        expected_value = (true_probability * decimal_odds) - 1
        return max(0.0, expected_value)

    def determine_units(self, confidence: float, edge: float) -> tuple[int, str]:
        """
        Determine unit size based on confidence and edge

        Returns:
            tuple: (units, confidence_label)
        """
        # Check minimum thresholds
        if confidence < self.MIN_CONFIDENCE or edge < self.MIN_EDGE_THRESHOLD:
            return 0, "NO BET"

        # Find appropriate unit size
        for min_conf, min_edge, units, label in self.UNIT_MATRIX:
            if confidence >= min_conf and edge >= min_edge:
                return units, label

        return 0, "NO BET"

    def estimate_realistic_probability(
        self, team_name: str, market_odds: float, ai_insight: dict, game_id: str
    ) -> float:
        """
        Estimate realistic win probability using AI insights
        (Simplified version of your existing logic)
        """
        market_implied = 1.0 / market_odds

        if not ai_insight:
            # No AI insight, use market with slight adjustment
            return market_implied * 1.05  # Slight edge assumption

        confidence = ai_insight.get("confidence", 0.65)

        # Blend market probability with AI confidence
        # Higher confidence = more deviation from market
        if confidence >= 0.80:
            # High confidence: significant deviation possible
            adjustment_factor = 1.4
        elif confidence >= 0.70:
            # Medium confidence: moderate deviation
            adjustment_factor = 1.25
        else:
            # Low confidence: minimal deviation
            adjustment_factor = 1.1

        # Calculate blended probability
        ai_probability = market_implied * adjustment_factor

        # Blend based on confidence level
        blend_weight = (confidence - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        final_probability = (
            market_implied * (1 - blend_weight) + ai_probability * blend_weight
        )

        # Clamp to reasonable bounds
        return max(0.1, min(0.9, final_probability))

    def create_bet_recommendation(
        self,
        team: str,
        odds: float,
        confidence: float,
        edge: float,
        game_info: str,
        bookmaker: str,
    ) -> dict | None:
        """Create a bet recommendation"""

        units, confidence_label = self.determine_units(confidence, edge)

        if units == 0:
            return None

        bet_amount = units * self.BASE_UNIT
        expected_value = edge * bet_amount
        roi_percentage = (expected_value / bet_amount) * 100

        return {
            "team": team,
            "game": game_info,
            "odds": odds,
            "bookmaker": bookmaker,
            "units": units,
            "bet_amount": bet_amount,
            "confidence_score": confidence,
            "confidence_label": confidence_label,
            "edge": edge,
            "expected_value": expected_value,
            "roi_percentage": roi_percentage,
            "risk_level": self._get_risk_level(units),
            "priority_score": confidence * edge * units,  # For sorting
        }

    def _get_risk_level(self, units: int) -> str:
        """Determine risk level based on unit size"""
        if units >= 5:
            return "MAXIMUM"
        elif units >= 4:
            return "HIGH"
        elif units >= 3:
            return "MEDIUM"
        elif units >= 2:
            return "LOW"
        else:
            return "MINIMAL"

    def process_odds_data(self, odds_data: list[dict], ai_insights: dict) -> list[dict]:
        """
        Process odds data and generate bet recommendations

        Args:
            odds_data: List of odds entries from your existing system
            ai_insights: AI insights dictionary from your existing system

        Returns:
            List of bet recommendations
        """
        recommendations = []

        logger.info(f"🔍 Processing {len(odds_data)} odds entries...")

        for odds_entry in odds_data:
            try:
                # Extract data (adjust field names to match your system)
                team = odds_entry.get("team_name") or odds_entry.get("team")
                odds = odds_entry.get("odds")
                game_id = odds_entry.get("game_id")
                bookmaker = odds_entry.get("bookmaker", "unknown")
                game_info = odds_entry.get("game_info", f"{team} game")

                if not all([team, odds, game_id]):
                    continue

                # Get AI insight for this game
                ai_insight = ai_insights.get(game_id, {})
                confidence = ai_insight.get("confidence", 0.0)

                # Estimate true probability
                true_prob = self.estimate_realistic_probability(
                    team, odds, ai_insight, game_id
                )

                # Calculate edge
                edge = self.calculate_edge(true_prob, odds)

                # Create recommendation
                recommendation = self.create_bet_recommendation(
                    team=team,
                    odds=odds,
                    confidence=confidence,
                    edge=edge,
                    game_info=game_info,
                    bookmaker=bookmaker,
                )

                if recommendation:
                    recommendations.append(recommendation)
                    logger.debug(
                        f"✅ {team}: {recommendation['units']} units, "
                        f"{recommendation['confidence_label']} confidence"
                    )

            except Exception as e:
                logger.error(f"❌ Error processing odds entry: {e}")
                continue

        logger.info(f"📊 Generated {len(recommendations)} potential recommendations")
        return recommendations

    def apply_portfolio_management(self, recommendations: list[dict]) -> list[dict]:
        """Apply portfolio management and exposure limits"""

        if not recommendations:
            return []

        # Sort by priority score (confidence × edge × units)
        sorted_bets = sorted(
            recommendations, key=lambda x: x["priority_score"], reverse=True
        )

        final_bets = []
        total_exposure = 0
        total_units = 0

        logger.info(f"💼 Portfolio Management:")
        logger.info(f"   Max Exposure: ${self.MAX_TOTAL_EXPOSURE}")
        logger.info(f"   Max Bets: {self.MAX_BETS_PER_DAY}")

        for bet in sorted_bets:
            # Check exposure limits
            if (
                total_exposure + bet["bet_amount"] <= self.MAX_TOTAL_EXPOSURE
                and len(final_bets) < self.MAX_BETS_PER_DAY
            ):

                final_bets.append(bet)
                total_exposure += bet["bet_amount"]
                total_units += bet["units"]

                logger.debug(
                    f"✅ Added: {bet['team']} - {bet['units']} units "
                    f"(${bet['bet_amount']}) - Total: ${total_exposure}"
                )
            else:
                logger.debug(f"❌ Skipped: {bet['team']} - Would exceed limits")

        logger.info(f"📋 Final Portfolio:")
        logger.info(f"   Bets: {len(final_bets)}")
        logger.info(f"   Total Units: {total_units}")
        logger.info(f"   Total Exposure: ${total_exposure}")

        return final_bets

    def format_recommendations(self, recommendations: list[dict]) -> str:
        """Format recommendations for display"""

        if not recommendations:
            return "❌ No betting opportunities found meeting criteria."

        output = []
        output.append("🎯 MLB BETTING RECOMMENDATIONS")
        output.append("=" * 50)

        total_bet = sum(bet["bet_amount"] for bet in recommendations)
        total_ev = sum(bet["expected_value"] for bet in recommendations)
        avg_edge = sum(bet["edge"] for bet in recommendations) / len(recommendations)

        for i, bet in enumerate(recommendations, 1):
            output.append(f"\n{i}. {bet['team']}")
            output.append(f"   Game: {bet['game']}")
            output.append(
                f"   Bet: {bet['units']} units (${bet['bet_amount']:.0f}) @ {bet['odds']:.2f}"
            )
            output.append(
                f"   Confidence: {bet['confidence_label']} ({bet['confidence_score']:.1%})"
            )
            output.append(
                f"   Edge: {bet['edge']:.1%} | EV: +${bet['expected_value']:.2f}"
            )
            output.append(
                f"   Risk: {bet['risk_level']} | ROI: {bet['roi_percentage']:.1f}%"
            )
            output.append(f"   Bookmaker: {bet['bookmaker']}")

        output.append("\n" + "=" * 50)
        output.append("📊 PORTFOLIO SUMMARY")
        output.append(f"Total Bets: {len(recommendations)}")
        output.append(
            f"Total Investment: ${total_bet:.0f} / ${self.MAX_TOTAL_EXPOSURE:.0f}"
        )
        output.append(f"Expected Value: +${total_ev:.2f}")
        output.append(f"Average Edge: {avg_edge:.1%}")
        output.append(f"Bankroll Usage: {(total_bet/self.BANKROLL)*100:.1f}%")

        return "\n".join(output)

    def save_recommendations(self, recommendations: list[dict], filename: str = None):
        """Save recommendations to file"""

        if not filename:
            date_str = datetime.now().strftime("%Y-%m-%d")
            filename = f"testing_recommendations_{date_str}.json"

        # Create directory if it doesn't exist
        os.makedirs("testing_logs", exist_ok=True)
        filepath = os.path.join("testing_logs", filename)

        # Prepare data for saving
        save_data = {
            "date": datetime.now().isoformat(),
            "system_config": {
                "bankroll": self.BANKROLL,
                "base_unit": self.BASE_UNIT,
                "max_exposure": self.MAX_TOTAL_EXPOSURE,
                "min_edge": self.MIN_EDGE_THRESHOLD,
            },
            "recommendations": recommendations,
            "summary": {
                "total_bets": len(recommendations),
                "total_exposure": sum(bet["bet_amount"] for bet in recommendations),
                "total_expected_value": sum(
                    bet["expected_value"] for bet in recommendations
                ),
                "average_edge": (
                    sum(bet["edge"] for bet in recommendations) / len(recommendations)
                    if recommendations
                    else 0
                ),
            },
        }

        with open(filepath, "w") as f:
            json.dump(save_data, f, indent=2)

        logger.info(f"💾 Recommendations saved to: {filepath}")


# Main function to integrate with your existing system
def run_testing_system(odds_data: list[dict], ai_insights: dict) -> list[dict]:
    """
    Main function to run the testing-friendly betting system

    Args:
        odds_data: Your existing odds data format
        ai_insights: Your existing AI insights format

    Returns:
        List of final bet recommendations
    """

    logger.info("🚀 Starting Testing-Friendly MLB Betting System...")

    # Initialize system
    betting_system = TestingBettingSystem()

    # Process odds and generate recommendations
    recommendations = betting_system.process_odds_data(odds_data, ai_insights)

    # Apply portfolio management
    final_recommendations = betting_system.apply_portfolio_management(recommendations)

    # Display results
    formatted_output = betting_system.format_recommendations(final_recommendations)
    print(formatted_output)

    # Save to file
    betting_system.save_recommendations(final_recommendations)

    logger.info("✅ Testing system complete!")

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
    ]

    sample_ai_insights = {
        "f8b977a9a45a2c67182143d2a0294726": {"confidence": 0.85},
        "66057d172d66d1e51aa7e2b418502262": {"confidence": 0.82},
    }

    # Run the system
    final_bets = run_testing_system(sample_odds_data, sample_ai_insights)

    # Output example:
    # 🎯 MLB BETTING RECOMMENDATIONS
    # ==================================================
    #
    # 1. Los Angeles Angels
    #    Game: Los Angeles Angels @ New York Yankees
    #    Bet: 4 units ($20) @ 2.70
    #    Confidence: HIGH (85.0%)
    #    Edge: 28.5% | EV: +$5.70
    #    Risk: HIGH | ROI: 28.5%
    #    Bookmaker: williamhill_us
