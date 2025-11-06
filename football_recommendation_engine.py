#!/usr/bin/env python3
"""
Football Recommendation Engine - NFL & College Football
Adapted from MLB system for football-specific markets and betting strategies.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Import football-specific data structures
from football_odds_fetcher import H2HBet, PlayerPropBet, SpreadBet, StructuredOdds, TotalBet

logger = logging.getLogger(__name__)


@dataclass
class FinalBet:
    """Final betting recommendation for football."""

    game_id: str
    home_team: str
    away_team: str
    bet_type: str  # "moneyline", "spread", "total", "player_prop"
    selection: str  # "home", "away", "over", "under", or player name
    odds: float
    stake: float
    expected_value: float
    confidence: float
    reasoning: str
    risk_level: str
    market_type: str
    bookmaker: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            "game_id": self.game_id,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "bet_type": self.bet_type,
            "selection": self.selection,
            "odds": self.odds,
            "stake": self.stake,
            "expected_value": self.expected_value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "risk_level": self.risk_level,
            "market_type": self.market_type,
            "bookmaker": self.bookmaker,
            "timestamp": self.timestamp,
        }


@dataclass
class FootballBettingConfig:
    """Configuration for football betting strategy."""

    base_unit_size: float = 5.0  # The dollar value of 1 unit
    min_edge_threshold: float = 0.03  # Minimum 3% edge to consider a bet
    min_confidence: float = 0.60  # Minimum 60% AI confidence
    max_exposure_pct: float = 0.10  # Maximum 10% of bankroll exposed
    kelly_fraction: float = 0.25  # Conservative Kelly fraction

    # Football-specific thresholds
    min_spread_edge: float = 0.02  # Minimum edge for spread bets
    min_total_edge: float = 0.025  # Minimum edge for total bets
    min_prop_edge: float = 0.04  # Minimum edge for player props

    # Unit sizing rules for different bet types
    unit_rules: List[Tuple[float, float, int]] = field(
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


class FootballRecommendationEngine:
    """Generates football betting recommendations with Kelly Criterion."""

    def __init__(self, bankroll: float, config: FootballBettingConfig | None = None):
        self.bankroll = bankroll
        self.config = config or FootballBettingConfig()
        self.max_exposure = bankroll * self.config.max_exposure_pct

        logger.info("🏈 Football Recommendation Engine initialized")
        logger.info(f"   Bankroll: ${bankroll:,.2f}")
        logger.info(f"   Max exposure: ${self.max_exposure:,.2f}")
        logger.info(f"   Base unit: ${self.config.base_unit_size:,.2f}")

    def generate_recommendations(
        self, odds_data: StructuredOdds, ai_analysis: Dict
    ) -> List[FinalBet]:
        """Generate comprehensive betting recommendations."""
        recommendations = []

        # Process moneyline bets
        moneyline_recs = self._analyze_moneyline_bets(odds_data.h2h_bets, ai_analysis)
        recommendations.extend(moneyline_recs)

        # Process spread bets
        spread_recs = self._analyze_spread_bets(odds_data.spread_bets, ai_analysis)
        recommendations.extend(spread_recs)

        # Process total bets
        total_recs = self._analyze_total_bets(odds_data.total_bets, ai_analysis)
        recommendations.extend(total_recs)

        # Process player props
        prop_recs = self._analyze_player_props(odds_data.prop_bets, ai_analysis)
        recommendations.extend(prop_recs)

        # Sort by expected value and apply portfolio optimization
        recommendations.sort(key=lambda x: x.expected_value, reverse=True)
        optimized_recs = self._optimize_portfolio(recommendations)

        logger.info(f"🏈 Generated {len(optimized_recs)} optimized recommendations")
        return optimized_recs

    def _analyze_moneyline_bets(self, h2h_bets: List[H2HBet], ai_analysis: Dict) -> List[FinalBet]:
        """Analyze moneyline betting opportunities."""
        recommendations = []

        for bet in h2h_bets:
            # Find AI analysis for this game
            ai_rec = self._find_ai_recommendation(bet.game_id, ai_analysis)
            if not ai_rec:
                continue

            # Calculate implied probabilities
            home_implied = 1 / bet.home_odds
            away_implied = 1 / bet.away_odds

            # Get AI probability (assuming it's for home team)
            ai_prob = ai_rec.get("confidence_score", 0.5)

            # Calculate edges
            home_edge = ai_prob - home_implied
            away_edge = (1 - ai_prob) - away_implied

            # Determine best bet
            if home_edge > away_edge and home_edge > self.config.min_edge_threshold:
                recommendations.append(
                    self._create_moneyline_bet(
                        bet, "home", bet.home_odds, home_edge, ai_prob, ai_rec
                    )
                )
            elif away_edge > self.config.min_edge_threshold:
                recommendations.append(
                    self._create_moneyline_bet(
                        bet, "away", bet.away_odds, away_edge, 1 - ai_prob, ai_rec
                    )
                )

        return recommendations

    def _analyze_spread_bets(
        self, spread_bets: List[SpreadBet], ai_analysis: Dict
    ) -> List[FinalBet]:
        """Analyze point spread betting opportunities."""
        recommendations = []

        for bet in spread_bets:
            ai_rec = self._find_ai_recommendation(bet.game_id, ai_analysis)
            if not ai_rec:
                continue

            # For spreads, we need to estimate the probability of covering
            # This is a simplified approach - in practice you'd want more sophisticated modeling
            ai_prob = ai_rec.get("confidence_score", 0.5)

            # Adjust for spread (simplified)
            spread_adjustment = bet.home_spread * 0.03  # Rough adjustment
            adjusted_prob = ai_prob - spread_adjustment

            # Calculate edges
            home_edge = adjusted_prob - (1 / bet.home_odds)
            away_edge = (1 - adjusted_prob) - (1 / bet.away_odds)

            if home_edge > self.config.min_spread_edge:
                recommendations.append(
                    self._create_spread_bet(
                        bet, "home", bet.home_odds, home_edge, adjusted_prob, ai_rec
                    )
                )
            elif away_edge > self.config.min_spread_edge:
                recommendations.append(
                    self._create_spread_bet(
                        bet, "away", bet.away_odds, away_edge, 1 - adjusted_prob, ai_rec
                    )
                )

        return recommendations

    def _analyze_total_bets(self, total_bets: List[TotalBet], ai_analysis: Dict) -> List[FinalBet]:
        """Analyze over/under total betting opportunities."""
        recommendations = []

        for bet in total_bets:
            ai_rec = self._find_ai_recommendation(bet.game_id, ai_analysis)
            if not ai_rec:
                continue

            # For totals, we need to estimate scoring probability
            # This is simplified - you'd want actual scoring models
            ai_prob = ai_rec.get("confidence_score", 0.5)

            # Assume higher confidence = higher scoring (simplified)
            over_prob = ai_prob
            under_prob = 1 - ai_prob

            # Calculate edges
            over_edge = over_prob - (1 / bet.over_odds)
            under_edge = under_prob - (1 / bet.under_odds)

            if over_edge > self.config.min_total_edge:
                recommendations.append(
                    self._create_total_bet(bet, "over", bet.over_odds, over_edge, over_prob, ai_rec)
                )
            elif under_edge > self.config.min_total_edge:
                recommendations.append(
                    self._create_total_bet(
                        bet, "under", bet.under_odds, under_edge, under_prob, ai_rec
                    )
                )

        return recommendations

    def _analyze_player_props(
        self, prop_bets: List[PlayerPropBet], ai_analysis: Dict
    ) -> List[FinalBet]:
        """Analyze player prop betting opportunities."""
        recommendations = []

        for bet in prop_bets:
            # For player props, we'd need specific player analysis
            # This is a placeholder - you'd want player-specific models
            ai_rec = self._find_ai_recommendation(bet.game_id, ai_analysis)
            if not ai_rec:
                continue

            # Simplified analysis - assume 50/50 for props without specific data
            over_prob = 0.5
            under_prob = 0.5

            # Calculate edges
            over_edge = over_prob - (1 / bet.over_odds)
            under_edge = under_prob - (1 / bet.under_odds)

            if over_edge > self.config.min_prop_edge:
                recommendations.append(
                    self._create_prop_bet(bet, "over", bet.over_odds, over_edge, over_prob, ai_rec)
                )
            elif under_edge > self.config.min_prop_edge:
                recommendations.append(
                    self._create_prop_bet(
                        bet, "under", bet.under_odds, under_edge, under_prob, ai_rec
                    )
                )

        return recommendations

    def _find_ai_recommendation(self, game_id: str, ai_analysis: Dict) -> Optional[Dict]:
        """Find AI recommendation for a specific game."""
        recommendations = ai_analysis.get("recommendations", [])
        for rec in recommendations:
            if rec.get("game_id") == game_id:
                return rec
        return None

    def _create_moneyline_bet(
        self,
        bet: H2HBet,
        selection: str,
        odds: float,
        edge: float,
        probability: float,
        ai_rec: Dict,
    ) -> FinalBet:
        """Create a moneyline betting recommendation."""
        stake = self._calculate_kelly_stake(edge, odds, probability)

        return FinalBet(
            game_id=bet.game_id,
            home_team=bet.home_team,
            away_team=bet.away_team,
            bet_type="moneyline",
            selection=selection,
            odds=odds,
            stake=stake,
            expected_value=edge * stake,
            confidence=probability,
            reasoning=ai_rec.get("reasoning", "AI analysis"),
            risk_level=self._assess_risk(probability, edge),
            market_type="moneyline",
            bookmaker=bet.bookmaker,
        )

    def _create_spread_bet(
        self,
        bet: SpreadBet,
        selection: str,
        odds: float,
        edge: float,
        probability: float,
        ai_rec: Dict,
    ) -> FinalBet:
        """Create a spread betting recommendation."""
        stake = self._calculate_kelly_stake(edge, odds, probability)

        return FinalBet(
            game_id=bet.game_id,
            home_team=bet.home_team,
            away_team=bet.away_team,
            bet_type="spread",
            selection=selection,
            odds=odds,
            stake=stake,
            expected_value=edge * stake,
            confidence=probability,
            reasoning=ai_rec.get("reasoning", "AI analysis"),
            risk_level=self._assess_risk(probability, edge),
            market_type="spread",
            bookmaker=bet.bookmaker,
        )

    def _create_total_bet(
        self,
        bet: TotalBet,
        selection: str,
        odds: float,
        edge: float,
        probability: float,
        ai_rec: Dict,
    ) -> FinalBet:
        """Create a total betting recommendation."""
        stake = self._calculate_kelly_stake(edge, odds, probability)

        return FinalBet(
            game_id=bet.game_id,
            home_team=bet.home_team,
            away_team=bet.away_team,
            bet_type="total",
            selection=selection,
            odds=odds,
            stake=stake,
            expected_value=edge * stake,
            confidence=probability,
            reasoning=ai_rec.get("reasoning", "AI analysis"),
            risk_level=self._assess_risk(probability, edge),
            market_type="total",
            bookmaker=bet.bookmaker,
        )

    def _create_prop_bet(
        self,
        bet: PlayerPropBet,
        selection: str,
        odds: float,
        edge: float,
        probability: float,
        ai_rec: Dict,
    ) -> FinalBet:
        """Create a player prop betting recommendation."""
        stake = self._calculate_kelly_stake(edge, odds, probability)

        return FinalBet(
            game_id=bet.game_id,
            home_team="",  # Props don't have home/away teams
            away_team="",
            bet_type="player_prop",
            selection=f"{bet.player_name} {selection}",
            odds=odds,
            stake=stake,
            expected_value=edge * stake,
            confidence=probability,
            reasoning=ai_rec.get("reasoning", "AI analysis"),
            risk_level=self._assess_risk(probability, edge),
            market_type="player_prop",
            bookmaker=bet.bookmaker,
        )

    def _calculate_kelly_stake(self, edge: float, odds: float, probability: float) -> float:
        """Calculate Kelly Criterion stake size."""
        if edge <= 0:
            return 0.0

        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - probability
        b = odds - 1
        p = probability
        q = 1 - probability

        kelly_fraction = (b * p - q) / b

        # Apply conservative fraction
        kelly_fraction *= self.config.kelly_fraction

        # Convert to dollar amount
        stake = kelly_fraction * self.bankroll

        # Apply unit sizing rules
        units = self._calculate_units(probability, edge)
        unit_stake = units * self.config.base_unit_size

        # Use the smaller of Kelly or unit sizing
        return min(stake, unit_stake)

    def _calculate_units(self, confidence: float, edge: float) -> int:
        """Calculate number of units based on confidence and edge."""
        for min_conf, min_edge, units in self.config.unit_rules:
            if confidence >= min_conf and edge >= min_edge:
                return units
        return 0

    def _assess_risk(self, confidence: float, edge: float) -> str:
        """Assess risk level of a bet."""
        if confidence >= 0.8 and edge >= 0.15:
            return "low"
        elif confidence >= 0.65 and edge >= 0.08:
            return "medium"
        else:
            return "high"

    def _optimize_portfolio(self, recommendations: List[FinalBet]) -> List[FinalBet]:
        """Optimize the betting portfolio within exposure limits."""
        optimized = []
        total_exposure = 0.0

        for rec in recommendations:
            if total_exposure + rec.stake <= self.max_exposure:
                optimized.append(rec)
                total_exposure += rec.stake
            else:
                # Reduce stake to fit within limits
                remaining_exposure = self.max_exposure - total_exposure
                if remaining_exposure > 0:
                    old_stake = rec.stake
                    rec.stake = remaining_exposure
                    rec.expected_value = rec.expected_value * (remaining_exposure / old_stake)
                    optimized.append(rec)
                break

        logger.info(
            f"🏈 Portfolio optimized: {len(optimized)} bets, ${total_exposure:,.2f} exposure"
        )
        return optimized


def main():
    """Test the football recommendation engine."""
    # Sample data
    config = FootballBettingConfig()
    engine = FootballRecommendationEngine(bankroll=1000.0, config=config)

    # Sample odds data
    odds_data = StructuredOdds(
        games=[],
        h2h_bets=[
            H2HBet(
                game_id="BUF_NE_20250101",
                home_team="Buffalo Bills",
                away_team="New England Patriots",
                home_odds=1.85,
                away_odds=2.05,
                bookmaker="fanduel",
                last_update="2025-01-01T12:00:00Z",
            )
        ],
        spread_bets=[],
        total_bets=[],
        prop_bets=[],
    )

    # Sample AI analysis
    ai_analysis = {
        "recommendations": [
            {
                "game_id": "BUF_NE_20250101",
                "confidence_score": 0.75,
                "reasoning": "Bills have strong home field advantage and recent form",
                "key_factors": ["home_field", "recent_form", "weather"],
            }
        ]
    }

    recommendations = engine.generate_recommendations(odds_data, ai_analysis)

    print(f"🏈 Generated {len(recommendations)} recommendations:")
    for rec in recommendations:
        print(f"  {rec.bet_type}: {rec.selection} @ {rec.odds:.2f}")
        print(f"    Stake: ${rec.stake:.2f}, EV: ${rec.expected_value:.2f}")


if __name__ == "__main__":
    main()
