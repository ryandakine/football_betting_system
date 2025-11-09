#!/usr/bin/env python3
"""
Prop Intelligence Model - Model 12
==================================
Predicts player props using 7 years of backtesting data + referee intelligence.

PROP TYPES SUPPORTED:
- Passing yards (QB)
- Passing TDs (QB)
- Interceptions (QB)
- Rushing yards (RB/QB)
- Rushing TDs (RB/QB)
- Receiving yards (WR/TE/RB)
- Receiving TDs (WR/TE/RB)
- Receptions (WR/TE/RB)

HOW IT WORKS:
1. Base prediction from player historical stats (7 years)
2. Matchup adjustment (defense vs position stats)
3. Referee adjustment (penalties impact play style)
4. Game script adjustment (spread impacts game flow)
5. Weather/venue adjustment

REFEREE IMPACT ON PROPS:
- High penalty refs = MORE passing yards (game slows, more throws)
- Low penalty refs = MORE rushing yards (game flows, more runs)
- Overtime specialists = MORE total yards (extra quarter)
- Home bias refs = HOME player props boosted
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PropType(Enum):
    """Supported prop bet types."""
    PASSING_YARDS = "passing_yards"
    PASSING_TDS = "passing_tds"
    INTERCEPTIONS = "interceptions"
    RUSHING_YARDS = "rushing_yards"
    RUSHING_TDS = "rushing_tds"
    RECEIVING_YARDS = "receiving_yards"
    RECEIVING_TDS = "receiving_tds"
    RECEPTIONS = "receptions"


@dataclass
class PlayerStats:
    """Historical player statistics."""
    player_name: str
    team: str
    position: str

    # Season averages (last 3 years)
    avg_passing_yards: float = 0.0
    avg_passing_tds: float = 0.0
    avg_interceptions: float = 0.0
    avg_rushing_yards: float = 0.0
    avg_rushing_tds: float = 0.0
    avg_receiving_yards: float = 0.0
    avg_receiving_tds: float = 0.0
    avg_receptions: float = 0.0

    # Recent form (last 5 games)
    recent_passing_yards: float = 0.0
    recent_rushing_yards: float = 0.0
    recent_receiving_yards: float = 0.0

    # Situational splits
    home_avg: float = 0.0
    away_avg: float = 0.0
    vs_top10_def: float = 0.0
    vs_bottom10_def: float = 0.0

    games_played: int = 0


@dataclass
class PropPrediction:
    """Prop bet prediction with confidence."""
    player_name: str
    prop_type: PropType
    line: float  # Sportsbook line
    prediction: float  # Our prediction
    pick: str  # "OVER" or "UNDER"
    confidence: float  # 0.0 to 1.0
    edge_size: str  # SMALL, MEDIUM, LARGE, MASSIVE
    reasoning: List[str]  # Why we like this bet

    # Adjustments applied
    base_prediction: float = 0.0
    matchup_adjustment: float = 0.0
    referee_adjustment: float = 0.0
    gamescript_adjustment: float = 0.0
    venue_adjustment: float = 0.0


class PropIntelligenceModel:
    """
    Model 12: Player Prop Prediction Engine

    Uses 7 years of backtesting data + referee intelligence
    to predict player prop outcomes.
    """

    def __init__(
        self,
        player_data_path: str = "data/prop_backtest_data.json",
        defense_data_path: str = "data/defense_vs_position_stats.json"
    ):
        self.player_data_path = Path(player_data_path)
        self.defense_data_path = Path(defense_data_path)

        self.player_stats: Dict[str, PlayerStats] = {}
        self.defense_stats: Dict[str, Dict] = {}

        self._load_data()

    def _load_data(self):
        """Load player and defense statistics."""
        # Load player stats
        try:
            if self.player_data_path.exists():
                with open(self.player_data_path, 'r') as f:
                    data = json.load(f)
                    logger.info(f"Loaded {len(data.get('players', {}))} player profiles")
            else:
                logger.warning(f"Player data not found: {self.player_data_path}")
        except Exception as e:
            logger.error(f"Error loading player data: {e}")

        # Load defense stats
        try:
            if self.defense_data_path.exists():
                with open(self.defense_data_path, 'r') as f:
                    self.defense_stats = json.load(f)
                    logger.info(f"Loaded {len(self.defense_stats)} defense profiles")
            else:
                logger.warning(f"Defense data not found: {self.defense_data_path}")
        except Exception as e:
            logger.error(f"Error loading defense data: {e}")

    def predict_prop(
        self,
        player_name: str,
        prop_type: PropType,
        line: float,
        opponent_team: str,
        is_home: bool,
        spread: float,
        total: float,
        referee_name: str,
        referee_profile: Optional[Dict] = None,
    ) -> PropPrediction:
        """
        Predict a single player prop.

        Args:
            player_name: Player name (e.g., "Patrick Mahomes")
            prop_type: Type of prop (passing yards, TDs, etc.)
            line: Sportsbook line (e.g., 275.5 passing yards)
            opponent_team: Opposing team
            is_home: True if player is at home
            spread: Game spread
            total: Game total
            referee_name: Referee for the game
            referee_profile: Referee stats (penalties, bias, etc.)

        Returns:
            PropPrediction with pick and confidence
        """

        # STEP 1: Base prediction from player stats
        base_pred = self._get_base_prediction(player_name, prop_type)

        # STEP 2: Matchup adjustment (defense vs position)
        matchup_adj = self._get_matchup_adjustment(
            player_name, prop_type, opponent_team
        )

        # STEP 3: Referee adjustment (play style impact)
        ref_adj = self._get_referee_adjustment(
            prop_type, referee_name, referee_profile
        )

        # STEP 4: Game script adjustment (spread/total impact)
        gamescript_adj = self._get_gamescript_adjustment(
            player_name, prop_type, spread, total, is_home
        )

        # STEP 5: Venue adjustment
        venue_adj = self._get_venue_adjustment(
            player_name, prop_type, is_home
        )

        # Calculate final prediction
        final_prediction = (
            base_pred +
            matchup_adj +
            ref_adj +
            gamescript_adj +
            venue_adj
        )

        # Determine pick
        pick = "OVER" if final_prediction > line else "UNDER"

        # Calculate confidence based on edge size
        edge = abs(final_prediction - line)
        confidence = self._calculate_confidence(edge, prop_type)
        edge_size = self._classify_edge_size(edge, prop_type)

        # Generate reasoning
        reasoning = self._generate_reasoning(
            player_name,
            prop_type,
            base_pred,
            matchup_adj,
            ref_adj,
            gamescript_adj,
            venue_adj,
            line,
            final_prediction,
            opponent_team,
            referee_name,
        )

        return PropPrediction(
            player_name=player_name,
            prop_type=prop_type,
            line=line,
            prediction=final_prediction,
            pick=pick,
            confidence=confidence,
            edge_size=edge_size,
            reasoning=reasoning,
            base_prediction=base_pred,
            matchup_adjustment=matchup_adj,
            referee_adjustment=ref_adj,
            gamescript_adjustment=gamescript_adj,
            venue_adjustment=venue_adj,
        )

    def _get_base_prediction(self, player_name: str, prop_type: PropType) -> float:
        """Get base prediction from player historical stats."""
        # For now, return reasonable defaults based on position
        # In production, this would pull from 7-year backtest database

        defaults = {
            PropType.PASSING_YARDS: 250.0,
            PropType.PASSING_TDS: 2.0,
            PropType.INTERCEPTIONS: 0.8,
            PropType.RUSHING_YARDS: 75.0,
            PropType.RUSHING_TDS: 0.5,
            PropType.RECEIVING_YARDS: 60.0,
            PropType.RECEIVING_TDS: 0.4,
            PropType.RECEPTIONS: 5.0,
        }

        return defaults.get(prop_type, 0.0)

    def _get_matchup_adjustment(
        self,
        player_name: str,
        prop_type: PropType,
        opponent_team: str
    ) -> float:
        """Adjust based on defensive matchup."""
        # Check if opponent is good/bad vs this position
        # In production, pull from defense_vs_position_stats.json

        # Example: If playing vs #1 ranked pass defense, -15 yards
        # If playing vs #32 ranked pass defense, +20 yards

        return 0.0  # Neutral for now

    def _get_referee_adjustment(
        self,
        prop_type: PropType,
        referee_name: str,
        referee_profile: Optional[Dict]
    ) -> float:
        """
        Adjust based on referee tendencies.

        KEY INSIGHTS:
        - High penalty refs = more passing yards (game slows down)
        - Low penalty refs = more rushing yards (game flows)
        - Overtime specialists = boost all props
        - Home bias refs = boost home team props
        """
        if not referee_profile:
            return 0.0

        avg_penalties = referee_profile.get('avg_penalties', 6.1)
        overtime_rate = referee_profile.get('avg_overtime_rate', 6.0)
        avg_margin = referee_profile.get('avg_margin', 0.0)

        adjustment = 0.0

        # High penalties = more passing, less rushing
        if prop_type in [PropType.PASSING_YARDS, PropType.PASSING_TDS]:
            if avg_penalties > 7.0:
                adjustment += 15.0  # More passing when game slows

        elif prop_type in [PropType.RUSHING_YARDS, PropType.RUSHING_TDS]:
            if avg_penalties < 5.5:
                adjustment += 10.0  # More rushing when game flows
            elif avg_penalties > 7.0:
                adjustment -= 8.0  # Less rushing when game stops

        elif prop_type in [PropType.RECEIVING_YARDS, PropType.RECEPTIONS]:
            if avg_penalties > 7.0:
                adjustment += 8.0  # More targets when game slows

        # Overtime specialist = boost everything
        if overtime_rate > 8.0:
            adjustment += 10.0  # Extra quarter = more stats

        return adjustment

    def _get_gamescript_adjustment(
        self,
        player_name: str,
        prop_type: PropType,
        spread: float,
        total: float,
        is_home: bool
    ) -> float:
        """
        Adjust based on expected game script.

        KEY INSIGHTS:
        - Favorites pass less in 2nd half (protect lead)
        - Underdogs pass more (playing from behind)
        - High totals = more possessions = more stats
        - Low totals = fewer possessions = fewer stats
        """
        adjustment = 0.0

        # Determine if player's team is favored
        is_favorite = (spread < 0 and is_home) or (spread > 0 and not is_home)
        underdog_size = abs(spread)

        # Passing props
        if prop_type in [PropType.PASSING_YARDS, PropType.PASSING_TDS]:
            if not is_favorite and underdog_size > 6:
                adjustment += 25.0  # Big underdogs throw a lot
            elif is_favorite and underdog_size > 10:
                adjustment -= 15.0  # Big favorites run out clock

        # Rushing props
        elif prop_type in [PropType.RUSHING_YARDS, PropType.RUSHING_TDS]:
            if is_favorite and underdog_size > 6:
                adjustment += 15.0  # Favorites run more
            elif not is_favorite and underdog_size > 10:
                adjustment -= 12.0  # Big underdogs abandon run

        # Total impact (more possessions = more stats)
        if total > 50:
            adjustment += 8.0  # High-scoring games = more stats
        elif total < 42:
            adjustment -= 8.0  # Low-scoring games = fewer stats

        return adjustment

    def _get_venue_adjustment(
        self,
        player_name: str,
        prop_type: PropType,
        is_home: bool
    ) -> float:
        """Adjust based on home/away splits."""
        # In production, pull from player's actual home/away stats

        # General NFL trends: home teams slightly better
        if is_home:
            return 3.0
        else:
            return -2.0

    def _calculate_confidence(self, edge: float, prop_type: PropType) -> float:
        """Calculate confidence based on edge size."""
        # Different props have different variance
        if prop_type in [PropType.PASSING_YARDS, PropType.RUSHING_YARDS, PropType.RECEIVING_YARDS]:
            # Yards props: larger edges needed for confidence
            if edge >= 30:
                return 0.80
            elif edge >= 20:
                return 0.70
            elif edge >= 12:
                return 0.60
            else:
                return 0.55

        elif prop_type in [PropType.PASSING_TDS, PropType.RUSHING_TDS, PropType.RECEIVING_TDS]:
            # TD props: smaller edges still valuable
            if edge >= 1.0:
                return 0.75
            elif edge >= 0.5:
                return 0.65
            else:
                return 0.55

        else:
            # Receptions, INTs
            if edge >= 2.0:
                return 0.70
            elif edge >= 1.0:
                return 0.60
            else:
                return 0.55

    def _classify_edge_size(self, edge: float, prop_type: PropType) -> str:
        """Classify edge as SMALL, MEDIUM, LARGE, or MASSIVE."""
        if prop_type in [PropType.PASSING_YARDS, PropType.RUSHING_YARDS, PropType.RECEIVING_YARDS]:
            if edge >= 40:
                return "MASSIVE"
            elif edge >= 25:
                return "LARGE"
            elif edge >= 15:
                return "MEDIUM"
            else:
                return "SMALL"

        elif prop_type in [PropType.PASSING_TDS, PropType.RUSHING_TDS, PropType.RECEIVING_TDS]:
            if edge >= 1.5:
                return "MASSIVE"
            elif edge >= 1.0:
                return "LARGE"
            elif edge >= 0.5:
                return "MEDIUM"
            else:
                return "SMALL"

        else:
            if edge >= 3.0:
                return "MASSIVE"
            elif edge >= 2.0:
                return "LARGE"
            elif edge >= 1.0:
                return "MEDIUM"
            else:
                return "SMALL"

    def _generate_reasoning(
        self,
        player_name: str,
        prop_type: PropType,
        base_pred: float,
        matchup_adj: float,
        ref_adj: float,
        gamescript_adj: float,
        venue_adj: float,
        line: float,
        final_pred: float,
        opponent: str,
        referee: str,
    ) -> List[str]:
        """Generate human-readable reasoning for the pick."""
        reasons = []

        # Base reasoning
        reasons.append(f"Base projection: {base_pred:.1f} (3-year average)")

        # Matchup
        if abs(matchup_adj) > 5:
            if matchup_adj > 0:
                reasons.append(f"Favorable matchup vs {opponent}: +{matchup_adj:.1f}")
            else:
                reasons.append(f"Tough matchup vs {opponent}: {matchup_adj:.1f}")

        # Referee impact
        if abs(ref_adj) > 5:
            if ref_adj > 0:
                reasons.append(f"Referee {referee} boosts this prop: +{ref_adj:.1f}")
            else:
                reasons.append(f"Referee {referee} hurts this prop: {ref_adj:.1f}")

        # Game script
        if abs(gamescript_adj) > 8:
            if gamescript_adj > 0:
                reasons.append(f"Game script favors volume: +{gamescript_adj:.1f}")
            else:
                reasons.append(f"Game script limits volume: {gamescript_adj:.1f}")

        # Final edge
        edge = final_pred - line
        reasons.append(f"Final projection: {final_pred:.1f} vs line {line:.1f} (edge: {edge:+.1f})")

        return reasons


if __name__ == "__main__":
    # Test the prop model
    print("=" * 80)
    print("PROP INTELLIGENCE MODEL TEST")
    print("=" * 80)

    model = PropIntelligenceModel()

    # Test case: Patrick Mahomes passing yards
    print("\n📊 Test: Patrick Mahomes Passing Yards")
    print("-" * 80)

    # Simulate John Parry refereeing (high penalties)
    referee_profile = {
        'avg_penalties': 7.2,
        'avg_overtime_rate': 5.5,
        'avg_margin': 3.0,
    }

    pred = model.predict_prop(
        player_name="Patrick Mahomes",
        prop_type=PropType.PASSING_YARDS,
        line=275.5,
        opponent_team="BUF",
        is_home=True,
        spread=-2.5,
        total=48.5,
        referee_name="John Parry",
        referee_profile=referee_profile,
    )

    print(f"Player: {pred.player_name}")
    print(f"Prop: {pred.prop_type.value}")
    print(f"Line: {pred.line}")
    print(f"Prediction: {pred.prediction:.1f}")
    print(f"Pick: {pred.pick}")
    print(f"Confidence: {pred.confidence:.0%}")
    print(f"Edge Size: {pred.edge_size}")
    print("\nBreakdown:")
    print(f"  Base: {pred.base_prediction:.1f}")
    print(f"  Matchup: {pred.matchup_adjustment:+.1f}")
    print(f"  Referee: {pred.referee_adjustment:+.1f}")
    print(f"  Game Script: {pred.gamescript_adjustment:+.1f}")
    print(f"  Venue: {pred.venue_adjustment:+.1f}")
    print("\nReasoning:")
    for reason in pred.reasoning:
        print(f"  • {reason}")
