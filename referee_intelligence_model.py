#!/usr/bin/env python3
"""
Referee Intelligence Model - Model 11
======================================
Dedicated model for exploiting referee bias patterns.

Based on your data showing:
- Carl Cheffers: 8.62% OT rate (bet overs!)
- Walt Anderson: +1.96 home margin (bet home!)
- John Hussey: Low flags, high blowouts (bet spread extremes!)
- Tony Corrente: Overseas flag surge (situational edge)

This model turns referee assignments into betting edges.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

logger = logging.getLogger(__name__)


class RefereeIntelligenceModel:
    """
    Model 11: Referee Intelligence System

    Analyzes referee assignment and generates betting recommendations
    based on historical bias patterns.
    """

    def __init__(self, referee_data_path: str = "data/referee_training_features.json"):
        self.referee_data_path = Path(referee_data_path)
        self.referee_profiles: Dict[str, Dict[str, Any]] = {}
        self.league_averages = {
            'margin': 0.0,
            'penalties': 6.1,
            'overtime_rate': 6.0,
        }
        self._load_referee_data()

    def _load_referee_data(self):
        """Load referee profiles from JSON."""
        try:
            if not self.referee_data_path.exists():
                logger.warning(f"Referee data not found: {self.referee_data_path}")
                return

            with open(self.referee_data_path, 'r') as f:
                data = json.load(f)
                self.referee_profiles = data.get('referee_profiles', {})

            logger.info(f"Loaded {len(self.referee_profiles)} referee profiles")

        except Exception as e:
            logger.error(f"Error loading referee data: {e}")

    def get_referee_profile(self, referee_name: str) -> Optional[Dict[str, Any]]:
        """Get profile for specific referee."""
        return self.referee_profiles.get(referee_name)

    def get_referee_features(self, referee_name: str) -> Dict[str, float]:
        """
        Extract numerical features for a referee.

        Returns dict with:
            - ref_avg_margin: Home team advantage with this ref
            - ref_avg_penalties: Penalties called per game
            - ref_penalty_diff: vs league average
            - ref_odds_delta: Impact on betting lines
            - ref_overtime_rate: % of games going OT
            - ref_is_high_penalties: Binary flag
            - ref_is_overtime_frequent: Binary flag (>7%)
            - ref_is_low_flags: Binary flag
            - ref_is_blowout_prone: Binary flag
        """
        profile = self.get_referee_profile(referee_name)

        if not profile:
            # Return neutral features if referee unknown
            return {
                'ref_avg_margin': 0.0,
                'ref_avg_penalties': 6.1,
                'ref_penalty_diff': 0.0,
                'ref_odds_delta': 0.0,
                'ref_overtime_rate': 6.0,
                'ref_is_high_penalties': 0,
                'ref_is_overtime_frequent': 0,
                'ref_is_low_flags': 0,
                'ref_is_blowout_prone': 0,
            }

        # Extract raw stats
        avg_margin = profile.get('avg_margin', 0.0)
        avg_penalties = profile.get('avg_penalties', 6.1)
        penalty_diff = profile.get('avg_penalty_diff', 0.0)
        odds_delta = profile.get('avg_odds_delta', 0.0)
        overtime_rate = profile.get('avg_overtime_rate', 6.0)
        labels = profile.get('labels', [])

        # Binary flags from labels
        is_high_penalties = 1 if 'high_penalties_close_games' in labels else 0
        is_overtime_frequent = 1 if 'overtime_frequency_gt_15pct' in labels or overtime_rate > 7.0 else 0
        is_low_flags = 1 if 'low_flags_high_blowouts' in labels else 0
        is_blowout_prone = 1 if 'low_flags_high_blowouts' in labels else 0

        return {
            'ref_avg_margin': float(avg_margin),
            'ref_avg_penalties': float(avg_penalties),
            'ref_penalty_diff': float(penalty_diff),
            'ref_odds_delta': float(odds_delta),
            'ref_overtime_rate': float(overtime_rate),
            'ref_is_high_penalties': is_high_penalties,
            'ref_is_overtime_frequent': is_overtime_frequent,
            'ref_is_low_flags': is_low_flags,
            'ref_is_blowout_prone': is_blowout_prone,
        }

    def detect_referee_edges(
        self,
        referee_name: str,
        spread: float,
        total: float,
        home_team: str,
        away_team: str
    ) -> List[Dict[str, Any]]:
        """
        Detect betting edges based on referee assignment.

        NOW DETECTS ALL BET TYPES:
        - Full game spread
        - Full game total (over/under)
        - Moneyline
        - First half spread
        - Home team total
        - Away team total

        Returns:
            List of edge opportunities with reasoning
        """
        profile = self.get_referee_profile(referee_name)

        if not profile:
            return []

        edges = []

        avg_margin = profile.get('avg_margin', 0.0)
        avg_penalties = profile.get('avg_penalties', 6.1)
        overtime_rate = profile.get('avg_overtime_rate', 6.0)
        labels = profile.get('labels', [])

        # Calculate expected team totals (split the total)
        expected_home_total = total / 2 + (avg_margin / 2)  # Home team gets margin boost
        expected_away_total = total / 2 - (avg_margin / 2)

        # First half spread is typically ~55% of full game
        first_half_spread = spread * 0.55

        # ====================================================================
        # FULL GAME SPREAD EDGES
        # ====================================================================

        # EDGE 1: Significant Home Bias + Close Spread = HOME COVER
        if avg_margin > 1.5 and abs(spread) < 7:
            edges.append({
                'type': 'SPREAD',
                'pick': 'HOME',
                'confidence': min(0.70, avg_margin / 3.0),
                'reason': f"{referee_name} favors home by +{avg_margin:.1f} pts (league avg 0). Historical bias.",
                'edge_size': 'LARGE',
                'signal': 'REF_HOME_BIAS',
            })

        # EDGE 2: Significant Away Bias (Rare but Valuable)
        if avg_margin < -1.5 and abs(spread) < 7:
            edges.append({
                'type': 'SPREAD',
                'pick': 'AWAY',
                'confidence': min(0.70, abs(avg_margin) / 3.0),
                'reason': f"{referee_name} favors away by {avg_margin:.1f} pts. Rare reverse bias.",
                'edge_size': 'LARGE',
                'signal': 'REF_AWAY_BIAS',
            })

        # EDGE 3: High Penalties + Close Spread = UNDERDOG VALUE
        if 'high_penalties_close_games' in labels and abs(spread) < 3:
            edges.append({
                'type': 'SPREAD',
                'pick': 'AWAY' if spread < 0 else 'HOME',  # Bet underdog
                'confidence': 0.62,
                'reason': f"{referee_name} keeps games close with penalties. Underdog value.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_CLOSE_GAME_SPECIALIST',
            })

        # ====================================================================
        # FULL GAME TOTAL EDGES
        # ====================================================================

        # EDGE 4: High Overtime Rate + Low Total = OVER OPPORTUNITY
        if overtime_rate > 7.5 and total < 45:
            edges.append({
                'type': 'TOTAL',
                'pick': 'OVER',
                'confidence': min(0.75, overtime_rate / 10.0),
                'reason': f"{referee_name} OT rate {overtime_rate:.1f}% (league avg 6%). Overtime adds 10+ points.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_OVERTIME_SPECIALIST',
            })

        # EDGE 5: Low Flags + High Total = UNDER OPPORTUNITY
        if 'low_flags_high_blowouts' in labels and total > 47:
            edges.append({
                'type': 'TOTAL',
                'pick': 'UNDER',
                'confidence': 0.60,
                'reason': f"{referee_name} labeled 'low_flags_high_blowouts'. Games end quickly.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_LOW_FLAGS_BLOWOUT',
            })

        # EDGE 6: High Penalties = UNDER (game slows down)
        if 'high_penalties_close_games' in labels or avg_penalties > 7.0:
            edges.append({
                'type': 'TOTAL',
                'pick': 'UNDER',
                'confidence': 0.58,
                'reason': f"{referee_name} calls {avg_penalties:.1f} penalties (league avg 6.1). More stoppages = fewer possessions.",
                'edge_size': 'SMALL',
                'signal': 'REF_HIGH_PENALTIES_UNDER',
            })

        # EDGE 7: Overseas Flag Surge
        if 'overseas_flag_surge' in labels:
            edges.append({
                'type': 'TOTAL',
                'pick': 'UNDER',
                'confidence': 0.55,
                'reason': f"{referee_name} has 'overseas_flag_surge' pattern. Unpredictable penalties.",
                'edge_size': 'SMALL',
                'signal': 'REF_OVERSEAS_PATTERN',
            })

        # ====================================================================
        # MONEYLINE EDGES
        # ====================================================================

        # EDGE 8: Home Underdog + Home Bias = HOME ML VALUE
        if avg_margin > 1.0 and spread > 0:  # Home is underdog
            edges.append({
                'type': 'MONEYLINE',
                'pick': 'HOME',
                'confidence': min(0.65, avg_margin / 2.5),
                'reason': f"{referee_name} +{avg_margin:.1f} home bias. Home underdog = value.",
                'edge_size': 'LARGE',
                'signal': 'REF_HOME_UNDERDOG_VALUE',
            })

        # EDGE 9: Away Underdog + Away Bias = AWAY ML VALUE
        if avg_margin < -1.0 and spread < 0:  # Away is underdog
            edges.append({
                'type': 'MONEYLINE',
                'pick': 'AWAY',
                'confidence': min(0.65, abs(avg_margin) / 2.5),
                'reason': f"{referee_name} {avg_margin:.1f} away bias. Away underdog = value.",
                'edge_size': 'LARGE',
                'signal': 'REF_AWAY_UNDERDOG_VALUE',
            })

        # ====================================================================
        # FIRST HALF SPREAD EDGES (NEW!)
        # ====================================================================

        # EDGE 10: Home bias applies to 1H
        if avg_margin > 2.0 and abs(first_half_spread) < 4:
            edges.append({
                'type': '1H_SPREAD',
                'pick': 'HOME',
                'confidence': min(0.68, avg_margin / 3.5),
                'reason': f"{referee_name} home bias shows early. 1H spread {first_half_spread:.1f}.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_1H_HOME_BIAS',
            })

        # EDGE 11: Away bias applies to 1H
        if avg_margin < -2.0 and abs(first_half_spread) < 4:
            edges.append({
                'type': '1H_SPREAD',
                'pick': 'AWAY',
                'confidence': min(0.68, abs(avg_margin) / 3.5),
                'reason': f"{referee_name} away bias shows early. 1H spread {first_half_spread:.1f}.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_1H_AWAY_BIAS',
            })

        # ====================================================================
        # TEAM TOTAL EDGES (NEW!)
        # ====================================================================

        # EDGE 12: Home Team Total OVER (home bias + high scoring)
        if avg_margin > 2.0:
            home_team_total = total / 2  # Books typically split evenly
            edges.append({
                'type': 'TEAM_TOTAL_HOME',
                'pick': 'OVER',
                'confidence': min(0.64, avg_margin / 4.0),
                'reason': f"{referee_name} adds +{avg_margin:.1f} to home scoring. Expected {expected_home_total:.1f}.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_HOME_TEAM_TOTAL_OVER',
                'line': home_team_total,
            })

        # EDGE 13: Away Team Total UNDER (home bias hurts away)
        if avg_margin > 2.5:
            away_team_total = total / 2
            edges.append({
                'type': 'TEAM_TOTAL_AWAY',
                'pick': 'UNDER',
                'confidence': min(0.62, avg_margin / 5.0),
                'reason': f"{referee_name} hurts away by -{avg_margin:.1f} pts. Expected {expected_away_total:.1f}.",
                'edge_size': 'SMALL',
                'signal': 'REF_AWAY_TEAM_TOTAL_UNDER',
                'line': away_team_total,
            })

        # EDGE 14: Away Team Total OVER (away bias + high scoring)
        if avg_margin < -2.0:
            away_team_total = total / 2
            edges.append({
                'type': 'TEAM_TOTAL_AWAY',
                'pick': 'OVER',
                'confidence': min(0.64, abs(avg_margin) / 4.0),
                'reason': f"{referee_name} adds {abs(avg_margin):.1f} to away scoring. Expected {expected_away_total:.1f}.",
                'edge_size': 'MEDIUM',
                'signal': 'REF_AWAY_TEAM_TOTAL_OVER',
                'line': away_team_total,
            })

        # EDGE 15: Home Team Total UNDER (away bias hurts home)
        if avg_margin < -2.5:
            home_team_total = total / 2
            edges.append({
                'type': 'TEAM_TOTAL_HOME',
                'pick': 'UNDER',
                'confidence': min(0.62, abs(avg_margin) / 5.0),
                'reason': f"{referee_name} hurts home by {avg_margin:.1f} pts. Expected {expected_home_total:.1f}.",
                'edge_size': 'SMALL',
                'signal': 'REF_HOME_TEAM_TOTAL_UNDER',
                'line': home_team_total,
            })

        # EDGE 16: High Penalties = Both Team Totals UNDER
        if avg_penalties > 7.5:
            edges.append({
                'type': 'TEAM_TOTAL_HOME',
                'pick': 'UNDER',
                'confidence': 0.56,
                'reason': f"{referee_name} calls {avg_penalties:.1f} penalties. Slows home offense.",
                'edge_size': 'SMALL',
                'signal': 'REF_HIGH_PENALTIES_TEAM_UNDER',
                'line': total / 2,
            })
            edges.append({
                'type': 'TEAM_TOTAL_AWAY',
                'pick': 'UNDER',
                'confidence': 0.56,
                'reason': f"{referee_name} calls {avg_penalties:.1f} penalties. Slows away offense.",
                'edge_size': 'SMALL',
                'signal': 'REF_HIGH_PENALTIES_TEAM_UNDER',
                'line': total / 2,
            })

        return edges

    def get_referee_confidence_boost(
        self,
        referee_name: str,
        bet_type: str,
        bet_pick: str,
        spread: float,
        total: float
    ) -> Tuple[float, str]:
        """
        Calculate confidence boost based on referee alignment.

        Returns:
            (boost_amount, reasoning)
        """
        edges = self.detect_referee_edges(referee_name, spread, total, "", "")

        # Find edges that match the bet
        matching_edges = [
            e for e in edges
            if e['type'].upper() == bet_type.upper() and e['pick'].upper() == bet_pick.upper()
        ]

        if not matching_edges:
            return 0.0, ""

        # Take the strongest edge
        best_edge = max(matching_edges, key=lambda e: e['confidence'])

        # Boost is based on edge confidence
        boost = best_edge['confidence'] * 0.20  # Up to +20% confidence boost

        return boost, best_edge['reason']

    def generate_referee_report(self, referee_name: str) -> Dict[str, Any]:
        """
        Generate comprehensive report on referee tendencies.
        """
        profile = self.get_referee_profile(referee_name)

        if not profile:
            return {
                'referee': referee_name,
                'status': 'UNKNOWN',
                'recommendation': 'No historical data available',
            }

        total_games = profile.get('total_games', 0)
        avg_margin = profile.get('avg_margin', 0.0)
        avg_penalties = profile.get('avg_penalties', 6.1)
        overtime_rate = profile.get('avg_overtime_rate', 6.0)
        labels = profile.get('labels', [])

        # Determine primary bias
        if avg_margin > 1.5:
            bias = 'HOME_FAVORED'
            bias_desc = f"Historically favors home team by +{avg_margin:.1f} pts"
        elif avg_margin < -1.5:
            bias = 'AWAY_FAVORED'
            bias_desc = f"Historically favors away team by {avg_margin:.1f} pts"
        else:
            bias = 'NEUTRAL'
            bias_desc = f"Balanced ({avg_margin:+.1f} pts)"

        # Overtime tendency
        if overtime_rate > 7.5:
            ot_tendency = 'FREQUENT'
            ot_desc = f"Games go to OT {overtime_rate:.1f}% (league avg 6%)"
        else:
            ot_tendency = 'NORMAL'
            ot_desc = f"OT rate {overtime_rate:.1f}%"

        # Penalty style
        if avg_penalties > 6.5:
            penalty_style = 'FLAG_HEAVY'
            penalty_desc = f"Calls {avg_penalties:.1f} penalties/team (league avg 6.1)"
        elif avg_penalties < 5.7:
            penalty_style = 'LET_THEM_PLAY'
            penalty_desc = f"Calls {avg_penalties:.1f} penalties/team (league avg 6.1)"
        else:
            penalty_style = 'AVERAGE'
            penalty_desc = f"Calls {avg_penalties:.1f} penalties/team"

        return {
            'referee': referee_name,
            'status': 'ACTIVE',
            'experience': f"{total_games} games",
            'bias_type': bias,
            'bias_description': bias_desc,
            'overtime_tendency': ot_tendency,
            'overtime_description': ot_desc,
            'penalty_style': penalty_style,
            'penalty_description': penalty_desc,
            'labels': labels,
            'betting_recommendations': self._generate_betting_tips(profile),
        }

    def _generate_betting_tips(self, profile: Dict[str, Any]) -> List[str]:
        """Generate betting tips based on profile."""
        tips = []

        avg_margin = profile.get('avg_margin', 0.0)
        overtime_rate = profile.get('avg_overtime_rate', 6.0)
        labels = profile.get('labels', [])

        if avg_margin > 1.5:
            tips.append(f"✅ BET HOME when spread is under 7 (+{avg_margin:.1f} home bias)")

        if avg_margin < -1.5:
            tips.append(f"✅ BET AWAY when spread is under 7 ({avg_margin:.1f} away bias)")

        if overtime_rate > 7.5:
            tips.append(f"✅ BET OVER on low totals (<45) - {overtime_rate:.1f}% OT rate")

        if 'low_flags_high_blowouts' in labels:
            tips.append("✅ BET FAVORITE -7+ (games end in blowouts)")
            tips.append("✅ BET UNDER on high totals (games end quickly)")

        if 'high_penalties_close_games' in labels:
            tips.append("✅ BET UNDERDOG on small spreads (penalties keep it close)")
            tips.append("✅ BET UNDER (more stoppages = fewer possessions)")

        if overtime_rate > 8 and avg_margin > 1:
            tips.append("✅ BET HOME ML when close (OT + home bias = huge edge)")

        if not tips:
            tips.append("No strong tendencies detected - use other models")

        return tips


# ========================================================================
# Integration with Enhanced AI Council
# ========================================================================
class RefereeEnhancedConfidence:
    """
    Wraps referee intelligence into confidence adjustments.
    """

    def __init__(self, referee_intel: Optional[RefereeIntelligenceModel] = None):
        self.referee_intel = referee_intel or RefereeIntelligenceModel()

    def adjust_prediction_with_referee(
        self,
        base_confidence: float,
        referee_name: str,
        bet_type: str,
        bet_pick: str,
        spread: float,
        total: float,
    ) -> Tuple[float, List[str]]:
        """
        Adjust base confidence using referee intelligence.

        Returns:
            (adjusted_confidence, edge_signals)
        """
        # Get referee edges
        edges = self.referee_intel.detect_referee_edges(
            referee_name, spread, total, "", ""
        )

        # Find matching edges
        matching = [
            e for e in edges
            if e['type'].upper() == bet_type.upper() and e['pick'].upper() == bet_pick.upper()
        ]

        if not matching:
            return base_confidence, []

        # Apply boost from strongest edge
        best_edge = max(matching, key=lambda e: e['confidence'])
        boost = best_edge['confidence'] * 0.25  # Up to +25% boost

        adjusted = min(1.0, base_confidence + boost)

        signals = [best_edge['signal']]

        return adjusted, signals


if __name__ == "__main__":
    # Test the referee intelligence
    print("=" * 60)
    print("REFEREE INTELLIGENCE MODEL TEST")
    print("=" * 60)

    model = RefereeIntelligenceModel()

    # Test Carl Cheffers (overtime specialist)
    print("\n📊 Carl Cheffers Report:")
    print("-" * 60)
    report = model.generate_referee_report("Carl Cheffers")
    print(f"Bias: {report['bias_description']}")
    print(f"OT: {report['overtime_description']}")
    print(f"Penalties: {report['penalty_description']}")
    print("\n💰 Betting Tips:")
    for tip in report['betting_recommendations']:
        print(f"  {tip}")

    # Test Walt Anderson (home bias + OT)
    print("\n📊 Walt Anderson Report:")
    print("-" * 60)
    report = model.generate_referee_report("Walt Anderson")
    print(f"Bias: {report['bias_description']}")
    print(f"OT: {report['overtime_description']}")
    print("\n💰 Betting Tips:")
    for tip in report['betting_recommendations']:
        print(f"  {tip}")

    # Test John Hussey (blowout specialist)
    print("\n📊 John Hussey Report:")
    print("-" * 60)
    report = model.generate_referee_report("John Hussey")
    print(f"Bias: {report['bias_description']}")
    print(f"Labels: {report['labels']}")
    print("\n💰 Betting Tips:")
    for tip in report['betting_recommendations']:
        print(f"  {tip}")

    # Test edge detection
    print("\n" + "=" * 60)
    print("EDGE DETECTION TEST")
    print("=" * 60)
    print("\nGame: Bills @ Dolphins, Spread: MIA -3.5, Total: 43.5")
    print("Referee: Carl Cheffers\n")

    edges = model.detect_referee_edges("Carl Cheffers", -3.5, 43.5, "MIA", "BUF")
    for edge in edges:
        print(f"✅ {edge['type']} {edge['pick']}")
        print(f"   Confidence: {edge['confidence']:.0%}")
        print(f"   Reason: {edge['reason']}")
        print(f"   Signal: {edge['signal']}\n")
