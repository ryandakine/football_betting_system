#!/usr/bin/env python3
"""
Single Game Analyzer
====================
Run the full 11-model system on a single game.

Usage:
    python analyze_single_game.py --home KC --away BUF --referee "Brad Rogers" --spread -2.5 --total 48.5

Quick example:
    python analyze_single_game.py --home KC --away BUF --referee "Brad Rogers" --spread -2.5 --total 48.5 --home-ml -140 --away-ml 120
"""

import argparse
import json
from datetime import datetime
from typing import Optional

from enhanced_ai_council import EnhancedAICouncil
from betting_types import GameData
from referee_intelligence_model import RefereeIntelligenceModel
from parse_team_referee_pairings import TeamRefereeParser


def analyze_game(
    home_team: str,
    away_team: str,
    referee: str,
    spread: float,
    total: float,
    home_ml: Optional[int] = None,
    away_ml: Optional[int] = None,
    kickoff_time: Optional[str] = None,
    network: Optional[str] = None,
) -> None:
    """Analyze a single game with the full 11-model system."""

    print("\n" + "="*80)
    print(f"🏈 ANALYZING: {away_team} @ {home_team}")
    print("="*80)

    # Build game data
    game_data: GameData = {
        "game_id": f"{away_team}_{home_team}_{datetime.now().strftime('%Y%m%d')}",
        "home_team": home_team,
        "away_team": away_team,
        "referee": referee,
        "spread": spread,
        "total": total,
        "timestamp": datetime.now(),
    }

    if home_ml:
        game_data["home_ml_odds"] = home_ml
    if away_ml:
        game_data["away_ml_odds"] = away_ml
    if kickoff_time:
        game_data["kickoff_window"] = kickoff_time

    # Add some default model predictions (you'd get these from your actual models)
    # For now, use neutral 50/50 to show the referee edge
    game_data["spread_model_home_pct"] = 0.52
    game_data["total_model_over_pct"] = 0.48
    game_data["home_advantage_pct"] = 0.51

    print(f"\n📊 GAME DETAILS:")
    print(f"   Home: {home_team}")
    print(f"   Away: {away_team}")
    print(f"   Spread: {home_team} {spread}")
    print(f"   Total: {total}")
    if home_ml and away_ml:
        print(f"   Moneyline: {home_team} {home_ml} / {away_team} {away_ml}")
    print(f"   Referee: {referee}")
    if kickoff_time:
        print(f"   Kickoff: {kickoff_time}")
    if network:
        print(f"   Network: {network}")

    # Initialize referee intelligence
    print(f"\n🔍 CHECKING REFEREE INTELLIGENCE...")
    ref_intel = RefereeIntelligenceModel()
    team_parser = TeamRefereeParser()

    try:
        team_parser.parse_all_teams()
    except:
        print("   ⚠️  Could not load team-referee pairing data")

    # Get referee profile
    ref_profile = ref_intel.get_referee_profile(referee)
    if ref_profile:
        print(f"\n   Referee Profile: {referee}")
        print(f"   Total Games: {ref_profile.get('total_games', 'N/A')}")
        print(f"   Avg Margin: {ref_profile.get('avg_margin', 0):+.2f}")
        print(f"   Avg Penalties: {ref_profile.get('avg_penalties', 0):.1f}")
        print(f"   OT Rate: {ref_profile.get('avg_overtime_rate', 0):.1f}%")
        print(f"   Labels: {', '.join(ref_profile.get('labels', []))}")

    # Get team-specific bias
    home_bias = team_parser.get_referee_team_bias(referee, home_team)
    away_bias = team_parser.get_referee_team_bias(referee, away_team)

    if home_bias:
        print(f"\n   📈 {referee} vs {home_team} (Home Team):")
        print(f"      Games: {home_bias.get('games', 0)}")
        print(f"      Avg Margin: {home_bias.get('avg_margin', 0):+.2f}")
        print(f"      Penalties: {home_bias.get('penalties_on_team', 0):.1f}")
        print(f"      OT Rate: {home_bias.get('overtime_rate', 0):.1f}%")

    if away_bias:
        print(f"\n   📈 {referee} vs {away_team} (Away Team):")
        print(f"      Games: {away_bias.get('games', 0)}")
        print(f"      Avg Margin: {away_bias.get('avg_margin', 0):+.2f}")
        print(f"      Penalties: {away_bias.get('penalties_on_team', 0):.1f}")
        print(f"      OT Rate: {away_bias.get('overtime_rate', 0):.1f}%")

    # Detect referee edges
    print(f"\n🎯 REFEREE EDGE DETECTION:")
    ref_edges = team_parser.detect_team_referee_edges(
        referee, home_team, away_team, spread, total
    )

    if ref_edges:
        print(f"\n   Found {len(ref_edges)} referee edge(s)!")
        for i, edge in enumerate(ref_edges, 1):
            print(f"\n   Edge #{i}:")
            print(f"   Type: {edge['type']} {edge['pick']}")
            print(f"   Confidence: {edge['confidence']:.0%}")
            print(f"   Edge Size: {edge['edge_size']}")
            print(f"   Signal: {edge['signal']}")
            print(f"   Reason: {edge['reason']}")
    else:
        print(f"   No strong referee edges detected")

    # Run through Enhanced AI Council
    print(f"\n🤖 RUNNING 11-MODEL ENHANCED AI COUNCIL...")

    try:
        council = EnhancedAICouncil()
        prediction = council.make_unified_prediction(game_data)

        print(f"\n{'='*80}")
        print(f"📊 PREDICTION RESULTS")
        print(f"{'='*80}")

        print(f"\n🎯 OVERALL CONFIDENCE: {prediction.confidence:.0%}")
        confidence_stars = "⭐" * int(prediction.confidence * 5)
        print(f"   {confidence_stars}")

        print(f"\n   Risk Level: {prediction.risk_level.value.upper()}")
        print(f"   Version: {prediction.version}")
        if prediction.degraded:
            print(f"   ⚠️  DEGRADED MODE: {', '.join(prediction.degraded_reasons)}")

        # Spread prediction
        print(f"\n📈 SPREAD PREDICTION:")
        spread_pick = prediction.spread_prediction
        print(f"   Pick: {spread_pick.pick.upper()}")
        print(f"   Adjusted Line: {spread_pick.adjusted_line:+.1f}")
        print(f"   Confidence: {spread_pick.confidence:.0%}")
        print(f"   Edge: {spread_pick.edge:+.2f} points")

        # Total prediction
        print(f"\n📊 TOTAL PREDICTION:")
        total_pick = prediction.total_prediction
        print(f"   Pick: {total_pick.pick.upper()}")
        print(f"   Adjusted Line: {total_pick.adjusted_line:.1f}")
        print(f"   Confidence: {total_pick.confidence:.0%}")
        print(f"   Edge: {total_pick.edge:+.2f} points")

        # Moneyline prediction
        print(f"\n💰 MONEYLINE PREDICTION:")
        ml_pick = prediction.moneyline_prediction
        print(f"   Pick: {ml_pick.pick.upper()}")
        print(f"   Confidence: {ml_pick.confidence:.0%}")
        print(f"   Edge: {ml_pick.edge:+.3f}")

        # Edge signals
        print(f"\n🎯 EDGE SIGNALS ({len(prediction.edge_signals)} total):")
        if prediction.edge_signals:
            for signal in prediction.edge_signals:
                print(f"   ✅ {signal}")
        else:
            print(f"   No edge signals detected")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        rec = prediction.recommendation
        if rec.get('primary_play'):
            print(f"   Primary Play: {rec['primary_play']}")
        print(f"   Bet Size: {rec.get('size', 'N/A').upper()}")

        if rec.get('reasoning'):
            print(f"\n   Reasoning:")
            for reason in rec['reasoning']:
                print(f"   - {reason}")

        if rec.get('secondary_plays'):
            print(f"\n   Secondary Plays:")
            for play in rec['secondary_plays']:
                print(f"   - {play.get('play', 'N/A')}: {play.get('reason', 'N/A')}")

        # Enhanced predictions
        if prediction.first_half_spread_prediction:
            fh = prediction.first_half_spread_prediction
            print(f"\n📊 FIRST HALF SPREAD:")
            print(f"   Pick: {fh.pick.upper()}")
            print(f"   Line: {fh.adjusted_line:+.1f}")
            print(f"   Confidence: {fh.confidence:.0%}")

        if prediction.home_team_total_prediction:
            ht = prediction.home_team_total_prediction
            print(f"\n🏠 HOME TEAM TOTAL:")
            print(f"   Pick: {ht.pick.upper()} {ht.adjusted_line:.1f}")
            print(f"   Confidence: {ht.confidence:.0%}")

        if prediction.away_team_total_prediction:
            at = prediction.away_team_total_prediction
            print(f"\n✈️  AWAY TEAM TOTAL:")
            print(f"   Pick: {at.pick.upper()} {at.adjusted_line:.1f}")
            print(f"   Confidence: {at.confidence:.0%}")

        # Final recommendation
        print(f"\n{'='*80}")
        print(f"🎯 FINAL VERDICT")
        print(f"{'='*80}")

        if prediction.confidence >= 0.80 and prediction.risk_level.value == "low":
            print(f"\n✅ STRONG BET - {prediction.confidence:.0%} confidence, LOW risk")
            print(f"   Recommended: MAX BET (5 units)")
        elif prediction.confidence >= 0.70 and prediction.risk_level.value != "high":
            print(f"\n✅ GOOD BET - {prediction.confidence:.0%} confidence")
            print(f"   Recommended: 2-3 units")
        elif prediction.confidence >= 0.65:
            print(f"\n⚠️  MODERATE BET - {prediction.confidence:.0%} confidence")
            print(f"   Recommended: 1-2 units")
        else:
            print(f"\n❌ PASS - Confidence too low ({prediction.confidence:.0%})")
            print(f"   Recommended: WAIT FOR BETTER SPOT")

        if prediction.degraded:
            print(f"\n⚠️  WARNING: System in degraded mode - consider passing")

        if "REFEREE_EDGE_JACKPOT" in prediction.edge_signals:
            print(f"\n🎰 JACKPOT! Multiple referee edges aligned - MAX BET!")

        if "TEAM_REF_UNDERDOG_VALUE" in prediction.edge_signals:
            print(f"\n💎 UNDERDOG VALUE! Team-specific ref bias on underdog - BET ML!")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"\n❌ Error running Enhanced AI Council: {e}")
        import traceback
        traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single NFL game with the 11-model system"
    )
    parser.add_argument("--home", required=True, help="Home team abbreviation (e.g., KC)")
    parser.add_argument("--away", required=True, help="Away team abbreviation (e.g., BUF)")
    parser.add_argument("--referee", required=True, help="Referee name (e.g., 'Brad Rogers')")
    parser.add_argument("--spread", type=float, required=True, help="Spread (home team perspective, e.g., -2.5)")
    parser.add_argument("--total", type=float, required=True, help="Total points (e.g., 48.5)")
    parser.add_argument("--home-ml", type=int, help="Home team moneyline (e.g., -140)")
    parser.add_argument("--away-ml", type=int, help="Away team moneyline (e.g., 120)")
    parser.add_argument("--time", help="Kickoff time (e.g., 'SNF', 'MNF', 'Sun 1:00 PM')")
    parser.add_argument("--network", help="TV network (e.g., 'NBC', 'CBS')")

    args = parser.parse_args()

    analyze_game(
        home_team=args.home,
        away_team=args.away,
        referee=args.referee,
        spread=args.spread,
        total=args.total,
        home_ml=args.home_ml,
        away_ml=args.away_ml,
        kickoff_time=args.time,
        network=args.network,
    )


if __name__ == "__main__":
    main()
