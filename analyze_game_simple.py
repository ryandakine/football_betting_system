#!/usr/bin/env python3
"""
Simple Game Analyzer - Referee Intelligence Focus
==================================================
Analyzes a single game using Model 11 (Referee Intelligence).

Usage:
    python analyze_game_simple.py --home KC --away BUF --referee "Brad Rogers" --spread -2.5 --total 48.5

Quick examples:
    python analyze_game_simple.py --home KC --away BUF --referee "Brad Rogers" --spread -2.5 --total 48.5
    python analyze_game_simple.py --home CIN --away BAL --referee "Carl Cheffers" --spread 3.5 --total 42.0
    python analyze_game_simple.py --home DAL --away PHI --referee "John Hussey" --spread -6.5 --total 45.0
"""

import argparse
from typing import Optional

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
) -> None:
    """Analyze a single game focusing on referee intelligence (Model 11)."""

    print("\n" + "="*80)
    print(f"🏈 GAME ANALYSIS: {away_team} @ {home_team}")
    print("="*80)

    print(f"\n📊 GAME DETAILS:")
    print(f"   Home: {home_team}")
    print(f"   Away: {away_team}")
    print(f"   Spread: {home_team} {spread:+.1f}")
    print(f"   Total: {total}")
    if home_ml and away_ml:
        print(f"   Moneyline: {home_team} {home_ml:+d} / {away_team} {away_ml:+d}")
    print(f"   Referee: {referee}")

    # Initialize referee intelligence (Model 11)
    print(f"\n" + "="*80)
    print(f"🎯 MODEL 11: REFEREE INTELLIGENCE ANALYSIS")
    print("="*80)

    ref_intel = RefereeIntelligenceModel()
    team_parser = TeamRefereeParser()

    # Load team-referee pairing data
    print(f"\n📚 Loading referee database...")
    try:
        team_parser.parse_all_teams()
        print(f"   ✅ Loaded 32 teams × 20+ referees = 640+ pairings")
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")

    # Get general referee profile
    print(f"\n🔍 REFEREE PROFILE: {referee}")
    print("-" * 80)

    ref_profile = ref_intel.get_referee_profile(referee)
    if ref_profile:
        print(f"   Total Games Worked: {ref_profile.get('total_games', 'N/A')}")
        print(f"   Average Margin: {ref_profile.get('avg_margin', 0):+.2f} points")
        print(f"   Average Penalties: {ref_profile.get('avg_penalties', 0):.1f} per team")
        print(f"   Overtime Rate: {ref_profile.get('avg_overtime_rate', 0):.1f}% (league avg: 6%)")

        labels = ref_profile.get('labels', [])
        if labels:
            print(f"   Classifications: {', '.join(labels)}")

        print(f"\n   💡 Interpretation:")
        ot_rate = ref_profile.get('avg_overtime_rate', 0)
        if ot_rate > 7.5:
            print(f"      - HIGH OT FREQUENCY! Games go to overtime often")

        penalties = ref_profile.get('avg_penalties', 0)
        if penalties > 6.5:
            print(f"      - Calls MANY penalties (slows game down)")
        elif penalties < 5.7:
            print(f"      - Lets teams play (fewer stoppages)")
    else:
        print(f"   ❌ No profile found for {referee}")

    # Get team-specific bias (THE GOLD!)
    print(f"\n🔥 TEAM-SPECIFIC BIAS (THE EDGE!):")
    print("-" * 80)

    home_bias = team_parser.get_referee_team_bias(referee, home_team)
    away_bias = team_parser.get_referee_team_bias(referee, away_team)

    if home_bias:
        print(f"\n   📈 {referee} + {home_team} (HOME TEAM):")
        games = home_bias.get('games', 0)
        margin = home_bias.get('avg_margin', 0)
        penalties = home_bias.get('penalties_on_team', 0)
        ot_rate = home_bias.get('overtime_rate', 0)

        print(f"      Games: {games}")
        print(f"      Avg Margin: {margin:+.1f} points")
        print(f"      Penalties on Team: {penalties:.1f}")
        print(f"      Overtime Rate: {ot_rate:.1f}%")

        if games >= 5:
            if abs(margin) > 7.0:
                if margin > 0:
                    print(f"\n      🎯 STRONG EDGE: {referee} favors {home_team} by +{margin:.1f} pts!")
                else:
                    print(f"\n      ⚠️  WARNING: {referee} unfavorable to {home_team} ({margin:.1f} pts)")
        else:
            print(f"\n      ℹ️  Small sample ({games} games) - use caution")
    else:
        print(f"\n   ℹ️  No {home_team} history with {referee}")

    if away_bias:
        print(f"\n   📈 {referee} + {away_team} (AWAY TEAM):")
        games = away_bias.get('games', 0)
        margin = away_bias.get('avg_margin', 0)
        penalties = away_bias.get('penalties_on_team', 0)
        ot_rate = away_bias.get('overtime_rate', 0)

        print(f"      Games: {games}")
        print(f"      Avg Margin: {margin:+.1f} points (away team perspective)")
        print(f"      Penalties on Team: {penalties:.1f}")
        print(f"      Overtime Rate: {ot_rate:.1f}%")
    else:
        print(f"\n   ℹ️  No {away_team} history with {referee}")

    # Detect edges
    print(f"\n" + "="*80)
    print(f"🎯 EDGE DETECTION")
    print("="*80)

    # Get all edges
    ref_edges = team_parser.detect_team_referee_edges(
        referee, home_team, away_team, spread, total
    )

    if ref_edges:
        print(f"\n✅ FOUND {len(ref_edges)} REFEREE EDGE(S)!\n")

        for i, edge in enumerate(ref_edges, 1):
            confidence = edge['confidence']
            confidence_stars = "⭐" * int(confidence * 5)

            print(f"\n{'='*80}")
            print(f"EDGE #{i}: {edge['type']} {edge['pick']}")
            print(f"{'='*80}")
            print(f"   Confidence: {confidence:.0%} {confidence_stars}")
            print(f"   Edge Size: {edge['edge_size']}")
            print(f"   Signal: {edge['signal']}")
            print(f"\n   💰 Reasoning:")
            print(f"   {edge['reason']}")

            # Bet recommendation
            print(f"\n   📋 Recommendation:")
            if edge['edge_size'] == 'MASSIVE':
                print(f"      🔥 MAX BET (5 units) - RARE OPPORTUNITY!")
            elif edge['edge_size'] == 'LARGE':
                print(f"      ✅ BET STRONG (3-4 units)")
            elif edge['edge_size'] == 'MEDIUM':
                print(f"      ✅ BET MODERATE (2-3 units)")
            else:
                print(f"      ⚠️  BET SMALL (1 unit)")

        # Overall recommendation
        print(f"\n" + "="*80)
        print(f"🎯 OVERALL VERDICT")
        print("="*80)

        best_edge = max(ref_edges, key=lambda x: x['confidence'])
        total_edges = len(ref_edges)

        if total_edges >= 3:
            print(f"\n🎰 JACKPOT! {total_edges} edges aligned on this game!")
            print(f"   When multiple referee edges stack = MAX BET TERRITORY!")

        if best_edge['confidence'] >= 0.75:
            print(f"\n✅ STRONG PLAY")
            print(f"   Best Edge: {best_edge['type']} {best_edge['pick']} ({best_edge['confidence']:.0%})")
            print(f"   This is a HIGH CONFIDENCE bet based on referee intelligence.")
        elif best_edge['confidence'] >= 0.65:
            print(f"\n✅ GOOD PLAY")
            print(f"   Best Edge: {best_edge['type']} {best_edge['pick']} ({best_edge['confidence']:.0%})")
            print(f"   This is a SOLID bet worth considering.")
        else:
            print(f"\n⚠️  MODERATE PLAY")
            print(f"   Best Edge: {best_edge['type']} {best_edge['pick']} ({best_edge['confidence']:.0%})")
            print(f"   Edge detected but confidence is moderate.")

    else:
        print(f"\n❌ NO STRONG REFEREE EDGES DETECTED")
        print(f"\nThis doesn't mean the game has no value - just that the referee")
        print(f"assignment doesn't provide a statistical edge based on historical data.")
        print(f"\nConsider:")
        print(f"  - Other models (spread, total, moneyline ensembles)")
        print(f"  - Situational factors (weather, rest, injuries)")
        print(f"  - Sharp vs public money divergence")

    # Generate referee report
    print(f"\n" + "="*80)
    print(f"📊 REFEREE SCOUTING REPORT")
    print("="*80)

    report = ref_intel.generate_referee_report(referee)

    print(f"\nReferee: {report['referee']}")
    print(f"Experience: {report.get('experience', 'N/A')}")
    print(f"\nBias Profile:")
    print(f"  Type: {report.get('bias_type', 'N/A')}")
    print(f"  {report.get('bias_description', 'N/A')}")
    print(f"\nOvertme Tendency:")
    print(f"  {report.get('overtime_description', 'N/A')}")
    print(f"\nPenalty Style:")
    print(f"  {report.get('penalty_description', 'N/A')}")

    tips = report.get('betting_recommendations', [])
    if tips:
        print(f"\n💰 Betting Tips with {referee}:")
        for tip in tips:
            print(f"  {tip}")

    print(f"\n{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single NFL game using Referee Intelligence (Model 11)"
    )
    parser.add_argument("--home", required=True, help="Home team abbreviation (e.g., KC)")
    parser.add_argument("--away", required=True, help="Away team abbreviation (e.g., BUF)")
    parser.add_argument("--referee", required=True, help="Referee name (e.g., 'Brad Rogers')")
    parser.add_argument("--spread", type=float, required=True, help="Spread (home perspective, e.g., -2.5)")
    parser.add_argument("--total", type=float, required=True, help="Total points (e.g., 48.5)")
    parser.add_argument("--home-ml", type=int, help="Home moneyline (e.g., -140)")
    parser.add_argument("--away-ml", type=int, help="Away moneyline (e.g., 120)")

    args = parser.parse_args()

    analyze_game(
        home_team=args.home,
        away_team=args.away,
        referee=args.referee,
        spread=args.spread,
        total=args.total,
        home_ml=args.home_ml,
        away_ml=args.away_ml,
    )


if __name__ == "__main__":
    main()
