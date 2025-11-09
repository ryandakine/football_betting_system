#!/usr/bin/env python3
"""
Test NCAA Officiating Bias System
Demonstrates how conference crew analysis works
"""

from ncaa_models.officiating_analysis.officiating_bias_detector import OfficiatingBiasDetector
from ncaa_models.officiating_analysis.conference_crew_analyzer import ConferenceCrewAnalyzer

def main():
    print("\n" + "="*70)
    print("🏈 NCAA OFFICIATING BIAS ANALYSIS SYSTEM")
    print("="*70)

    # Initialize detector
    detector = OfficiatingBiasDetector()

    print("\n📋 CONFERENCE OFFICIATING PROFILES")
    print("="*70)

    # Show all conference profiles
    for conf, profile in detector.get_all_profiles().items():
        print(f"\n{conf}:")
        print(f"  Home Bias: {profile.home_bias_score:.2f} (0.50 = neutral)")
        print(f"  Protection: {profile.protection_score:.2f}")
        print(f"  Penalties/Game: {profile.penalty_strictness:.1f}")
        print(f"  Risk Level: {profile.risk_level}")

    print("\n" + "="*70)
    print("🎯 EXAMPLE GAME ANALYSES")
    print("="*70)

    # Example 1: SEC home game
    print("\n" + "-"*70)
    print("Example 1: Alabama (SEC) vs Texas A&M (SEC)")
    print("Officiating: SEC crew")
    print("-"*70)
    
    report1 = detector.generate_officiating_report(
        home_team="Alabama",
        away_team="Texas A&M",
        home_conference="SEC",
        away_conference="SEC",
        officiating_conference="SEC"
    )
    print(report1)

    # Example 2: Cross-conference with SEC crew
    print("\n" + "-"*70)
    print("Example 2: Georgia (SEC) vs Clemson (ACC)")
    print("Officiating: SEC crew (protecting home)")
    print("-"*70)
    
    report2 = detector.generate_officiating_report(
        home_team="Georgia",
        away_team="Clemson",
        home_conference="SEC",
        away_conference="ACC",
        officiating_conference="SEC"
    )
    print(report2)

    # Example 3: Balanced Pac-12 crew
    print("\n" + "-"*70)
    print("Example 3: USC (Pac-12) vs Notre Dame (Independent)")
    print("Officiating: Pac-12 crew (most balanced)")
    print("-"*70)
    
    report3 = detector.generate_officiating_report(
        home_team="USC",
        away_team="Notre Dame",
        home_conference="Pac-12",
        away_conference="Independent",
        officiating_conference="Pac-12"
    )
    print(report3)

    # Example 4: Big Ten crew
    print("\n" + "-"*70)
    print("Example 4: Ohio State (Big Ten) vs Michigan (Big Ten)")
    print("Officiating: Big Ten crew, Rivalry game")
    print("-"*70)
    
    adjustments = detector.get_bias_adjustment(
        home_team="Ohio State",
        away_team="Michigan",
        home_conference="Big Ten",
        away_conference="Big Ten",
        officiating_conference="Big Ten",
        is_rivalry=True
    )
    
    print("\nBetting Adjustments:")
    print(f"  Spread Adjustment: {adjustments['spread_adjustment']:+.1f} points")
    print(f"  Confidence Penalty: {adjustments['confidence_penalty']:.1%}")
    print(f"  Risk Score: {adjustments['risk_score']:.2f}")
    print(f"  Recommendation: {adjustments['recommendation']}")
    print(f"\n  Reason: {adjustments['reason']}")

    print("\n" + "="*70)
    print("💡 KEY INSIGHTS")
    print("="*70)
    print("""
1. SEC crews show highest home bias (0.58 vs 0.50 neutral)
   → Adds ~1.6 points to home team in SEC games

2. Conference protection is real in cross-conference games
   → SEC crew officiating SEC vs ACC: +1.5 pts for SEC team

3. Pac-12 crews are most balanced (0.50 home bias)
   → Best for objective officiating

4. Big 12 calls most penalties (13.5/game)
   → Higher variance, more impact on game

5. Rivalry games reduce confidence
   → Officials call these differently

6. Use for betting adjustments:
   - Adjust spread by bias amount
   - Reduce confidence in high-risk scenarios
   - Avoid betting when extreme protection exists
""")

    print("\n" + "="*70)
    print("🎯 HOW TO USE WITH NCAA AGENT")
    print("="*70)
    print("""
The officiating bias system automatically integrates with predictions:

1. Feature Engineering:
   - Extracts conference matchup info
   - Detects rivalry games
   - Identifies officiating crew

2. Prediction Adjustment:
   - Adds spread adjustment to model prediction
   - Reduces confidence in high-risk scenarios
   - Flags games to avoid

3. Pick Generation:
   - Incorporates officiating bias into edge calculation
   - Shows officiating analysis in reasoning
   - Adjusts bet sizing based on risk

Example integration:
  Model predicts: Alabama -7.5
  Officiating bias: +1.5 (SEC crew protecting home)
  Adjusted line: Alabama -9.0
  Confidence: Reduced by 5%
""")

    print("\n✅ Officiating bias system ready to use!")


if __name__ == "__main__":
    main()
