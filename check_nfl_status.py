#!/usr/bin/env python3
"""
Quick NFL Live Tracking Status Check
=====================================
Run this while watching NFL games to see what the AI is doing.
"""

from nfl_live_tracker import NFLLiveGameTracker
from datetime import datetime

def main():
    print("🏈 NFL LIVE TRACKING STATUS CHECK")
    print("=" * 40)

    # Initialize tracker
    tracker = NFLLiveGameTracker()

    # Get current status
    status = tracker.get_tracking_status()
    insights = tracker.get_learning_insights()

    print(f"📅 Time: {datetime.now().strftime('%H:%M:%S')}")
    print()

    # System status
    print("📊 SYSTEM STATUS:")
    print(f"   Active Games: {status['active_games']}")
    print(f"   Games Completed Today: {status['completed_games_today']}")
    print(".3f")
    print()

    # Live games
    tracker._update_live_games()
    live_games = tracker.get_live_games()

    print(f"🏟️ LIVE NFL GAMES ({len(live_games)}):")
    if live_games:
        for game in live_games:
            home = game.get('home_team', 'Unknown')
            away = game.get('away_team', 'Unknown')
            score = f"{game.get('home_score', 0)}-{game.get('away_score', 0)}"
            quarter = game.get('quarter', 1)

            # Get prediction
            prediction = game.get('prediction_home_win', 0.5)
            confidence = game.get('prediction_confidence', 0.5)

            print(f"   {home} vs {away}: {score} (Q{quarter})")
            print(".3f")
            print(".3f")

            # Betting advice
            if confidence > 0.7:
                team = home if prediction > 0.5 else away
                print(f"   💰 STRONG BET: {team} ML")
            elif confidence > 0.6:
                team = home if prediction > 0.5 else away
                print(f"   🎯 CONSIDER: {team} ML")
            else:
                print("   ⚠️ WAIT: Low confidence")
            print()
    else:
        print("   No live games currently tracked")
        print("   (NFL season starts September 2025)")
        print()

    # Learning status
    print("🧠 AI LEARNING:")
    print(f"   Games Processed: {insights['games_processed']}")
    print(f"   Predictions Made: {insights['total_predictions']}")
    print(f"   Models Updated: {insights['models_updated']}")
    print()

    print("🎯 RUN THIS AGAIN ANYTIME:")
    print("   python3 check_nfl_status.py")
    print()
    print("💡 ENJOY THE NFL GAMES!")
    print("   Your AI is learning and improving! 🏈🤖")

if __name__ == "__main__":
    main()
