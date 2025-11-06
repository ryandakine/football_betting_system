#!/usr/bin/env python3
"""
NFL Live Tracking Dashboard
===========================
Real-time dashboard for monitoring NFL live game tracking and learning progress.
"""

import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
import os

class NFLTrackingDashboard:
    """Dashboard for monitoring NFL live tracking system."""

    def __init__(self, db_path: str = "data/nfl_live_tracking.db"):
        self.db_path = db_path

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                # Current live games
                cursor.execute("SELECT COUNT(*) FROM live_games WHERE status = 'in_progress'")
                active_games = cursor.fetchone()[0]

                # Today's completed games
                today = datetime.now().strftime("%Y-%m-%d")
                cursor.execute("""
                    SELECT COUNT(*) FROM game_outcomes
                    WHERE recorded_at LIKE ?
                """, (f"{today}%",))
                completed_today = cursor.fetchone()[0]

                # Learning metrics
                cursor.execute("""
                    SELECT games_processed, models_updated, current_accuracy, predictions_made
                    FROM learning_metrics
                    ORDER BY timestamp DESC
                    LIMIT 1
                """)
                latest_metrics = cursor.fetchone()

                # Recent predictions
                cursor.execute("""
                    SELECT prediction_correct, prediction_confidence
                    FROM game_outcomes
                    WHERE recorded_at > datetime('now', '-24 hours')
                """)
                recent_predictions = cursor.fetchall()

                # Calculate recent accuracy
                if recent_predictions:
                    correct = sum(1 for pred in recent_predictions if pred[0] == 1)
                    total = len(recent_predictions)
                    recent_accuracy = correct / total if total > 0 else 0
                    avg_confidence = sum(pred[1] for pred in recent_predictions) / total
                else:
                    recent_accuracy = 0
                    avg_confidence = 0

                # Live games details
                cursor.execute("""
                    SELECT home_team, away_team, home_score, away_score, quarter,
                           prediction_home_win, confidence
                    FROM live_games
                    WHERE status = 'in_progress'
                    ORDER BY last_updated DESC
                    LIMIT 10
                """)
                live_games_data = cursor.fetchall()

                return {
                    "timestamp": datetime.now().isoformat(),
                    "active_games": active_games,
                    "completed_today": completed_today,
                    "learning_metrics": {
                        "games_processed": latest_metrics[0] if latest_metrics else 0,
                        "models_updated": latest_metrics[1] if latest_metrics else 0,
                        "current_accuracy": latest_metrics[2] if latest_metrics else 0,
                        "predictions_made": latest_metrics[3] if latest_metrics else 0,
                        "recent_accuracy": recent_accuracy,
                        "avg_confidence": avg_confidence
                    },
                    "live_games": [
                        {
                            "home_team": game[0],
                            "away_team": game[1],
                            "score": f"{game[2]}-{game[3]}",
                            "quarter": game[4],
                            "prediction": game[5],
                            "confidence": game[6]
                        }
                        for game in live_games_data
                    ]
                }

        except Exception as e:
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    def display_dashboard(self, data: Dict[str, Any]):
        """Display the dashboard in a formatted way."""
        if "error" in data:
            print(f"❌ Dashboard Error: {data['error']}")
            return

        print("🏈 NFL Live Tracking Dashboard")
        print("=" * 50)
        print(f"📅 Time: {data['timestamp'][:19].replace('T', ' ')}")
        print()

        # Status Overview
        print("📊 System Status:")
        print(f"   🏟️ Active Games: {data['active_games']}")
        print(f"   ✅ Completed Today: {data['completed_today']}")
        print()

        # Learning Metrics
        metrics = data['learning_metrics']
        print("🧠 Learning Performance:")
        print(f"   📈 Games Processed: {metrics['games_processed']}")
        print(f"   🔄 Models Updated: {metrics['models_updated']}")
        print(".3f")
        print(f"   🎯 Predictions Made: {metrics['predictions_made']}")
        print(".3f")
        print(".3f")
        print()

        # Live Games
        live_games = data['live_games']
        if live_games:
            print("🏟️ Live Games:")
            for game in live_games[:5]:  # Show top 5
                pred_symbol = "🏠" if game['prediction'] > 0.5 else "✈️"
                conf_level = "HIGH" if game['confidence'] > 0.7 else "MED" if game['confidence'] > 0.5 else "LOW"
                print(f"   {game['home_team']} vs {game['away_team']}: {game['score']} (Q{game['quarter']})")
                print(".3f"
        else:
            print("🏟️ No live games currently tracked")
            print()

        print("🔄 Dashboard updates automatically every 30 seconds")
        print("💡 System is learning and improving in real-time!")

def main():
    """Main dashboard loop."""
    dashboard = NFLTrackingDashboard()

    print("🏈 NFL Live Tracking Dashboard - 2025 Season")
    print("🎯 System ready for September 2025 NFL season kickoff")
    print("Press Ctrl+C to exit")
    print()

    try:
        while True:
            # Clear screen (Unix systems)
            os.system('clear' if os.name != 'nt' else 'cls')

            # Get and display data
            data = dashboard.get_dashboard_data()
            dashboard.display_dashboard(data)

            # Add 2025 season preparation info
            print("\n📅 2025 NFL Season Preparation:")
            print("   • Thursday Night Football: Ready for Week 1")
            print("   • Sunday Slate Coverage: Multi-game tracking active")
            print("   • Learning Systems: Calibrated for 2025 rosters")
            print("   • Model Updates: Ready for season-long improvement")

            # Wait before next update
            time.sleep(30)

    except KeyboardInterrupt:
        print("\n⏹️ Dashboard stopped - See you for the 2025 NFL season!")
    except Exception as e:
        print(f"\n❌ Dashboard error: {e}")

if __name__ == "__main__":
    main()
