#!/usr/bin/env python3
"""
Launch Script for Football Betting Master System
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

def main():
    """Launch the Football Betting Master System"""
    print("🏈 Football Betting Master System")
    print("=" * 50)
    print()

    print("🤖 AI Intelligence Status:")
    print("  ✅ Premium Providers: Perplexity AI, ChatGPT")
    print("  ✅ Free Backup: HuggingFace")
    print("  ✅ Smart Fallbacks: User permission required")
    print()

    print("📊 Data Sources:")
    print("  ✅ Real Odds: The Odds API (FanDuel, etc.)")
    print("  ✅ Live Scores: ESPN & NFL Official APIs")
    print("  ✅ Game Data: Real-time game information")
    print()

    print("🎯 System Features:")
    print("  ✅ Predict All Games - Mass AI analysis")
    print("  ✅ Individual Predictions - Single game analysis")
    print("  ✅ Learning System - Improves over time")
    print("  ✅ Mobile Responsive - Any screen size")
    print("  ✅ Offline Caching - Data persistence")
    print()

    print("🚀 Launching GUI...")
    print("Note: Close the terminal window to exit the system")
    print()

    try:
        from football_master_gui import FootballMasterGUI

        # Create and run the GUI
        gui = FootballMasterGUI()

        # This will block until the GUI is closed
        gui.root.mainloop()

    except KeyboardInterrupt:
        print("\n👋 System shutdown requested by user")
    except Exception as e:
        print(f"\n❌ Error launching system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()