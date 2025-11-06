import json
from datetime import datetime

from flask import Flask, jsonify, request

app = Flask(__name__)

# Store fresh odds data
fresh_odds = {}


@app.route("/update-odds", methods=["POST"])
def update_odds():
    """Receive fresh odds from N8N"""
    try:
        game_data = request.json

        # Store the fresh data
        fresh_odds[game_data["gameId"]] = game_data

        # Log the update
        print(f"📊 Updated odds for: {game_data['matchup']}")

        # Special logging for Tigers games
        if game_data.get("isTigersGame"):
            print(f"🐅 FRESH TIGERS ODDS RECEIVED!")
            print(f"   Matchup: {game_data['matchup']}")
            print(f"   Fresh Tigers odds: {game_data['bestOdds']['away']['price']}")
            print(f"   Bookmaker: {game_data['bestOdds']['away']['bookmaker']}")
            print(f"   ✅ This replaces stale +120 data!")

        return jsonify(
            {
                "status": "success",
                "game": game_data["matchup"],
                "timestamp": datetime.now().isoformat(),
            }
        )

    except Exception as e:
        print(f"❌ Error processing odds: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400


@app.route("/get-odds", methods=["GET"])
def get_odds():
    """Get all stored fresh odds"""
    return jsonify(
        {
            "total_games": len(fresh_odds),
            "games": fresh_odds,
            "last_update": datetime.now().isoformat(),
        }
    )


if __name__ == "__main__":
    print("🚀 Starting odds receiver server...")
    print("📡 Listening for fresh odds from N8N...")
    app.run(host="0.0.0.0", port=5000, debug=True)
