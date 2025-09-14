#!/usr/bin/env python3
"""External AI Council Trigger Script for n8n Integration"""

import json
import sys
from datetime import datetime

import requests


def fetch_odds_data():
    """Fetch current MLB odds from The Odds API"""
    try:
        # You'll need to set this environment variable or replace with your key
        api_key = "your_odds_api_key_here"  # Replace with actual key

        url = "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
        params = {
            "regions": "us",
            "markets": "h2h",
            "oddsFormat": "american",
            "apiKey": api_key,
        }

        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        print(f"Error fetching odds: {e}", file=sys.stderr)
        return []


def call_ai_council(odds_data):
    """Call the AI Council with odds data"""
    try:
        url = "http://localhost:8767/opportunities"
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=odds_data, headers=headers, timeout=30)
        response.raise_for_status()

        return response.json()
    except Exception as e:
        print(f"Error calling AI Council: {e}", file=sys.stderr)
        return {"error": str(e), "predictions": []}


def main():
    """Main function to orchestrate the process"""
    try:
        # Fetch odds data
        odds_data = fetch_odds_data()

        if not odds_data:
            result = {"error": "No odds data available", "predictions": []}
        else:
            # Call AI Council
            ai_response = call_ai_council(odds_data)
            result = ai_response

        # Add timestamp
        result["timestamp"] = datetime.now().isoformat()

        # Output as JSON for n8n
        print(json.dumps(result))

    except Exception as e:
        error_result = {
            "error": str(e),
            "predictions": [],
            "timestamp": datetime.now().isoformat(),
        }
        print(json.dumps(error_result))


if __name__ == "__main__":
    main()
