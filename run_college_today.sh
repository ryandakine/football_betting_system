#!/bin/bash
# Quick runner for College Football Today Analysis

echo "🏈 College Football Today - Quick Runner"
echo "========================================"
echo ""

# Check if API key is set
if [ -z "$THE_ODDS_API_KEY" ]; then
    echo "⚠️  THE_ODDS_API_KEY not set - will use MOCK data"
    echo ""
    echo "To use REAL odds, set your API key:"
    echo "  export THE_ODDS_API_KEY='your_api_key_here'"
    echo ""
    echo "Get a free API key at: https://the-odds-api.com/"
    echo ""
    echo "Running with mock data in 3 seconds..."
    sleep 3
else
    echo "✅ API key found - will fetch REAL odds"
    echo ""
fi

# Run the analyzer
python3 college_football_today.py
