#!/usr/bin/env python3
"""Quick fetch for tonight's TNF odds"""

import os
import sys
from datetime import date
from tri_model_api_config import get_trimodel_api_keys
from nfl_odds_integration import fetch_and_integrate_nfl_odds
import asyncio

# Set API key from config
os.environ['THE_ODDS_API_KEY'] = get_trimodel_api_keys()['odds_api']

# Fetch for Oct 24 (UTC date of tonight's TNF game)
target = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2025, 10, 24)

print(f"🏈 Fetching NFL odds for {target}...")
result = asyncio.run(fetch_and_integrate_nfl_odds(target))

print("\n" + "=" * 80)
print(f"✅ SUCCESS - {result['games_count']} games found")
print("=" * 80)
