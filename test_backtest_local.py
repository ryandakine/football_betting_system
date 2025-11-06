#!/usr/bin/env python3
"""
Test the backtesting analyzer with local data
"""

import json
import boto3
from datetime import datetime

s3 = boto3.client('s3')
BUCKET = 'football-betting-system-data'

# Fetch today's odds data
response = s3.get_object(Bucket=BUCKET, Key='odds/2025-10-27/2025-10-27T04:21:12.839615.json')
odds_data = json.loads(response['Body'].read())

print(f"\n📊 BACKTEST ANALYSIS")
print(f"=" * 60)
print(f"Games found: {len(odds_data)}")
print()

# Analyze line discrepancies
total_discrepancies = 0
valuable_opportunities = 0

for game in odds_data:
    home = game['home_team']
    away = game['away_team']
    
    # Get all spreads for each team
    spreads_home = []
    spreads_away = []
    
    for bookmaker in game.get('bookmakers', []):
        for market in bookmaker.get('markets', []):
            if market['key'] == 'spreads':
                for outcome in market['outcomes']:
                    if outcome['name'] == home:
                        spreads_home.append({
                            'book': bookmaker['title'],
                            'point': outcome['point'],
                            'price': outcome['price']
                        })
                    elif outcome['name'] == away:
                        spreads_away.append({
                            'book': bookmaker['title'],
                            'point': outcome['point'],
                            'price': outcome['price']
                        })
    
    # Find discrepancies
    if len(spreads_home) >= 2:
        best_home = min(spreads_home, key=lambda x: abs(x['point']))
        worst_home = max(spreads_home, key=lambda x: abs(x['point']))
        disc_home = abs(best_home['point'] - worst_home['point'])
        
        if disc_home >= 0.5:
            total_discrepancies += 1
            value = disc_home * 0.02  # 2% per point
            
            if value >= 0.03:  # 3% edge
                valuable_opportunities += 1
                print(f"💰 {home} vs {away}")
                print(f"   Spread discrepancy: {disc_home} points")
                print(f"   Best: {best_home['point']} @ {best_home['book']}")
                print(f"   Worst: {worst_home['point']} @ {worst_home['book']}")
                print(f"   Estimated edge: {value*100:.1f}%")
                print()

print(f"\n📈 SUMMARY")
print(f"=" * 60)
print(f"Total discrepancies found: {total_discrepancies}")
print(f"Valuable opportunities (>3% edge): {valuable_opportunities}")
print(f"Value rate: {valuable_opportunities/len(odds_data)*100:.1f}% of games")
