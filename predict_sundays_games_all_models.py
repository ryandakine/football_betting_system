#!/usr/bin/env python3
"""
Run predictions on Sunday's games using ALL 5 GGUF models for maximum accuracy.
"""

import sys
import requests
from datetime import datetime, timezone
from practical_gguf_ensemble import PracticalGGUFEnsemble

def main():
    print("🏈 NFL PREDICTIONS - ALL 5 GGUF MODELS")
    print("=" * 80)
    
    # Get games
    api_key = 'e84d496405014d166f5dce95094ea024'
    url = f'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey={api_key}&regions=us&markets=h2h,spreads,totals'
    
    print("\n📊 Fetching games from The Odds API...")
    resp = requests.get(url, timeout=10)
    games = resp.json()
    
    # Filter for Sunday games (next 48-72 hours)
    now = datetime.now(timezone.utc)
    sunday_games = []
    
    for g in games:
        game_time = datetime.fromisoformat(g['commence_time'].replace('Z', '+00:00'))
        hours = (game_time - now).total_seconds() / 3600
        
        # Games in next 24-72 hours (Sunday)
        if 24 <= hours <= 72:
            sunday_games.append((g, hours))
    
    sunday_games.sort(key=lambda x: x[1])
    
    print(f"✅ Found {len(sunday_games)} Sunday games\n")
    
    # Load ensemble
    print("🤖 Loading GGUF Ensemble (all 5 models)...")
    ensemble = PracticalGGUFEnsemble()
    print("✅ Ensemble ready\n")
    
    # Analyze each game
    results = []
    
    for i, (game_data, hours) in enumerate(sunday_games, 1):
        print("=" * 80)
        print(f"🏟️  GAME {i}/{len(sunday_games)}: {game_data['away_team']} @ {game_data['home_team']}")
        print("=" * 80)
        
        # Extract odds
        spread_line = None
        total_line = None
        home_ml = None
        away_ml = None
        
        for book in game_data['bookmakers'][:1]:
            for market in book['markets']:
                if market['key'] == 'spreads':
                    for outcome in market['outcomes']:
                        if outcome['name'] == game_data['home_team']:
                            spread_line = outcome['point']
                elif market['key'] == 'totals':
                    total_line = market['outcomes'][0]['point']
                elif market['key'] == 'h2h':
                    for outcome in market['outcomes']:
                        if outcome['name'] == game_data['home_team']:
                            home_ml = outcome['price']
                        else:
                            away_ml = outcome['price']
        
        print(f"📏 Spread: {game_data['home_team']} {spread_line}")
        print(f"📈 Total: {total_line}")
        print(f"💰 ML: {game_data['away_team']} {away_ml:+.0f} | {game_data['home_team']} {home_ml:+.0f}")
        
        # Prepare game data
        game = {
            'home_team': game_data['home_team'],
            'away_team': game_data['away_team'],
            'spread': spread_line,
            'total': total_line,
            'moneyline_home': home_ml,
            'moneyline_away': away_ml,
            'season': '2024',
            'week': 'Week 10'
        }
        
        print(f"\n🧠 Running ALL 5 GGUF models (this takes ~2-3 minutes)...")
        
        try:
            start = datetime.now()
            
            # Use ALL 5 models
            result = ensemble.get_ensemble_prediction(game, num_models=5)
            
            elapsed = (datetime.now() - start).total_seconds()
            
            if result:
                print(f"\n✅ PREDICTION (completed in {elapsed:.1f}s):")
                print("-" * 60)
                print(f"🎲 {game_data['home_team']} Win Probability: {result['probability']:.1%}")
                print(f"🎯 Ensemble Confidence: {result['confidence']:.1%}")
                print(f"⚠️  Risk Level: {result['risk_level'].title()}")
                print(f"💡 Recommendation: {result['recommendation']}")
                
                print(f"\n🤖 Models Contributing:")
                for model in result.get('models_used', []):
                    print(f"   ✓ {model}")
                
                if result.get('key_factors'):
                    print(f"\n🔍 Key Factors:")
                    for factor in result['key_factors'][:3]:
                        print(f"   • {factor}")
                
                results.append({
                    'game': f"{game_data['away_team']} @ {game_data['home_team']}",
                    'probability': result['probability'],
                    'confidence': result['confidence'],
                    'recommendation': result['recommendation'],
                    'spread': spread_line,
                    'pick': game_data['home_team'] if result['probability'] > 0.55 else game_data['away_team']
                })
            else:
                print("❌ No prediction returned")
        
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 BETTING RECOMMENDATIONS SUMMARY")
    print("=" * 80)
    
    high_conf = [r for r in results if r['confidence'] > 0.70]
    
    print(f"\n🎯 High Confidence Picks ({len(high_conf)} games):\n")
    
    for i, r in enumerate(high_conf, 1):
        prob = r['probability']
        pick_type = "HOME" if prob > 0.55 else "AWAY"
        edge = abs(prob - 0.5) * 100
        
        print(f"{i}. {r['game']}")
        print(f"   Pick: {r['pick']} ({pick_type})")
        print(f"   Edge: {edge:.1f}% | Confidence: {r['confidence']:.1%}")
        print(f"   Recommendation: {r['recommendation']}")
        print()

if __name__ == "__main__":
    main()
