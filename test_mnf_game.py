#!/usr/bin/env python3
"""
Test the GGUF ensemble on tonight's Monday Night Football game.
"""

import asyncio
from datetime import datetime
from practical_gguf_ensemble import PracticalGGUFEnsemble
from football_odds_fetcher import FootballOddsFetcher
from api_config import get_api_keys

async def analyze_mnf_game():
    """Analyze tonight's Monday Night Football game."""
    
    print("🏈 Monday Night Football GGUF Analysis")
    print("=" * 60)
    
    # Get API keys
    api_keys = get_api_keys()
    
    # Fetch today's games
    print("📊 Fetching current NFL odds...")
    async with FootballOddsFetcher(api_keys['odds_api']) as fetcher:
        odds = await fetcher.get_all_odds_with_props()
    
    if not odds.games:
        print("❌ No NFL games found for today")
        return
    
    # Find tonight's primetime game (earliest game)
    tonight_game = min(odds.games, key=lambda g: g.commence_time)
    
    print(f"\n🎯 Tonight's Game:")
    print(f"   🏟️  {tonight_game.away_team} @ {tonight_game.home_team}")
    print(f"   ⏰ {tonight_game.commence_time}")
    
    # Get odds for this game
    game_h2h = next((h2h for h2h in odds.h2h_bets if h2h.game_id == tonight_game.game_id), None)
    game_spread = next((spread for spread in odds.spread_bets if spread.game_id == tonight_game.game_id), None)
    game_total = next((total for total in odds.total_bets if total.game_id == tonight_game.game_id), None)
    
    if game_h2h:
        print(f"   💰 Moneyline: {game_h2h.away_team} {game_h2h.away_odds:+.0f} | {game_h2h.home_team} {game_h2h.home_odds:+.0f}")
    if game_spread:
        print(f"   📏 Spread: {game_spread.away_team} {game_spread.away_spread:+.1f} | {game_spread.home_team} {game_spread.home_spread:+.1f}")
    if game_total:
        print(f"   📈 Total: O/U {game_total.total_points}")
    
    # Initialize GGUF ensemble
    print("\n🤖 Initializing GGUF Ensemble...")
    ensemble = PracticalGGUFEnsemble()
    
    # Create analysis prompt
    analysis_prompt = f"""
Analyze this Monday Night Football game for betting value:

Game: {tonight_game.away_team} @ {tonight_game.home_team}
Time: {tonight_game.commence_time}

Current Odds:
"""
    
    if game_h2h:
        analysis_prompt += f"- Moneyline: {game_h2h.away_team} {game_h2h.away_odds:+.0f}, {game_h2h.home_team} {game_h2h.home_odds:+.0f}\n"
    if game_spread:
        analysis_prompt += f"- Spread: {game_spread.away_team} {game_spread.away_spread:+.1f} ({game_spread.away_odds:+.0f}), {game_spread.home_team} {game_spread.home_spread:+.1f} ({game_spread.home_odds:+.0f})\n"
    if game_total:
        analysis_prompt += f"- Total: Over {game_total.total_points} ({game_total.over_odds:+.0f}), Under {game_total.total_points} ({game_total.under_odds:+.0f})\n"
    
    analysis_prompt += """
Provide analysis covering:
1. Team matchup and recent form
2. Key injuries and lineup changes
3. Weather/venue factors
4. Betting value opportunities
5. Recommended plays with confidence (1-10)

Focus on uncensored, sharp analysis for profitable betting angles.
"""
    
    # Create game data dictionary for ensemble
    game_data = {
        'home_team': tonight_game.home_team,
        'away_team': tonight_game.away_team,
        'season': '2024',
        'week': 'TBD',
        'commence_time': tonight_game.commence_time
    }
    
    if game_spread:
        game_data['spread'] = game_spread.home_spread
    if game_total:
        game_data['total'] = game_total.total_points
    
    # Get ensemble predictions
    print("\n🧠 Running GGUF Ensemble Analysis...")
    print("   (This may take 2-3 minutes with CPU inference)")
    
    try:
        results = ensemble.get_ensemble_prediction(
            game_data=game_data,
            num_models=3  # Use 3 models for speed
        )
        
        if results:
            print("\n" + "="*60)
            print("🎯 GGUF ENSEMBLE ANALYSIS RESULTS")
            print("="*60)
            
            print(f"\n🏆 ENSEMBLE RESULT:")
            print(f"   🎲 Win Probability: {results['probability']:.1%}")
            print(f"   🎯 Confidence: {results['confidence']:.1%}")
            print(f"   ⚠️  Risk Level: {results['risk_level'].title()}")
            print(f"   💡 Recommendation: {results['recommendation']}")
            
            if results.get('key_factors'):
                print(f"\n🔍 Key Factors:")
                for factor in results['key_factors']:
                    print(f"   • {factor}")
            
            if results.get('analysis'):
                print(f"\n📝 Analysis:")
                print(f"   {results['analysis']}")
            
            print(f"\n📊 Ensemble Stats:")
            print(f"   Models Used: {results.get('ensemble_size', 0)}")
            print(f"   Model Names: {', '.join(results.get('models_used', []))}")
            print(f"   Specialties: {', '.join(results.get('model_specialties', []))}")
        else:
            print("❌ No results returned from ensemble")
        
    except Exception as e:
        print(f"❌ Ensemble analysis failed: {e}")
    
    finally:
        print("\n✅ Analysis complete!")

if __name__ == "__main__":
    asyncio.run(analyze_mnf_game())