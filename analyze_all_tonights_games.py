#!/usr/bin/env python3
"""
Comprehensive GGUF ensemble analysis for all tonight's NFL games.
This will help evaluate system performance across multiple scenarios.
"""

import asyncio
from datetime import datetime, timezone
from practical_gguf_ensemble import PracticalGGUFEnsemble
from football_odds_fetcher import FootballOddsFetcher
from api_config import get_api_keys

async def analyze_all_tonights_games():
    """Analyze all of tonight's NFL games with the GGUF ensemble."""
    
    print("🏈 COMPREHENSIVE TONIGHT'S GAMES ANALYSIS")
    print("=" * 70)
    print("🎯 Objective: Test GGUF ensemble across multiple game scenarios")
    print("=" * 70)
    
    # Get API keys and fetch odds
    api_keys = get_api_keys()
    
    print("\n📊 Fetching current NFL odds...")
    async with FootballOddsFetcher(api_keys['odds_api']) as fetcher:
        odds = await fetcher.get_all_odds_with_props()
    
    if not odds.games:
        print("❌ No NFL games found")
        return
    
    # Filter for tonight's games (within next 12 hours or recently started)
    current_time = datetime.now(timezone.utc)
    tonights_games = []
    
    for game in odds.games:
        game_time = datetime.fromisoformat(game.commence_time.replace('Z', '+00:00'))
        time_diff = (game_time - current_time).total_seconds() / 3600  # hours
        
        # Games within next 12 hours or started within last 4 hours
        if -4 <= time_diff <= 12:
            tonights_games.append((game, time_diff))
    
    tonights_games.sort(key=lambda x: x[1])
    
    if not tonights_games:
        print("❌ No games found for tonight")
        return
    
    print(f"\n🎮 Found {len(tonights_games)} games to analyze:")
    for i, (game, hours_away) in enumerate(tonights_games, 1):
        status = f"Starts in {hours_away:.1f}h" if hours_away > 0 else f"Started {abs(hours_away):.1f}h ago"
        print(f"   {i}. {game.away_team} @ {game.home_team} ({status})")
    
    # Initialize GGUF ensemble
    print(f"\n🤖 Initializing GGUF Ensemble...")
    ensemble = PracticalGGUFEnsemble()
    
    # Analyze each game
    all_results = []
    
    for game_num, (game, hours_away) in enumerate(tonights_games, 1):
        print(f"\n" + "="*70)
        print(f"🏟️  GAME {game_num}/{len(tonights_games)}: {game.away_team} @ {game.home_team}")
        print("="*70)
        
        # Get odds for this game
        game_h2h = next((h2h for h2h in odds.h2h_bets if h2h.game_id == game.game_id), None)
        game_spread = next((spread for spread in odds.spread_bets if spread.game_id == game.game_id), None)
        game_total = next((total for total in odds.total_bets if total.game_id == game.game_id), None)
        
        # Display game info
        game_time = datetime.fromisoformat(game.commence_time.replace('Z', '+00:00'))
        status = f"🕐 Starts in {hours_away:.1f} hours" if hours_away > 0 else f"🔴 Started {abs(hours_away):.1f} hours ago"
        print(f"⏰ {game_time.strftime('%I:%M %p ET')} - {status}")
        
        if game_h2h:
            print(f"💰 Moneyline: {game_h2h.away_team} {game_h2h.away_odds:+.0f} | {game_h2h.home_team} {game_h2h.home_odds:+.0f}")
        if game_spread:
            print(f"📏 Spread: {game_spread.away_team} {game_spread.away_spread:+.1f} | {game_spread.home_team} {game_spread.home_spread:+.1f}")
        if game_total:
            print(f"📈 Total: O/U {game_total.total_points}")
        
        # Create game data for ensemble
        game_data = {
            'home_team': game.home_team,
            'away_team': game.away_team,
            'season': '2024',
            'week': 'Week 7',
            'commence_time': game.commence_time,
            'game_status': 'live' if hours_away <= 0 else 'upcoming'
        }
        
        if game_spread:
            game_data['spread'] = game_spread.home_spread
            game_data['spread_odds_away'] = game_spread.away_odds
            game_data['spread_odds_home'] = game_spread.home_odds
        if game_total:
            game_data['total'] = game_total.total_points
            game_data['over_odds'] = game_total.over_odds
            game_data['under_odds'] = game_total.under_odds
        if game_h2h:
            game_data['moneyline_away'] = game_h2h.away_odds
            game_data['moneyline_home'] = game_h2h.home_odds
        
        # Run ensemble analysis
        print(f"\n🧠 Running GGUF Ensemble Analysis...")
        print(f"   📊 Using 3 models for comprehensive analysis")
        
        try:
            start_time = datetime.now()
            results = ensemble.get_ensemble_prediction(
                game_data=game_data,
                num_models=3
            )
            analysis_time = (datetime.now() - start_time).total_seconds()
            
            if results:
                print(f"\n✅ ENSEMBLE RESULTS (Analysis time: {analysis_time:.1f}s)")
                print("-" * 50)
                print(f"🎲 Win Probability: {results['probability']:.1%}")
                print(f"🎯 Confidence: {results['confidence']:.1%}")
                print(f"⚠️  Risk Level: {results['risk_level'].title()}")
                print(f"💡 Recommendation: {results['recommendation']}")
                
                if results.get('key_factors'):
                    print(f"\n🔍 Key Factors:")
                    for factor in results['key_factors'][:5]:  # Top 5 factors
                        print(f"   • {factor}")
                
                if results.get('analysis'):
                    print(f"\n📝 Analysis:")
                    print(f"   {results['analysis']}")
                
                print(f"\n🤖 Models: {', '.join(results.get('models_used', []))}")
                
                # Store result for summary
                game_result = {
                    'game': f"{game.away_team} @ {game.home_team}",
                    'status': 'live' if hours_away <= 0 else 'upcoming',
                    'probability': results['probability'],
                    'confidence': results['confidence'],
                    'risk_level': results['risk_level'],
                    'recommendation': results['recommendation'],
                    'analysis_time': analysis_time,
                    'models_used': len(results.get('models_used', [])),
                    'spread': game_spread.home_spread if game_spread else None,
                    'total': game_total.total_points if game_total else None
                }
                all_results.append(game_result)
                
            else:
                print("❌ No results returned from ensemble")
                all_results.append({
                    'game': f"{game.away_team} @ {game.home_team}",
                    'status': 'failed',
                    'analysis_time': analysis_time
                })
                
        except Exception as e:
            print(f"❌ Ensemble analysis failed: {e}")
            all_results.append({
                'game': f"{game.away_team} @ {game.home_team}",
                'status': 'error',
                'error': str(e)
            })
    
    # Summary analysis
    print(f"\n" + "="*70)
    print("📊 COMPREHENSIVE ANALYSIS SUMMARY")
    print("="*70)
    
    successful_analyses = [r for r in all_results if r.get('probability') is not None]
    
    print(f"\n🎯 Overall Performance:")
    print(f"   Total Games Analyzed: {len(all_results)}")
    print(f"   Successful Analyses: {len(successful_analyses)}")
    print(f"   Success Rate: {len(successful_analyses)/len(all_results)*100:.1f}%")
    
    if successful_analyses:
        avg_confidence = sum(r['confidence'] for r in successful_analyses) / len(successful_analyses)
        avg_time = sum(r['analysis_time'] for r in successful_analyses) / len(successful_analyses)
        avg_models = sum(r['models_used'] for r in successful_analyses) / len(successful_analyses)
        
        print(f"   Average Confidence: {avg_confidence:.1%}")
        print(f"   Average Analysis Time: {avg_time:.1f}s")
        print(f"   Average Models Used: {avg_models:.1f}")
        
        print(f"\n🎲 Game-by-Game Results:")
        for i, result in enumerate(successful_analyses, 1):
            status_emoji = "🔴" if result['status'] == 'live' else "🕐"
            print(f"   {i}. {result['game']} {status_emoji}")
            print(f"      Probability: {result['probability']:.1%} | Confidence: {result['confidence']:.1%}")
            print(f"      Risk: {result['risk_level'].title()} | Rec: {result['recommendation']}")
        
        # Risk analysis
        risk_levels = [r['risk_level'] for r in successful_analyses]
        risk_distribution = {
            'low': risk_levels.count('low'),
            'medium': risk_levels.count('medium'), 
            'high': risk_levels.count('high')
        }
        print(f"\n⚠️  Risk Distribution:")
        for risk, count in risk_distribution.items():
            print(f"   {risk.title()}: {count} games ({count/len(successful_analyses)*100:.1f}%)")
        
        # Betting recommendations analysis
        recommendations = [r['recommendation'].lower() for r in successful_analyses]
        bet_types = {}
        for rec in recommendations:
            if 'under' in rec:
                bet_types['under'] = bet_types.get('under', 0) + 1
            elif 'over' in rec:
                bet_types['over'] = bet_types.get('over', 0) + 1
            elif 'spread' in rec:
                bet_types['spread'] = bet_types.get('spread', 0) + 1
            elif 'moneyline' in rec or 'ml' in rec:
                bet_types['moneyline'] = bet_types.get('moneyline', 0) + 1
        
        if bet_types:
            print(f"\n💡 Recommendation Patterns:")
            for bet_type, count in bet_types.items():
                print(f"   {bet_type.title()}: {count} games")
    
    print(f"\n✅ Comprehensive analysis complete!")
    print(f"🔄 System ready for next analysis cycle")

if __name__ == "__main__":
    asyncio.run(analyze_all_tonights_games())