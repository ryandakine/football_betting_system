#!/usr/bin/env python3
"""
Get today's actual NFL games using your odds fetcher system
"""

from football_odds_fetcher import FootballOddsFetcher
from datetime import datetime

def main():
    print('🏈 FETCHING ACTUAL NFL GAMES FROM YOUR SYSTEM')
    print('='*60)

    # Initialize your odds fetcher
    try:
        fetcher = FootballOddsFetcher()
        print('✅ Odds fetcher initialized')
    except Exception as e:
        print(f'❌ Failed to initialize fetcher: {e}')
        return

    try:
        print('📊 Getting current NFL games...')
        
        # Fetch games using your system
        games = fetcher.fetch_all_odds()
        
        if games:
            print(f'✅ Found {len(games)} games')
            print()
            
            # Show all games
            today = datetime.now().date()
            
            for i, game in enumerate(games[:10], 1):  # Show first 10
                print(f'{i}. {game.away_team} @ {game.home_team}')
                
                if hasattr(game, 'commence_time') and game.commence_time:
                    print(f'   Time: {game.commence_time}')
                
                # Show betting lines if available
                if hasattr(game, 'spread_bet') and game.spread_bet:
                    print(f'   Spread: {game.spread_bet.line}')
                if hasattr(game, 'total_bet') and game.total_bet:
                    print(f'   Total: {game.total_bet.line}')
                    
                print()
                
        else:
            print('❌ No games returned from your fetcher')
            
    except Exception as e:
        print(f'❌ Error fetching games: {e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()