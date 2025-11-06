import requests
import pandas as pd
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

class NFLDataLoader:
    def __init__(self, db_path: str = "nfl_historical.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database for storing historical data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Games table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                date TEXT,
                home_team TEXT,
                away_team TEXT,
                home_score INTEGER,
                away_score INTEGER,
                week INTEGER,
                season INTEGER,
                game_type TEXT,
                weather TEXT,
                surface TEXT,
                roof TEXT,
                temp INTEGER,
                wind INTEGER
            )
        ''')
        
        # Referee crews table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS referee_crews (
                game_id TEXT,
                crew_id TEXT,
                referee TEXT,
                umpire TEXT,
                down_judge TEXT,
                line_judge TEXT,
                field_judge TEXT,
                side_judge TEXT,
                back_judge TEXT,
                replay_official TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        # Penalties table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS penalties (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id TEXT,
                penalty_type TEXT,
                team TEXT,
                quarter INTEGER,
                time_remaining TEXT,
                yards INTEGER,
                result TEXT,
                ref_position TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        # Betting lines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS betting_lines (
                game_id TEXT,
                timestamp TEXT,
                spread REAL,
                total REAL,
                home_ml INTEGER,
                away_ml INTEGER,
                source TEXT,
                FOREIGN KEY (game_id) REFERENCES games (game_id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def scrape_nfl_games(self, season: int, weeks: List[int] = None) -> List[Dict]:
        """Scrape NFL game data from ESPN or similar sources"""
        # This is a placeholder - you'd implement actual scraping
        # You could use ESPN's API or scrape from pro-football-reference
        
        games = []
        
        # Example structure - replace with actual scraping
        sample_games = [
            {
                'game_id': f'NFL_{season}_W1_KC_DET',
                'date': f'{season}-09-07',
                'home_team': 'DET',
                'away_team': 'KC',
                'home_score': 21,
                'away_score': 20,
                'week': 1,
                'season': season,
                'game_type': 'REG',
                'weather': 'Dome',
                'surface': 'Turf',
                'roof': 'Dome',
                'temp': 72,
                'wind': 0
            }
        ]
        
        return sample_games
    
    def scrape_referee_data(self, season: int) -> List[Dict]:
        """Scrape referee crew assignments and penalty data"""
        # This would scrape from NFL.com or pro-football-reference
        
        referee_data = [
            {
                'game_id': f'NFL_{season}_W1_KC_DET',
                'crew_id': 'CREW_7',
                'referee': 'John Smith',
                'umpire': 'Mike Johnson',
                'down_judge': 'Bob Wilson',
                'line_judge': 'Steve Davis',
                'field_judge': 'Tom Brown',
                'side_judge': 'Jim Miller',
                'back_judge': 'Dave Taylor',
                'replay_official': 'Mark Anderson'
            }
        ]
        
        return referee_data
    
    def scrape_penalty_data(self, game_id: str) -> List[Dict]:
        """Scrape detailed penalty data for a specific game"""
        # This would get play-by-play data and extract penalties
        
        penalties = [
            {
                'game_id': game_id,
                'penalty_type': 'Holding',
                'team': 'KC',
                'quarter': 4,
                'time_remaining': '3:12',
                'yards': 10,
                'result': 'Accepted',
                'ref_position': 'Back Judge'
            },
            {
                'game_id': game_id,
                'penalty_type': 'Pass Interference',
                'team': 'DET',
                'quarter': 2,
                'time_remaining': '8:45',
                'yards': 15,
                'result': 'Accepted',
                'ref_position': 'Field Judge'
            }
        ]
        
        return penalties
    
    def scrape_betting_lines(self, game_id: str, date: str) -> List[Dict]:
        """Scrape historical betting lines for a game"""
        # This would get line movements from various sportsbooks
        
        lines = [
            {
                'game_id': game_id,
                'timestamp': f'{date} 09:00:00',
                'spread': -7.0,
                'total': 52.5,
                'home_ml': -110,
                'away_ml': -110,
                'source': 'DraftKings'
            },
            {
                'game_id': game_id,
                'timestamp': f'{date} 15:00:00',
                'spread': -5.0,
                'total': 54.0,
                'home_ml': -105,
                'away_ml': -115,
                'source': 'DraftKings'
            },
            {
                'game_id': game_id,
                'timestamp': f'{date} 18:00:00',
                'spread': -4.5,
                'total': 54.0,
                'home_ml': -105,
                'away_ml': -115,
                'source': 'DraftKings'
            }
        ]
        
        return lines
    
    def load_historical_data(self, start_season: int, end_season: int) -> None:
        """Load historical data into database"""
        for season in range(start_season, end_season + 1):
            print(f"Loading data for {season} season...")
            
            # Load games
            games = self.scrape_nfl_games(season)
            self.store_games(games)
            
            # Load referee data
            referee_data = self.scrape_referee_data(season)
            self.store_referee_data(referee_data)
            
            # Load penalties and betting lines for each game
            for game in games:
                penalties = self.scrape_penalty_data(game['game_id'])
                self.store_penalties(penalties)
                
                betting_lines = self.scrape_betting_lines(game['game_id'], game['date'])
                self.store_betting_lines(betting_lines)
                
                # Rate limiting
                time.sleep(1)
    
    def store_games(self, games: List[Dict]) -> None:
        """Store games in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for game in games:
            cursor.execute('''
                INSERT OR REPLACE INTO games 
                (game_id, date, home_team, away_team, home_score, away_score, 
                 week, season, game_type, weather, surface, roof, temp, wind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                game['game_id'], game['date'], game['home_team'], game['away_team'],
                game['home_score'], game['away_score'], game['week'], game['season'],
                game['game_type'], game['weather'], game['surface'], game['roof'],
                game['temp'], game['wind']
            ))
        
        conn.commit()
        conn.close()
    
    def store_referee_data(self, referee_data: List[Dict]) -> None:
        """Store referee crew data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for crew in referee_data:
            cursor.execute('''
                INSERT OR REPLACE INTO referee_crews 
                (game_id, crew_id, referee, umpire, down_judge, line_judge, 
                 field_judge, side_judge, back_judge, replay_official)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                crew['game_id'], crew['crew_id'], crew['referee'], crew['umpire'],
                crew['down_judge'], crew['line_judge'], crew['field_judge'],
                crew['side_judge'], crew['back_judge'], crew['replay_official']
            ))
        
        conn.commit()
        conn.close()
    
    def store_penalties(self, penalties: List[Dict]) -> None:
        """Store penalty data in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for penalty in penalties:
            cursor.execute('''
                INSERT INTO penalties 
                (game_id, penalty_type, team, quarter, time_remaining, 
                 yards, result, ref_position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                penalty['game_id'], penalty['penalty_type'], penalty['team'],
                penalty['quarter'], penalty['time_remaining'], penalty['yards'],
                penalty['result'], penalty['ref_position']
            ))
        
        conn.commit()
        conn.close()
    
    def store_betting_lines(self, lines: List[Dict]) -> None:
        """Store betting lines in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for line in lines:
            cursor.execute('''
                INSERT INTO betting_lines 
                (game_id, timestamp, spread, total, home_ml, away_ml, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                line['game_id'], line['timestamp'], line['spread'], line['total'],
                line['home_ml'], line['away_ml'], line['source']
            ))
        
        conn.commit()
        conn.close()
    
    def get_historical_data_for_backtest(self, start_date: str, end_date: str) -> List[Dict]:
        """Get formatted historical data for backtesting"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT 
                g.game_id,
                g.date,
                g.home_team,
                g.away_team,
                g.home_score,
                g.away_score,
                rc.crew_id,
                rc.referee
            FROM games g
            JOIN referee_crews rc ON g.game_id = rc.game_id
            WHERE g.date BETWEEN ? AND ?
            ORDER BY g.date
        '''
        
        df = pd.read_sql_query(query, conn, params=(start_date, end_date))
        
        backtest_data = []
        
        for _, row in df.iterrows():
            game_id = row['game_id']
            
            # Get penalties for this game
            penalty_query = '''
                SELECT penalty_type, team, quarter, time_remaining, ref_position
                FROM penalties 
                WHERE game_id = ?
            '''
            penalties_df = pd.read_sql_query(penalty_query, conn, params=(game_id,))
            
            ref_events = []
            for _, penalty in penalties_df.iterrows():
                ref_events.append({
                    'ref_id': penalty['ref_position'][:6],  # Simplified
                    'penalty': penalty['penalty_type'],
                    'team': 'home' if penalty['team'] == row['home_team'] else 'away',
                    'quarter': penalty['quarter'],
                    'time': penalty['time_remaining']
                })
            
            # Get betting lines
            lines_query = '''
                SELECT timestamp, spread, total
                FROM betting_lines 
                WHERE game_id = ?
                ORDER BY timestamp
            '''
            lines_df = pd.read_sql_query(lines_query, conn, params=(game_id,))
            
            if not lines_df.empty:
                opening_spread = lines_df.iloc[0]['spread']
                closing_spread = lines_df.iloc[-1]['spread']
                opening_total = lines_df.iloc[0]['total']
                closing_total = lines_df.iloc[-1]['total']
            else:
                opening_spread = closing_spread = 0
                opening_total = closing_total = 45
            
            # Calculate outcomes
            home_score = row['home_score']
            away_score = row['away_score']
            total_score = home_score + away_score
            
            # Determine spread result
            if closing_spread > 0:  # Home favored
                spread_result = 'home_cover' if home_score - away_score > closing_spread else 'away_cover'
            else:  # Away favored
                spread_result = 'away_cover' if away_score - home_score > abs(closing_spread) else 'home_cover'
            
            # Determine total result
            total_result = 'over' if total_score > closing_total else 'under'
            
            game_data = {
                'game_id': game_id,
                'date': row['date'],
                'home_team': row['home_team'],
                'away_team': row['away_team'],
                'crew_id': row['crew_id'],
                'ref_events': ref_events,
                'spreads': {
                    'opening': opening_spread,
                    'closing': closing_spread,
                    'movements': []  # Could be populated from line history
                },
                'totals': {
                    'opening': opening_total,
                    'closing': closing_total
                },
                'outcomes': {
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_score': total_score,
                    'spread_result': spread_result,
                    'total_result': total_result
                },
                'prime_time': 'SNF' in game_id or 'MNF' in game_id or 'TNF' in game_id,
                'weather': 'dome'  # Simplified
            }
            
            backtest_data.append(game_data)
        
        conn.close()
        return backtest_data

# Usage example
if __name__ == "__main__":
    loader = NFLDataLoader()
    
    # Load historical data (replace with actual scraping)
    # loader.load_historical_data(2020, 2023)
    
    # Get data for backtesting
    historical_data = loader.get_historical_data_for_backtest('2023-01-01', '2023-12-31')
    
    # Save to JSON for use with backtest
    with open('historical_games.json', 'w') as f:
        json.dump(historical_data, f, indent=2)
    
    print(f"Loaded {len(historical_data)} games for backtesting")