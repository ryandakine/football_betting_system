#!/usr/bin/env python3
"""
Generate sample NFL data for backtesting the referee analysis system.
This creates realistic sample data to test the backtesting framework.
"""

import random
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict

# NFL teams
NFL_TEAMS = [
    'KC', 'BUF', 'CIN', 'BAL', 'TEN', 'IND', 'HOU', 'JAX',
    'LV', 'LAC', 'DEN', 'CLE', 'PIT', 'MIA', 'NYJ', 'NE',
    'DAL', 'PHI', 'NYG', 'WAS', 'GB', 'MIN', 'CHI', 'DET',
    'TB', 'NO', 'ATL', 'CAR', 'LAR', 'SF', 'SEA', 'AZ'
]

# Referee crews
REFEREE_CREWS = [
    'CREW_1', 'CREW_2', 'CREW_3', 'CREW_4', 'CREW_5', 
    'CREW_6', 'CREW_7', 'CREW_8', 'CREW_9', 'CREW_10'
]

# Penalty types
PENALTY_TYPES = [
    'Holding', 'Pass Interference', 'False Start', 'Roughing the Passer',
    'Unnecessary Roughness', 'Offensive PI', 'Illegal Formation', 
    'Delay of Game', 'Facemask', 'Unsportsmanlike Conduct'
]

def generate_sample_games(num_games: int = 100, season: int = 2023) -> List[Dict]:
    """Generate sample NFL games with realistic data"""
    games = []
    
    # Start date for season
    start_date = datetime(season, 9, 7)  # NFL season typically starts early September
    
    for i in range(num_games):
        # Pick random teams
        home_team = random.choice(NFL_TEAMS)
        away_team = random.choice([t for t in NFL_TEAMS if t != home_team])
        
        # Generate scores (realistic NFL scores)
        home_score = random.randint(3, 45)
        away_score = random.randint(3, 45)
        
        # Game date (spread over season)
        game_date = start_date + timedelta(days=random.randint(0, 120))
        
        game_id = f'NFL_{season}_G{i+1}_{away_team}_{home_team}'
        
        games.append({
            'game_id': game_id,
            'date': game_date.strftime('%Y-%m-%d'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': home_score,
            'away_score': away_score,
            'week': (i // 16) + 1,  # Roughly 16 games per week
            'season': season,
            'game_type': 'REG',
            'weather': random.choice(['Clear', 'Rain', 'Snow', 'Dome']),
            'surface': random.choice(['Grass', 'Turf']),
            'roof': random.choice(['Open', 'Dome', 'Retractable']),
            'temp': random.randint(32, 85),
            'wind': random.randint(0, 25)
        })
    
    return games

def generate_referee_assignments(games: List[Dict]) -> List[Dict]:
    """Generate referee crew assignments for games"""
    assignments = []
    
    referee_names = [
        'John Smith', 'Mike Johnson', 'Bob Wilson', 'Steve Davis',
        'Tom Brown', 'Jim Miller', 'Dave Taylor', 'Mark Anderson',
        'Chris Lee', 'Pat Williams'
    ]
    
    for game in games:
        crew_id = random.choice(REFEREE_CREWS)
        referee = random.choice(referee_names)
        
        assignments.append({
            'game_id': game['game_id'],
            'crew_id': crew_id,
            'referee': referee,
            'umpire': f'Umpire_{crew_id}',
            'down_judge': f'DJ_{crew_id}',
            'line_judge': f'LJ_{crew_id}',
            'field_judge': f'FJ_{crew_id}',
            'side_judge': f'SJ_{crew_id}',
            'back_judge': f'BJ_{crew_id}',
            'replay_official': f'RO_{crew_id}'
        })
    
    return assignments

def generate_penalties(games: List[Dict]) -> List[Dict]:
    """Generate penalty data for games"""
    penalties = []
    
    for game in games:
        # Random number of penalties per game (2-12 is realistic)
        num_penalties = random.randint(2, 12)
        
        for _ in range(num_penalties):
            penalty_team = random.choice([game['home_team'], game['away_team']])
            penalty_type = random.choice(PENALTY_TYPES)
            quarter = random.randint(1, 4)
            
            # Generate time remaining (MM:SS format)
            minutes = random.randint(0, 14)
            seconds = random.randint(0, 59)
            time_remaining = f'{minutes:02d}:{seconds:02d}'
            
            penalties.append({
                'game_id': game['game_id'],
                'penalty_type': penalty_type,
                'team': penalty_team,
                'quarter': quarter,
                'time_remaining': time_remaining,
                'yards': random.choice([5, 10, 15]),
                'result': random.choice(['Accepted', 'Declined']),
                'ref_position': random.choice(['Referee', 'Umpire', 'Back Judge', 'Field Judge'])
            })
    
    return penalties

def generate_betting_lines(games: List[Dict]) -> List[Dict]:
    """Generate betting lines with movements"""
    lines = []
    
    for game in games:
        # Generate realistic spread and total
        opening_spread = round(random.uniform(-14, 14), 1)
        opening_total = round(random.uniform(38, 58), 1)
        
        # Generate line movements throughout the day
        timestamps = ['09:00:00', '12:00:00', '15:00:00', '18:00:00']
        
        current_spread = opening_spread
        current_total = opening_total
        
        for timestamp in timestamps:
            # Small movements
            spread_movement = random.uniform(-1.5, 1.5)
            total_movement = random.uniform(-2, 2)
            
            current_spread += spread_movement
            current_total += total_movement
            
            lines.append({
                'game_id': game['game_id'],
                'timestamp': f"{game['date']} {timestamp}",
                'spread': round(current_spread, 1),
                'total': round(current_total, 1),
                'home_ml': random.randint(-200, 200),
                'away_ml': random.randint(-200, 200),
                'source': random.choice(['DraftKings', 'FanDuel', 'BetMGM'])
            })
    
    return lines

def populate_database(db_path: str = 'nfl_historical.db'):
    """Populate database with sample data"""
    
    print("🔧 Generating sample NFL data...")
    
    # Generate data
    games = generate_sample_games(200, 2023)  # 200 sample games
    referee_assignments = generate_referee_assignments(games)
    penalties = generate_penalties(games)
    betting_lines = generate_betting_lines(games)
    
    print(f"Generated {len(games)} games")
    print(f"Generated {len(penalties)} penalties")
    print(f"Generated {len(betting_lines)} betting lines")
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables (same as in historical_data_loader.py)
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
    
    # Insert games
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
    
    # Insert referee assignments
    for crew in referee_assignments:
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
    
    # Insert penalties
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
    
    # Insert betting lines
    for line in betting_lines:
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
    
    print(f"✅ Sample data populated in {db_path}")

if __name__ == "__main__":
    populate_database()
    print("\n🏈 Database ready for backtesting!")