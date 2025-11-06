"""
Real Data Loader - NO simulated data fallbacks
Fails hard and loud if real data not found
"""
import pandas as pd
import os
import sys

class RealDataLoadError(Exception):
    pass

class RealDataLoader:
    """Load ONLY real data. No exceptions, no fallbacks."""
    
    DATA_DIR = './data/referee_conspiracy/'
    REQUIRED_FILES = {
        'crew_game_log': 'crew_game_log.parquet',
        'schedules': 'schedules_2018_2024.parquet',
        'penalties': 'penalties_2018_2024.parquet',
        'team_penalty_log': 'team_penalty_log.parquet',
        'crew_features': 'crew_features.parquet'
    }
    
    @staticmethod
    def validate_files_exist():
        """Check all real data files exist - fail if not"""
        missing = []
        for name, filename in RealDataLoader.REQUIRED_FILES.items():
            path = os.path.join(RealDataLoader.DATA_DIR, filename)
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            raise RealDataLoadError(
                f"❌ CRITICAL: Real data files missing:\n" + 
                "\n".join(missing) +
                "\nCannot proceed without real data. NO FALLBACKS."
            )
    
    @staticmethod
    def load_crew_games():
        """Load crew game log - real NFL data 2018-2025"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['crew_game_log'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded crew_game_log: {len(df)} real games")
            return df
        except Exception as e:
            raise RealDataLoadError(f"Failed to load crew games: {e}")
    
    @staticmethod
    def load_schedules():
        """Load real game schedules"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['schedules'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded schedules: {len(df)} real games")
            return df
        except Exception as e:
            raise RealDataLoadError(f"Failed to load schedules: {e}")
    
    @staticmethod
    def load_penalties():
        """Load real penalty data"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['penalties'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded penalties: {len(df)} real penalties")
            return df
        except Exception as e:
            raise RealDataLoadError(f"Failed to load penalties: {e}")
    
    @staticmethod
    def load_team_penalties():
        """Load real team penalty log"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['team_penalty_log'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded team_penalty_log: {len(df)} real penalty records")
            return df
        except Exception as e:
            raise RealDataLoadError(f"Failed to load team penalties: {e}")
    
    @staticmethod
    def load_crew_features():
        """Load real crew feature data"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['crew_features'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded crew_features: {len(df)} real crew records")
            return df
        except Exception as e:
            raise RealDataLoadError(f"Failed to load crew features: {e}")
    
    @staticmethod
    def load_games_with_props():
        """Load crew game log which has real game data for prop backtesting"""
        path = os.path.join(RealDataLoader.DATA_DIR, RealDataLoader.REQUIRED_FILES['crew_game_log'])
        try:
            df = pd.read_parquet(path)
            print(f"✅ Loaded {len(df)} real crew games for prop backtesting")
            return df.to_dict('records')
        except Exception as e:
            raise RealDataLoadError(f"Failed to load games for props: {e}")
    
    @staticmethod
    def load_all():
        """Load ALL real data - fail if anything is missing"""
        print("Loading REAL data (no simulated fallbacks)...")
        print("=" * 70)
        
        RealDataLoader.validate_files_exist()
        
        data = {
            'crew_games': RealDataLoader.load_crew_games(),
            'schedules': RealDataLoader.load_schedules(),
            'penalties': RealDataLoader.load_penalties(),
            'team_penalties': RealDataLoader.load_team_penalties(),
            'crew_features': RealDataLoader.load_crew_features()
        }
        
        print("=" * 70)
        print("✅ ALL REAL DATA LOADED SUCCESSFULLY")
        return data


if __name__ == "__main__":
    try:
        data = RealDataLoader.load_all()
        print("\nData Summary:")
        print(f"  Crew Games: {len(data['crew_games'])} rows")
        print(f"  Schedules: {len(data['schedules'])} rows")
        print(f"  Penalties: {len(data['penalties'])} rows")
        print(f"  Team Penalties: {len(data['team_penalties'])} rows")
        print(f"  Crew Features: {len(data['crew_features'])} rows")
    except RealDataLoadError as e:
        print(f"\n❌ {e}")
        sys.exit(1)
