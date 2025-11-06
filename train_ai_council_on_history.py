#!/usr/bin/env python3
"""
Train AI Council Models on Historical NFL Data
================================================

Analyzes 5 seasons of real NFL game data to extract patterns,
calibrate edge calculations, and improve prediction accuracy.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HistoricalAnalyzer:
    """Analyze historical NFL data to extract patterns."""
    
    def __init__(self):
        self.data_dir = Path("data/football/historical/nfl")
    
    def load_all_seasons(self) -> pd.DataFrame:
        """Load all historical game data."""
        all_games = []
        
        for season_file in sorted(self.data_dir.glob("nfl_*.csv")):
            season = int(season_file.stem.split("_")[1])
            df = pd.read_csv(season_file)
            df["season"] = season
            all_games.append(df)
            logger.info(f"Loaded season {season}: {len(df)} games")
        
        if not all_games:
            raise ValueError(f"No data found in {self.data_dir}")
        
        combined = pd.concat(all_games, ignore_index=True)
        logger.info(f"Total games: {len(combined)}")
        return combined
    
    def calculate_home_team_win_rate(self, df: pd.DataFrame) -> float:
        """Calculate historical home team win rate."""
        home_wins = (df["actual_result"] == 1).sum()
        total = len(df)
        rate = home_wins / total if total > 0 else 0.5
        logger.info(f"Home team win rate: {rate:.1%} ({home_wins}/{total})")
        return rate
    
    def analyze_by_team(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Analyze team-specific patterns."""
        team_stats = {}
        
        for team in pd.concat([df["home_team"], df["away_team"]]).unique():
            home_games = df[df["home_team"] == team]
            away_games = df[df["away_team"] == team]
            
            home_wins = (home_games["actual_result"] == 1).sum()
            away_wins = (away_games["actual_result"] == 0).sum()
            total_wins = home_wins + away_wins
            
            total_games = len(home_games) + len(away_games)
            if total_games == 0:
                continue
            
            win_rate = total_wins / total_games
            home_win_pct = home_wins / len(home_games) if len(home_games) > 0 else 0.5
            away_win_pct = away_wins / len(away_games) if len(away_games) > 0 else 0.5
            
            team_stats[team] = {
                "total_games": total_games,
                "total_wins": total_wins,
                "win_rate": win_rate,
                "home_games": len(home_games),
                "home_wins": home_wins,
                "home_win_pct": home_win_pct,
                "away_games": len(away_games),
                "away_wins": away_wins,
                "away_win_pct": away_win_pct,
            }
        
        # Sort by win rate
        sorted_teams = sorted(team_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        
        logger.info("\n🏆 Top 10 Teams by Win Rate:")
        for i, (team, stats) in enumerate(sorted_teams[:10], 1):
            logger.info(f"   {i:2}. {team:3} | W/L: {stats['total_wins']:2}/{stats['total_games']-stats['total_wins']:2} | "
                       f"WR: {stats['win_rate']:.1%} | Home: {stats['home_win_pct']:.1%} | Away: {stats['away_win_pct']:.1%}")
        
        return team_stats
    
    def analyze_spread_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze spread prediction patterns."""
        df = df.copy()
        df["spread"] = pd.to_numeric(df["spread"], errors="coerce").fillna(0.0)
        df["actual_result"] = pd.to_numeric(df["actual_result"], errors="coerce").fillna(0)
        
        # Spread accuracy by line value
        spread_bins = [-5, -3, -2, 0, 2, 3, 5, 10]
        df["spread_bin"] = pd.cut(df["spread"], bins=spread_bins)
        
        spread_accuracy = {}
        for bin_label in df["spread_bin"].unique():
            if pd.isna(bin_label):
                continue
            subset = df[df["spread_bin"] == bin_label]
            if len(subset) > 0:
                wins = (subset["actual_result"] == 1).sum()
                accuracy = wins / len(subset)
                spread_accuracy[str(bin_label)] = {
                    "games": len(subset),
                    "accuracy": accuracy,
                }
        
        logger.info("\n📊 Spread Accuracy by Line:")
        for spread_range, stats in sorted(spread_accuracy.items()):
            logger.info(f"   {spread_range:20} | Games: {stats['games']:3} | Accuracy: {stats['accuracy']:.1%}")
        
        return spread_accuracy
    
    def generate_recommendations(self, home_wr: float, team_stats: Dict, spread_acc: Dict) -> List[str]:
        """Generate recommendations for model training."""
        recommendations = []
        
        if home_wr > 0.53:
            recommendations.append("Strong home field advantage detected - include home team bias in models")
        
        if home_wr < 0.47:
            recommendations.append("Surprising away team advantage - investigate data quality")
        
        # Find high-variance teams
        win_rates = [s["win_rate"] for s in team_stats.values()]
        std_wr = np.std(win_rates)
        mean_wr = np.mean(win_rates)
        
        if std_wr > 0.10:
            recommendations.append("High team variance detected - weight team strength heavily in models")
        
        recommendations.append("Use historical team stats as features in edge calculation")
        recommendations.append("Calibrate confidence thresholds based on actual prediction accuracy")
        
        return recommendations
    
    def run_analysis(self) -> None:
        """Run full historical analysis."""
        logger.info("=" * 70)
        logger.info("NFL Historical Data Analysis")
        logger.info("=" * 70)
        
        df = self.load_all_seasons()
        home_wr = self.calculate_home_team_win_rate(df)
        team_stats = self.analyze_by_team(df)
        spread_acc = self.analyze_spread_patterns(df)
        recs = self.generate_recommendations(home_wr, team_stats, spread_acc)
        
        logger.info("\n💡 Recommendations for Model Training:")
        for i, rec in enumerate(recs, 1):
            logger.info(f"   {i}. {rec}")
        
        # Save analysis
        analysis_file = Path("historical_analysis.json")
        analysis = {
            "total_games": len(df),
            "home_win_rate": float(home_wr),
            "top_teams": dict(sorted(
                [(k, v) for k, v in team_stats.items() if k],
                key=lambda x: x[1]["win_rate"],
                reverse=True
            )[:10]),
            "recommendations": recs,
        }
        
        with open(analysis_file, "w") as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"\n✓ Analysis saved to {analysis_file}")


if __name__ == "__main__":
    analyzer = HistoricalAnalyzer()
    analyzer.run_analysis()
