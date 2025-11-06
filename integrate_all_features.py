#!/usr/bin/env python3
"""
Complete Feature Integration for AI Council Training
Integrates ALL advanced features into historical data
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta
import sys
import os

# Import existing modules
sys.path.append(os.path.dirname(__file__))
from agent_influence_engine import AgentInfluenceEngine

class CompleteFeatureIntegrator:
    """Integrates all advanced features into training data"""
    
    def __init__(self):
        self.agent_engine = AgentInfluenceEngine()
        
    def calculate_epa_features(self, game: dict, team: str) -> dict:
        """
        Calculate EPA (Expected Points Added) features
        Based on advanced_feature_engineering.py
        """
        # Simplified EPA calculation for historical games
        # In production, would use play-by-play data
        
        score = game.get('home_score', 0) if team == game.get('home_team') else game.get('away_score', 0)
        opp_score = game.get('away_score', 0) if team == game.get('home_team') else game.get('home_score', 0)
        
        # EPA approximation from final score
        # Actual EPA requires play-by-play, this is simplified
        total_points = score + opp_score
        
        # EPA estimates (points per play)
        estimated_plays = 65  # Average NFL game
        epa_per_play = (score - 20) / estimated_plays  # Normalize around 20 points
        
        # Red zone efficiency (estimated from score)
        red_zone_epa = 0.3 if score > 24 else -0.1
        
        # Explosive plays (estimated)
        explosive_epa = 0.2 if score > 28 else 0.0
        
        return {
            'epa_per_play': round(epa_per_play, 3),
            'red_zone_epa': round(red_zone_epa, 3),
            'explosive_epa': round(explosive_epa, 3),
            'total_epa': round(score - 20, 1)  # Points above/below average
        }
    
    def calculate_dvoa(self, team_games: List[dict], team: str) -> dict:
        """
        Calculate DVOA (Defense-adjusted Value Over Average)
        Based on advanced_feature_engineering.py
        """
        if not team_games:
            return {
                'offensive_dvoa': 0.0,
                'defensive_dvoa': 0.0,
                'overall_dvoa': 0.0
            }
        
        # Calculate team's average points
        team_points = []
        opp_points = []
        
        for game in team_games:
            is_home = game.get('home_team') == team
            team_score = game.get('home_score') if is_home else game.get('away_score')
            opp_score = game.get('away_score') if is_home else game.get('home_score')
            
            if team_score is not None and opp_score is not None:
                team_points.append(team_score)
                opp_points.append(opp_score)
        
        if not team_points:
            return {
                'offensive_dvoa': 0.0,
                'defensive_dvoa': 0.0,
                'overall_dvoa': 0.0
            }
        
        # League average (approximate)
        league_avg = 21.5
        
        # DVOA calculations
        avg_points_for = np.mean(team_points)
        avg_points_against = np.mean(opp_points)
        
        offensive_dvoa = (avg_points_for - league_avg) / league_avg
        defensive_dvoa = (league_avg - avg_points_against) / league_avg  # Inverted
        overall_dvoa = (offensive_dvoa + defensive_dvoa) / 2
        
        return {
            'offensive_dvoa': round(offensive_dvoa, 3),
            'defensive_dvoa': round(defensive_dvoa, 3),
            'overall_dvoa': round(overall_dvoa, 3)
        }
    
    def calculate_recent_ats(self, team_games: List[dict], team: str, n_games: int = 5) -> dict:
        """Calculate recent ATS (Against The Spread) performance"""
        
        recent_games = sorted(team_games, key=lambda x: x.get('date', ''), reverse=True)[:n_games]
        
        if not recent_games:
            return {
                'ats_record_l5': '0-0',
                'ats_win_pct_l5': 0.5,
                'ats_wins': 0,
                'ats_losses': 0,
                'ats_pushes': 0
            }
        
        wins = 0
        losses = 0
        pushes = 0
        
        for game in recent_games:
            spread_result = game.get('spread_result')
            is_home = game.get('home_team') == team
            
            # Determine if team covered
            if spread_result == 'home_cover' and is_home:
                wins += 1
            elif spread_result == 'away_cover' and not is_home:
                wins += 1
            elif spread_result == 'push':
                pushes += 1
            else:
                losses += 1
        
        total_decided = wins + losses
        win_pct = wins / total_decided if total_decided > 0 else 0.5
        
        return {
            'ats_record_l5': f'{wins}-{losses}',
            'ats_win_pct_l5': round(win_pct, 3),
            'ats_wins': wins,
            'ats_losses': losses,
            'ats_pushes': pushes
        }
    
    def calculate_line_movement(self, game: dict) -> dict:
        """
        Track line movement (opening vs closing)
        Indicates sharp money movement
        """
        # In production, would track actual line movement
        # For historical data, estimate based on game outcome
        
        spread = game.get('spread')
        if spread is None:
            return {
                'line_movement': 0.0,
                'sharp_money_indicator': 0,
                'steam_move': 0
            }
        
        # Estimate line movement (would be actual historical data)
        # Sharp money typically moves lines 1-2 points
        estimated_movement = 0.0
        sharp_indicator = 0
        steam_move = 0
        
        # If actual spread_result suggests sharp money
        spread_result = game.get('spread_result')
        if spread_result:
            # Check for reverse line movement (RLM)
            # Simplified: if underdog covered, might indicate sharp money
            if abs(spread) > 3 and spread_result in ['away_cover', 'home_cover']:
                estimated_movement = 1.5
                sharp_indicator = 1
        
        return {
            'line_movement': round(estimated_movement, 1),
            'sharp_money_indicator': sharp_indicator,
            'steam_move': steam_move
        }
    
    def calculate_team_chemistry(self, team: str, season: int) -> dict:
        """
        Calculate team chemistry indicators
        Coordinator stability, roster continuity
        """
        # Simplified chemistry scores
        # In production, would use actual coaching/roster data
        
        # Estimate based on season continuity
        # Teams typically keep coordinators 2-3 years
        base_chemistry = 0.7
        
        # Championship teams (for example purposes)
        elite_teams = ['KC', 'SF', 'PHI', 'BUF', 'DAL']
        if team in elite_teams:
            base_chemistry = 0.85
        
        return {
            'coordinator_stability': round(base_chemistry, 2),
            'roster_continuity': round(base_chemistry + 0.05, 2),
            'qb_wr_chemistry': round(base_chemistry + 0.10, 2),
            'ol_cohesion': round(base_chemistry, 2),
            'overall_chemistry': round(base_chemistry, 2)
        }
    
    def get_agent_influence(self, game: dict) -> dict:
        """
        Get agent influence adjustments
        Uses agent_influence_engine.py
        """
        try:
            adjustments = self.agent_engine.compute_adjustments(game)
            return {
                'agent_edge_multiplier': adjustments.get('edge_multiplier', 1.0),
                'agent_confidence_delta': adjustments.get('confidence_delta', 0.0),
                'agent_rules_triggered': len(adjustments.get('rules_triggered', [])),
                'has_agent_conflict': 1 if adjustments.get('rules_triggered') else 0
            }
        except Exception as e:
            print(f"Agent influence error: {e}")
            return {
                'agent_edge_multiplier': 1.0,
                'agent_confidence_delta': 0.0,
                'agent_rules_triggered': 0,
                'has_agent_conflict': 0
            }
    
    def integrate_all_features(self, training_data_path: str) -> pd.DataFrame:
        """
        Main integration function
        Adds all advanced features to training data
        """
        print("🚀 COMPLETE FEATURE INTEGRATION")
        print("=" * 60)
        
        # Load training data
        with open(training_data_path, 'r') as f:
            data = json.load(f)
        
        games = data['games']
        print(f"📊 Processing {len(games)} games...")
        
        # Create DataFrame
        df = pd.DataFrame(games)
        
        # Group games by team for rolling calculations
        team_games = {}
        for game in games:
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            
            if home_team not in team_games:
                team_games[home_team] = []
            if away_team not in team_games:
                team_games[away_team] = []
            
            team_games[home_team].append(game)
            team_games[away_team].append(game)
        
        # Add features row by row
        enhanced_games = []
        
        for idx, game in enumerate(games):
            if idx % 100 == 0:
                print(f"   Processing game {idx}/{len(games)}...")
            
            enhanced = game.copy()
            
            home_team = game.get('home_team')
            away_team = game.get('away_team')
            season = game.get('season')
            
            # Get previous games for this team (for rolling stats)
            home_prev_games = [g for g in team_games.get(home_team, []) 
                              if g.get('date', '') < game.get('date', '')]
            away_prev_games = [g for g in team_games.get(away_team, []) 
                              if g.get('date', '') < game.get('date', '')]
            
            # 1. EPA Features
            home_epa = self.calculate_epa_features(game, home_team)
            away_epa = self.calculate_epa_features(game, away_team)
            enhanced.update({
                'home_epa_per_play': home_epa['epa_per_play'],
                'away_epa_per_play': away_epa['epa_per_play'],
                'epa_differential': home_epa['epa_per_play'] - away_epa['epa_per_play']
            })
            
            # 2. DVOA Features
            home_dvoa = self.calculate_dvoa(home_prev_games, home_team)
            away_dvoa = self.calculate_dvoa(away_prev_games, away_team)
            enhanced.update({
                'home_offensive_dvoa': home_dvoa['offensive_dvoa'],
                'home_defensive_dvoa': home_dvoa['defensive_dvoa'],
                'away_offensive_dvoa': away_dvoa['offensive_dvoa'],
                'away_defensive_dvoa': away_dvoa['defensive_dvoa'],
                'dvoa_differential': home_dvoa['overall_dvoa'] - away_dvoa['overall_dvoa']
            })
            
            # 3. Recent ATS Performance
            home_ats = self.calculate_recent_ats(home_prev_games, home_team)
            away_ats = self.calculate_recent_ats(away_prev_games, away_team)
            enhanced.update({
                'home_ats_l5': home_ats['ats_win_pct_l5'],
                'away_ats_l5': away_ats['ats_win_pct_l5'],
                'home_ats_wins': home_ats['ats_wins'],
                'away_ats_wins': away_ats['ats_wins']
            })
            
            # 4. Line Movement
            line_movement = self.calculate_line_movement(game)
            enhanced.update(line_movement)
            
            # 5. Team Chemistry
            home_chemistry = self.calculate_team_chemistry(home_team, season)
            away_chemistry = self.calculate_team_chemistry(away_team, season)
            enhanced.update({
                'home_chemistry': home_chemistry['overall_chemistry'],
                'away_chemistry': away_chemistry['overall_chemistry'],
                'chemistry_advantage': home_chemistry['overall_chemistry'] - away_chemistry['overall_chemistry']
            })
            
            # 6. Agent Influence
            agent_influence = self.get_agent_influence(game)
            enhanced.update(agent_influence)
            
            enhanced_games.append(enhanced)
        
        # Save enhanced data
        output_path = training_data_path.replace('.json', '_enhanced.json')
        with open(output_path, 'w') as f:
            json.dump({
                'collected_at': datetime.now().isoformat(),
                'total_games': len(enhanced_games),
                'features_added': [
                    'EPA', 'DVOA', 'ATS_Performance', 'Line_Movement', 
                    'Team_Chemistry', 'Agent_Influence'
                ],
                'games': enhanced_games
            }, f, indent=2)
        
        print(f"\n✅ Feature integration complete!")
        print(f"💾 Enhanced data saved to: {output_path}")
        
        # Print statistics
        df_enhanced = pd.DataFrame(enhanced_games)
        print(f"\n📊 FEATURE STATISTICS")
        print(f"=" * 60)
        print(f"   EPA range: {df_enhanced['epa_differential'].min():.3f} to {df_enhanced['epa_differential'].max():.3f}")
        print(f"   DVOA range: {df_enhanced['dvoa_differential'].min():.3f} to {df_enhanced['dvoa_differential'].max():.3f}")
        print(f"   ATS L5 avg: {df_enhanced['home_ats_l5'].mean():.3f}")
        print(f"   Agent conflicts: {df_enhanced['has_agent_conflict'].sum()} games")
        print(f"   Sharp money moves: {df_enhanced['sharp_money_indicator'].sum()} games")
        
        return df_enhanced

def main():
    """Run complete feature integration"""
    
    integrator = CompleteFeatureIntegrator()
    
    # Integrate features into training data
    training_data = 'data/nfl_training_data.json'
    
    if not os.path.exists(training_data):
        print(f"❌ Training data not found: {training_data}")
        print("   Run collect_historical_nfl.py first!")
        return
    
    df = integrator.integrate_all_features(training_data)
    
    print(f"\n🎯 Ready for AI Council training with ALL features!")

if __name__ == "__main__":
    main()
