#!/usr/bin/env python3
"""
Extract Referee Training Data from Autopsy Reports
==================================================

Parses referee conspiracy reports and creates structured training data
with referee impact features for AI Council model training.
"""

import os
import re
import json
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd

class RefereeDataExtractor:
    """Extract and structure referee data for ML training"""
    
    def __init__(self, reports_dir: str = "reports/referee_conspiracy"):
        self.reports_dir = Path(reports_dir)
        self.referee_profiles = {}
        self.team_referee_history = {}
        
    def parse_all_reports(self) -> Dict[str, Any]:
        """Parse all team referee reports"""
        
        print("🔍 Parsing referee autopsy reports...")
        
        for report_file in self.reports_dir.glob("*.md"):
            if "(copy)" in report_file.name:
                continue  # Skip duplicates
                
            team_code = report_file.stem
            print(f"   📄 Processing {team_code}...")
            
            self.team_referee_history[team_code] = self._parse_team_report(report_file)
        
        # Build referee profiles aggregated across all teams
        self._build_referee_profiles()
        
        print(f"\n✅ Parsed {len(self.team_referee_history)} team reports")
        print(f"✅ Built profiles for {len(self.referee_profiles)} referees")
        
        return {
            'team_history': self.team_referee_history,
            'referee_profiles': self.referee_profiles
        }
    
    def _parse_team_report(self, filepath: Path) -> Dict[str, Any]:
        """Parse a single team's referee report"""
        
        with open(filepath, 'r') as f:
            content = f.read()
        
        team_data = {
            'crew_rotation': [],
            'style_impact': {},
            'narrative_correlations': [],
            'broadcast_behavior': {}
        }
        
        # Parse crew rotation timeline
        rotation_section = re.search(r'## Crew Rotation Timeline\n(.*?)\n\n##', content, re.DOTALL)
        if rotation_section:
            for line in rotation_section.group(1).strip().split('\n'):
                match = re.match(r'- (\d{4}): ([^(]+) \((\d+) games; weeks ([^;]+); avg margin ([^;]+); labels: ([^)]+)\)', line)
                if match:
                    season, ref_name, games, weeks, margin, labels = match.groups()
                    team_data['crew_rotation'].append({
                        'season': int(season),
                        'referee': ref_name.strip(),
                        'games': int(games),
                        'weeks': weeks.strip(),
                        'avg_margin': float(margin),
                        'labels': [l.strip() for l in labels.split(',')]
                    })
        
        # Parse style impact
        style_section = re.search(r'## Style Impact\n(.*?)\n\n##', content, re.DOTALL)
        if style_section:
            for line in style_section.group(1).strip().split('\n'):
                match = re.match(
                    r'- ([^(]+) \(([^)]+)\): (\d+) games, avg margin ([^,]+), '
                    r'penalties on team ([^,]+), penalty diff ([^,]+), '
                    r'odds delta ([^,]+), overtime rate ([^%]+)',
                    line
                )
                if match:
                    ref_name, labels, games, margin, pen_on, pen_diff, odds_delta, ot_rate = match.groups()
                    team_data['style_impact'][ref_name.strip()] = {
                        'labels': [l.strip() for l in labels.split(',')],
                        'games': int(games),
                        'avg_margin': float(margin),
                        'penalties_on_team': float(pen_on),
                        'penalty_diff': float(pen_diff),
                        'odds_delta': float(odds_delta),
                        'overtime_rate': float(ot_rate.replace('%', ''))
                    }
        
        # Parse narrative correlations
        narrative_section = re.search(r'## Narrative Correlations\n(.*?)\n\n##', content, re.DOTALL)
        if narrative_section:
            for line in narrative_section.group(1).strip().split('\n'):
                team_data['narrative_correlations'].append(line.strip('- '))
        
        # Parse broadcast behavior
        broadcast_section = re.search(r'## Broadcast Behavior\n(.*?)\n\n', content, re.DOTALL)
        if broadcast_section:
            for line in broadcast_section.group(1).strip().split('\n'):
                match = re.match(r'- ([^:]+): ([^ ]+) penalties on team, ([^ ]+) total points', line)
                if match:
                    slot, penalties, total_points = match.groups()
                    team_data['broadcast_behavior'][slot.strip()] = {
                        'penalties_on_team': float(penalties),
                        'total_points': float(total_points)
                    }
        
        return team_data
    
    def _build_referee_profiles(self):
        """Aggregate referee profiles across all teams"""
        
        for team_code, team_data in self.team_referee_history.items():
            for ref_name, style_data in team_data['style_impact'].items():
                if ref_name not in self.referee_profiles:
                    self.referee_profiles[ref_name] = {
                        'total_games': 0,
                        'avg_margin': [],
                        'avg_penalties': [],
                        'avg_penalty_diff': [],
                        'avg_odds_delta': [],
                        'avg_overtime_rate': [],
                        'labels': set(),
                        'teams_worked': []
                    }
                
                profile = self.referee_profiles[ref_name]
                profile['total_games'] += style_data['games']
                profile['avg_margin'].append(style_data['avg_margin'])
                profile['avg_penalties'].append(style_data['penalties_on_team'])
                profile['avg_penalty_diff'].append(style_data['penalty_diff'])
                profile['avg_odds_delta'].append(style_data['odds_delta'])
                profile['avg_overtime_rate'].append(style_data['overtime_rate'])
                profile['labels'].update(style_data['labels'])
                profile['teams_worked'].append(team_code)
        
        # Calculate averages
        for ref_name, profile in self.referee_profiles.items():
            profile['avg_margin'] = sum(profile['avg_margin']) / len(profile['avg_margin'])
            profile['avg_penalties'] = sum(profile['avg_penalties']) / len(profile['avg_penalties'])
            profile['avg_penalty_diff'] = sum(profile['avg_penalty_diff']) / len(profile['avg_penalty_diff'])
            profile['avg_odds_delta'] = sum(profile['avg_odds_delta']) / len(profile['avg_odds_delta'])
            profile['avg_overtime_rate'] = sum(profile['avg_overtime_rate']) / len(profile['avg_overtime_rate'])
            profile['labels'] = list(profile['labels'])
    
    def create_game_referee_features(self, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create referee features for a single game
        
        Args:
            game_data: Dict with keys 'home_team', 'away_team', 'referee', 'season', 'week'
        
        Returns:
            Dict of referee-based features
        """
        
        referee = game_data.get('referee', 'Unknown')
        home_team = game_data.get('home_team')
        away_team = game_data.get('away_team')
        
        features = {
            # Referee global profile
            'ref_avg_margin': 0.0,
            'ref_avg_penalties': 6.0,
            'ref_penalty_diff': 0.0,
            'ref_odds_delta': 0.0,
            'ref_overtime_rate': 6.0,
            
            # Referee style labels (binary flags)
            'ref_is_baseline_control': 0,
            'ref_is_high_penalties': 0,
            'ref_is_low_flags': 0,
            'ref_is_overtime_frequent': 0,
            'ref_is_overseas_surge': 0,
            
            # Team-specific history with this referee
            'home_ref_history_margin': 0.0,
            'home_ref_history_penalties': 6.0,
            'home_ref_history_games': 0,
            'away_ref_history_margin': 0.0,
            'away_ref_history_penalties': 6.0,
            'away_ref_history_games': 0,
            
            # Referee advantage (positive = home favored)
            'ref_home_advantage': 0.0,
            'ref_penalty_advantage': 0.0
        }
        
        # Fill in referee profile if available
        if referee in self.referee_profiles:
            profile = self.referee_profiles[referee]
            features['ref_avg_margin'] = profile['avg_margin']
            features['ref_avg_penalties'] = profile['avg_penalties']
            features['ref_penalty_diff'] = profile['avg_penalty_diff']
            features['ref_odds_delta'] = profile['avg_odds_delta']
            features['ref_overtime_rate'] = profile['avg_overtime_rate']
            
            # Style labels
            features['ref_is_baseline_control'] = int('baseline_control' in profile['labels'])
            features['ref_is_high_penalties'] = int('high_penalties_close_games' in profile['labels'])
            features['ref_is_low_flags'] = int('low_flags_high_blowouts' in profile['labels'])
            features['ref_is_overtime_frequent'] = int('overtime_frequency_gt_15pct' in profile['labels'])
            features['ref_is_overseas_surge'] = int('overseas_flag_surge' in profile['labels'])
        
        # Team-specific history
        if home_team in self.team_referee_history:
            home_history = self.team_referee_history[home_team]['style_impact'].get(referee, {})
            if home_history:
                features['home_ref_history_margin'] = home_history['avg_margin']
                features['home_ref_history_penalties'] = home_history['penalties_on_team']
                features['home_ref_history_games'] = home_history['games']
        
        if away_team in self.team_referee_history:
            away_history = self.team_referee_history[away_team]['style_impact'].get(referee, {})
            if away_history:
                features['away_ref_history_margin'] = away_history['avg_margin']
                features['away_ref_history_penalties'] = away_history['penalties_on_team']
                features['away_ref_history_games'] = away_history['games']
        
        # Calculate advantages
        features['ref_home_advantage'] = features['home_ref_history_margin'] - features['away_ref_history_margin']
        features['ref_penalty_advantage'] = features['away_ref_history_penalties'] - features['home_ref_history_penalties']
        
        return features
    
    def export_training_data(self, output_path: str = "data/referee_training_features.json"):
        """Export referee data for training"""
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        export_data = {
            'referee_profiles': self.referee_profiles,
            'team_history': {
                team: {
                    'style_impact': data['style_impact'],
                    'broadcast_behavior': data['broadcast_behavior']
                }
                for team, data in self.team_referee_history.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"\n💾 Saved referee training data to {output_path}")
        
        # Also create a CSV for easy inspection
        csv_path = output_path.replace('.json', '.csv')
        df = pd.DataFrame([
            {
                'referee': ref_name,
                'total_games': profile['total_games'],
                'avg_margin': profile['avg_margin'],
                'avg_penalties': profile['avg_penalties'],
                'penalty_diff': profile['avg_penalty_diff'],
                'odds_delta': profile['avg_odds_delta'],
                'overtime_rate': profile['avg_overtime_rate'],
                'labels': ', '.join(profile['labels'])
            }
            for ref_name, profile in self.referee_profiles.items()
        ])
        df.to_csv(csv_path, index=False)
        print(f"💾 Saved referee profiles CSV to {csv_path}")

def main():
    """Main execution"""
    
    print("🏈 NFL REFEREE TRAINING DATA EXTRACTION")
    print("=" * 60)
    
    extractor = RefereeDataExtractor()
    extractor.parse_all_reports()
    extractor.export_training_data()
    
    print("\n✅ Referee data ready for AI Council training!")
    print("   Use this data to enrich game predictions with referee bias features")

if __name__ == "__main__":
    main()
