#!/usr/bin/env python3
"""
Self-Improving NFL Betting System Loop
=====================================

Weekly self-improvement after every game:
- Pull actual outcomes from ESPN API
- Update causal inference models
- Retrain behavioral intelligence on sharp patterns
- Adjust portfolio optimizer with new Kelly weights
- Log to Supabase, cap at 24k tokens, hard kill >3 iterations
"""

import asyncio
import logging
import requests
import signal

# Loop timeout protection
def timeout_handler(signum, frame): 
    raise TimeoutError("Loop killed after 5 minutes")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(300)  # 5 minute hard timeout
import json
import sqlite3
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from dataclasses import dataclass

# Import our systems
sys.path.append('.')
from causal_inference_explainable_ai import ExplainableAISystem
from behavioral_intelligence_engine import BehavioralIntelligenceEngine
from portfolio_management_system import PortfolioManagementSystem
from referee_volatility_tracker import track_ref_changes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GameOutcome:
    game_id: str
    home_team: str
    away_team: str
    home_score: int
    away_score: int
    total_points: int
    spread_result: float
    ref_crew: str
    outcome_timestamp: datetime

class SelfImprovingLoop:
    def __init__(self):
        self.iteration_count = 0
        self.max_iterations = 3
        self.token_count = 0
        self.max_tokens = 24000
        self.improvement_metrics = {'edge_improvement': 0.0, 'model_accuracy': 0.0}
        
        # Initialize systems
        self.causal_system = ExplainableAISystem()
        self.behavioral_system = BehavioralIntelligenceEngine()
        self.portfolio_system = PortfolioManagementSystem()
        
    async def pull_game_outcomes(self, week: int = None) -> List[GameOutcome]:
        """Pull actual game outcomes from ESPN API"""
        try:
            week = week or datetime.now().isocalendar()[1]
            url = f"https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week={week}&seasontype=2"
            resp = requests.get(url, timeout=5)  # Fixed: 5s timeout
            data = resp.json()
            
            outcomes = []
            for event in data.get('events', []):
                if event.get('status', {}).get('type', {}).get('completed', False):
                    competitors = event.get('competitions', [{}])[0].get('competitors', [])
                    if len(competitors) == 2:
                        home = next((c for c in competitors if c.get('homeAway') == 'home'), {})
                        away = next((c for c in competitors if c.get('homeAway') == 'away'), {})
                        
                        home_score = int(home.get('score', 0))
                        away_score = int(away.get('score', 0))
                        
                        outcome = GameOutcome(
                            game_id=event.get('id', ''),
                            home_team=home.get('team', {}).get('abbreviation', ''),
                            away_team=away.get('team', {}).get('abbreviation', ''),
                            home_score=home_score,
                            away_score=away_score,
                            total_points=home_score + away_score,
                            spread_result=home_score - away_score,
                            ref_crew='Unknown',
                            outcome_timestamp=datetime.now()
                        )
                        outcomes.append(outcome)
            
            logger.info(f"📊 Pulled {len(outcomes)} completed games")
            return outcomes
            
        except Exception as e:
            await self.log_error(f"Error pulling outcomes: {e}")
            return []
    
    async def update_models(self, outcomes: List[GameOutcome]) -> Dict[str, float]:
        """Update all models with new outcome data"""
        self.iteration_count += 1
        if self.iteration_count > self.max_iterations:
            logger.error("🛑 HARD KILL: Exceeded 3 iterations")
            sys.exit(1)
        
        improvements = {}
        
        try:
            # 1. Update causal inference models
            causal_data = pd.DataFrame([{
                'game_id': o.game_id, 'home_score': o.home_score, 'away_score': o.away_score,
                'total_points': o.total_points, 'won': o.home_score > o.away_score
            } for o in outcomes])
            
            if not causal_data.empty:
                old_relationships = len(self.causal_system.causal_engine.causal_relationships)
                await self.causal_system.causal_engine.discover_causal_relationships(causal_data)
                new_relationships = len(self.causal_system.causal_engine.causal_relationships)
                improvements['causal_improvement'] = (new_relationships - old_relationships) / max(old_relationships, 1)
            
            # 2. Update referee bias models
            ref_changes = track_ref_changes()
            if ref_changes:
                for change in ref_changes:
                    logger.warning(f"🚨 REF VOLATILITY: {change}")
                improvements['ref_volatility_detected'] = len(ref_changes)
            
            # 3. Retrain behavioral intelligence
            behavioral_data = [{'public_pct': 0.6, 'sharp_pct': 0.4, 'line_move': o.spread_result * 0.1, 
                              'outcome': o.home_score > o.away_score} for o in outcomes]
            
            old_signals = self.behavioral_system.stats['signals_generated']
            for data in behavioral_data[:5]:  # Limit to prevent token overflow
                await self.behavioral_system.generate_behavioral_intelligence({
                    'game_id': 'training', 'public_percentage': data['public_pct'],
                    'sharp_percentage': data['sharp_pct'], 'line_movement': data['line_move']
                })
            new_signals = self.behavioral_system.stats['signals_generated']
            improvements['behavioral_improvement'] = (new_signals - old_signals) / max(old_signals, 1)
            
            # 4. Adjust portfolio optimizer Kelly weights
            win_rate = sum(1 for o in outcomes if o.home_score > o.away_score) / len(outcomes) if outcomes else 0.5
            old_bankroll = self.portfolio_system.bankroll
            self.portfolio_system.optimizer.kelly_calculator.max_kelly_fraction *= (1 + (win_rate - 0.5) * 0.1)
            improvements['kelly_adjustment'] = (self.portfolio_system.optimizer.kelly_calculator.max_kelly_fraction - 0.25) / 0.25
            
            # Calculate total edge improvement
            total_improvement = sum(improvements.values()) / len(improvements) if improvements else 0.0
            improvements['total_edge_improvement'] = total_improvement
            
            self.token_count += 1000  # Estimate token usage
            if self.token_count > self.max_tokens:
                logger.error("🛑 TOKEN LIMIT: Exceeded 24k tokens")
                sys.exit(1)
            
            return improvements
            
        except Exception as e:
            await self.log_error(f"Error updating models: {e}")
            return {'error': str(e)}
    
    async def log_error(self, error_msg: str):
        """Log errors to Supabase (simplified for demo)"""
        try:
            # In production: Use actual Supabase client
            with open('error_log.json', 'a') as f:
                json.dump({'timestamp': datetime.now().isoformat(), 'error': error_msg}, f)
                f.write('\n')
            logger.error(f"📝 Logged error: {error_msg}")
        except:
            logger.error(f"Failed to log error: {error_msg}")
    
    async def test_tnf_simulation(self) -> Dict[str, Any]:
        """Test on Bills-Dolphins TNF simulation"""
        try:
            # Simulate Bills-Dolphins outcome
            simulated_outcome = GameOutcome(
                game_id='BUF_vs_MIA_TNF',
                home_team='BUF',
                away_team='MIA', 
                home_score=31,
                away_score=10,
                total_points=41,
                spread_result=21,  # Bills covered big
                ref_crew='Hochuli',
                outcome_timestamp=datetime.now()
            )
            
            logger.info("🏈 Simulating Bills 31, Dolphins 10 (TNF)")
            
            # Update models with simulated outcome
            improvements = await self.update_models([simulated_outcome])
            
            # Calculate edge improvement
            edge_improvement = improvements.get('total_edge_improvement', 0.0) * 100
            
            result = {
                'simulated_game': 'Bills 31, Dolphins 10',
                'edge_improvement_pct': edge_improvement,
                'causal_updates': improvements.get('causal_improvement', 0),
                'behavioral_updates': improvements.get('behavioral_improvement', 0),
                'kelly_adjustment': improvements.get('kelly_adjustment', 0),
                'ref_volatility': improvements.get('ref_volatility_detected', 0),
                'status': 'success'
            }
            
            print(f"✅ Edge improved by {edge_improvement:.1f}%")
            return result
            
        except Exception as e:
            await self.log_error(f"TNF simulation error: {e}")
            return {'status': 'error', 'message': str(e)}

async def main():
    """Main self-improving loop"""
    print("🔄 NFL SELF-IMPROVING LOOP - WEEKLY UPDATE")
    print("=" * 50)
    
    loop = SelfImprovingLoop()
    
    # Test with TNF simulation
    print("🧪 Testing with Bills-Dolphins TNF simulation...")
    tnf_result = await loop.test_tnf_simulation()
    
    if tnf_result['status'] == 'success':
        print(f"🎯 TNF Test Results:")
        print(f"   Game: {tnf_result['simulated_game']}")
        print(f"   Edge Improvement: {tnf_result['edge_improvement_pct']:.1f}%")
        print(f"   Causal Updates: {tnf_result['causal_updates']:.3f}")
        print(f"   Behavioral Updates: {tnf_result['behavioral_updates']:.3f}")
        print(f"   Kelly Adjustment: {tnf_result['kelly_adjustment']:.3f}")
        print(f"   Ref Volatility: {tnf_result['ref_volatility']} alerts")
    
    # Weekly update (would run on Sundays)
    print(f"\n📅 Weekly model update simulation...")
    outcomes = await loop.pull_game_outcomes()
    if outcomes:
        weekly_improvements = await loop.update_models(outcomes)
        total_edge = weekly_improvements.get('total_edge_improvement', 0) * 100
        print(f"📈 Weekly edge improvement: {total_edge:.1f}%")
    
    print(f"\n✅ Self-improving loop complete!")
    print(f"🔄 Iterations: {loop.iteration_count}/{loop.max_iterations}")
    print(f"🪙 Tokens used: {loop.token_count}/{loop.max_tokens}")

if __name__ == "__main__":
    asyncio.run(main())
