import json
import numpy as np
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class RefEvent:
    ref_id: str
    crew_id: str
    game_id: str
    penalty_type: str
    team_affected: str  # 'home' or 'away'
    game_time: str
    spread_before: float
    spread_after: float
    prime_time: bool

class BayesianRefAnalyzer:
    def __init__(self):
        # Priors based on historical data
        self.priors = {
            'home_bias': 0.52,  # Historical home advantage in penalty calls
            'prime_time_variance': 0.15,  # Increased variance in prime time
            'crew_consistency': 0.75,  # How consistent crews are
            'spread_correlation': 0.08  # Normal spread movement correlation
        }
        
        self.evidence_weights = {
            'penalty_timing': 0.3,
            'spread_movement': 0.4,
            'historical_pattern': 0.2,
            'game_context': 0.1
        }
    
    def calculate_posterior(self, prior: float, likelihood: float, evidence_strength: float) -> float:
        """Bayesian update: P(H|E) = P(E|H) * P(H) / P(E)"""
        # Simplified Bayesian update
        posterior = (likelihood * prior) / ((likelihood * prior) + ((1 - likelihood) * (1 - prior)))
        return posterior * evidence_strength + prior * (1 - evidence_strength)
    
    def analyze_crew_patterns(self, events: List[RefEvent]) -> Dict:
        """Analyze referee crew for statistical anomalies"""
        
        # Group by crew
        crew_stats = {}
        
        for event in events:
            if event.crew_id not in crew_stats:
                crew_stats[event.crew_id] = {
                    'home_penalties': 0,
                    'away_penalties': 0,
                    'prime_time_games': 0,
                    'spread_movements': [],
                    'total_games': 0
                }
            
            stats_dict = crew_stats[event.crew_id]
            stats_dict['total_games'] += 1
            
            if event.team_affected == 'home':
                stats_dict['home_penalties'] += 1
            else:
                stats_dict['away_penalties'] += 1
                
            if event.prime_time:
                stats_dict['prime_time_games'] += 1
                
            spread_change = abs(event.spread_after - event.spread_before)
            stats_dict['spread_movements'].append(spread_change)
        
        # Calculate anomaly scores
        results = {}
        
        for crew_id, stats_dict in crew_stats.items():
            if stats_dict['total_games'] < 3:  # Need minimum sample size
                continue
                
            # Home bias calculation
            total_penalties = stats_dict['home_penalties'] + stats_dict['away_penalties']
            if total_penalties > 0:
                home_bias_rate = stats_dict['home_penalties'] / total_penalties
            else:
                home_bias_rate = 0.5
                
            # Statistical significance test
            from scipy.stats import binomtest
            p_value = binomtest(stats_dict['home_penalties'], total_penalties, 0.5).pvalue
            
            # Spread movement analysis
            avg_spread_movement = np.mean(stats_dict['spread_movements']) if stats_dict['spread_movements'] else 0
            spread_volatility = np.std(stats_dict['spread_movements']) if len(stats_dict['spread_movements']) > 1 else 0
            
            # Bayesian update
            likelihood = 1 - p_value  # Convert p-value to likelihood
            evidence_strength = min(0.8, total_penalties / 20)  # More evidence = stronger update
            
            posterior = self.calculate_posterior(
                self.priors['home_bias'], 
                likelihood, 
                evidence_strength
            )
            
            # Anomaly score (1-10 scale)
            bias_component = abs(home_bias_rate - 0.5) * 10
            spread_component = min(5, avg_spread_movement)
            volatility_component = min(3, spread_volatility)
            
            anomaly_score = bias_component + spread_component + volatility_component
            anomaly_score = min(10, max(1, anomaly_score))
            
            results[crew_id] = {
                'prior_home_bias': self.priors['home_bias'],
                'observed_home_bias': home_bias_rate,
                'posterior_bias': posterior,
                'p_value': p_value,
                'total_penalties': total_penalties,
                'games_analyzed': stats_dict['total_games'],
                'avg_spread_movement': round(avg_spread_movement, 2),
                'spread_volatility': round(spread_volatility, 2),
                'anomaly_score': round(anomaly_score, 1),
                'statistical_flags': self._generate_flags(home_bias_rate, p_value, avg_spread_movement)
            }
        
        return results
    
    def _generate_flags(self, home_bias: float, p_value: float, spread_movement: float) -> List[str]:
        """Generate statistical flags for unusual patterns"""
        flags = []
        
        if abs(home_bias - 0.5) > 0.15:
            flags.append(f"Extreme bias: {home_bias:.1%} home penalty rate")
            
        if p_value < 0.05:
            flags.append(f"Statistically significant bias (p={p_value:.3f})")
            
        if spread_movement > 2.0:
            flags.append(f"High spread volatility: {spread_movement:.1f} avg movement")
            
        return flags

# Example usage function
def analyze_referee_data(data_file: str) -> str:
    """Main analysis function"""
    analyzer = BayesianRefAnalyzer()
    
    # This would load your actual referee data
    # For now, creating sample data structure
    sample_events = [
        RefEvent("REF42", "CREW7", "GAME123", "holding", "away", "Q4-3:12", -7.0, -4.5, True),
        RefEvent("REF42", "CREW7", "GAME124", "PI", "home", "Q2-8:45", -3.0, -3.5, False),
        # Add more events...
    ]
    
    results = analyzer.analyze_crew_patterns(sample_events)
    
    return json.dumps(results, indent=2)

if __name__ == "__main__":
    print(analyze_referee_data("referee_data.json"))