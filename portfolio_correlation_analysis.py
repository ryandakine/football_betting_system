"""
Portfolio Correlation Analysis Module
Task Master Task #15: Analyze correlations between bets to manage portfolio risk.

This module implements correlation analysis for betting portfolios to:
- Calculate correlation matrices between potential bets
- Identify independent vs. correlated opportunities  
- Apply dimensionality reduction for correlation visualization
- Limit exposure to correlated positions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BettingOpportunity:
    """Represents a potential betting opportunity."""
    game_id: str
    team_home: str
    team_away: str
    market_type: str  # 'h2h', 'spread', 'total', 'props'
    bet_type: str     # 'home', 'away', 'over', 'under', etc.
    odds: float
    probability: float
    expected_value: float
    confidence: float
    stake_suggested: float
    game_time: datetime
    league: str       # 'NFL', 'NCAAF'
    features: Dict[str, float] = field(default_factory=dict)
    
    @property
    def correlation_key(self) -> str:
        """Generate key for correlation analysis."""
        return f"{self.league}_{self.market_type}_{self.bet_type}"

@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    correlation_matrix: np.ndarray
    correlation_df: pd.DataFrame
    high_correlations: List[Tuple[str, str, float]]
    independent_bets: List[str]
    correlated_groups: List[List[str]]
    risk_score: float
    recommended_positions: Dict[str, float]

class PortfolioCorrelationAnalyzer:
    """Advanced portfolio correlation analysis for betting opportunities."""
    
    def __init__(self, 
                 max_correlation_threshold: float = 0.7,
                 min_correlation_threshold: float = 0.3,
                 max_portfolio_risk: float = 0.02,
                 correlation_lookback_days: int = 30):
        """
        Initialize correlation analyzer.
        
        Args:
            max_correlation_threshold: Maximum allowed correlation between positions
            min_correlation_threshold: Minimum correlation to consider significant
            max_portfolio_risk: Maximum portfolio risk per position (2%)
            correlation_lookback_days: Days of historical data for correlation calculation
        """
        self.max_correlation_threshold = max_correlation_threshold
        self.min_correlation_threshold = min_correlation_threshold
        self.max_portfolio_risk = max_portfolio_risk
        self.correlation_lookback_days = correlation_lookback_days
        
        # Historical data storage
        self.historical_outcomes: Dict[str, List[float]] = {}
        self.historical_features: Dict[str, List[Dict[str, float]]] = {}
        
        # Correlation cache
        self.correlation_cache: Dict[str, CorrelationResult] = {}
        self.cache_timestamp: Optional[datetime] = None
        self.cache_ttl_minutes: int = 15
        
        logger.info(f"🔗 Portfolio Correlation Analyzer initialized")
        logger.info(f"   Max correlation threshold: {max_correlation_threshold}")
        logger.info(f"   Portfolio risk limit: {max_portfolio_risk*100}%")
    
    async def analyze_portfolio_correlations(self, 
                                           opportunities: List[BettingOpportunity],
                                           historical_data: Optional[pd.DataFrame] = None) -> CorrelationResult:
        """
        Analyze correlations between betting opportunities.
        
        Args:
            opportunities: List of potential betting opportunities
            historical_data: Historical outcome data for correlation calculation
            
        Returns:
            CorrelationResult with correlation analysis
        """
        logger.info(f"🔍 Analyzing correlations for {len(opportunities)} opportunities")
        
        # Check cache first
        cache_key = self._generate_cache_key(opportunities)
        if self._is_cache_valid(cache_key):
            logger.info("📋 Using cached correlation analysis")
            return self.correlation_cache[cache_key]
        
        # Build correlation matrix
        correlation_matrix, feature_df = await self._build_correlation_matrix(
            opportunities, historical_data
        )
        
        # Create correlation DataFrame with labels
        opportunity_labels = [f"{opp.game_id}_{opp.market_type}_{opp.bet_type}" 
                            for opp in opportunities]
        correlation_df = pd.DataFrame(
            correlation_matrix,
            index=opportunity_labels,
            columns=opportunity_labels
        )
        
        # Identify high correlations
        high_correlations = self._find_high_correlations(correlation_df)
        
        # Group correlated bets
        correlated_groups = self._group_correlated_bets(correlation_df)
        
        # Find independent bets
        independent_bets = self._find_independent_bets(correlation_df)
        
        # Calculate portfolio risk score
        risk_score = self._calculate_portfolio_risk_score(
            opportunities, correlation_matrix
        )
        
        # Generate position recommendations
        recommended_positions = await self._recommend_positions(
            opportunities, correlation_matrix, risk_score
        )
        
        # Create result
        result = CorrelationResult(
            correlation_matrix=correlation_matrix,
            correlation_df=correlation_df,
            high_correlations=high_correlations,
            independent_bets=independent_bets,
            correlated_groups=correlated_groups,
            risk_score=risk_score,
            recommended_positions=recommended_positions
        )
        
        # Cache result
        self.correlation_cache[cache_key] = result
        self.cache_timestamp = datetime.now()
        
        logger.info(f"✅ Correlation analysis complete:")
        logger.info(f"   High correlations found: {len(high_correlations)}")
        logger.info(f"   Independent bets: {len(independent_bets)}")
        logger.info(f"   Portfolio risk score: {risk_score:.3f}")
        
        return result
    
    async def _build_correlation_matrix(self, 
                                      opportunities: List[BettingOpportunity],
                                      historical_data: Optional[pd.DataFrame]) -> Tuple[np.ndarray, pd.DataFrame]:
        """Build correlation matrix from opportunities and historical data."""
        
        # Extract features for correlation analysis
        feature_vectors = []
        for opp in opportunities:
            # Combine multiple feature sources
            features = {
                # Game timing features
                'game_hour': opp.game_time.hour,
                'game_day_of_week': opp.game_time.weekday(),
                'days_until_game': (opp.game_time - datetime.now()).days,
                
                # Market features
                'odds_decimal': opp.odds,
                'implied_probability': 1.0 / opp.odds if opp.odds > 0 else 0.5,
                'expected_value': opp.expected_value,
                'confidence': opp.confidence,
                
                # League and market type encoding
                'is_nfl': 1.0 if opp.league == 'NFL' else 0.0,
                'is_h2h': 1.0 if opp.market_type == 'h2h' else 0.0,
                'is_spread': 1.0 if opp.market_type == 'spread' else 0.0,
                'is_total': 1.0 if opp.market_type == 'total' else 0.0,
                
                # Custom features from opportunity
                **opp.features
            }
            feature_vectors.append(features)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_vectors)
        
        # Handle missing values
        feature_df = feature_df.fillna(feature_df.mean())
        
        # Standardize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_df)
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(scaled_features)
        
        # Handle NaN values (can occur with constant features)
        correlation_matrix = np.nan_to_num(correlation_matrix, nan=0.0)
        
        return correlation_matrix, feature_df
    
    def _find_high_correlations(self, correlation_df: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """Find pairs of bets with high correlation."""
        high_correlations = []
        
        for i in range(len(correlation_df.columns)):
            for j in range(i + 1, len(correlation_df.columns)):
                corr_value = correlation_df.iloc[i, j]
                if abs(corr_value) >= self.min_correlation_threshold:
                    high_correlations.append((
                        correlation_df.index[i],
                        correlation_df.columns[j],
                        corr_value
                    ))
        
        # Sort by absolute correlation value
        high_correlations.sort(key=lambda x: abs(x[2]), reverse=True)
        return high_correlations
    
    def _group_correlated_bets(self, correlation_df: pd.DataFrame) -> List[List[str]]:
        """Group bets into correlated clusters."""
        # Use simple threshold-based clustering
        visited = set()
        groups = []
        
        for bet1 in correlation_df.index:
            if bet1 in visited:
                continue
                
            group = [bet1]
            visited.add(bet1)
            
            for bet2 in correlation_df.index:
                if bet2 != bet1 and bet2 not in visited:
                    corr_value = correlation_df.loc[bet1, bet2]
                    if abs(corr_value) >= self.max_correlation_threshold:
                        group.append(bet2)
                        visited.add(bet2)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _find_independent_bets(self, correlation_df: pd.DataFrame) -> List[str]:
        """Find bets that are relatively independent."""
        independent_bets = []
        
        for bet in correlation_df.index:
            max_correlation = 0.0
            for other_bet in correlation_df.index:
                if bet != other_bet:
                    corr_value = abs(correlation_df.loc[bet, other_bet])
                    max_correlation = max(max_correlation, corr_value)
            
            if max_correlation < self.min_correlation_threshold:
                independent_bets.append(bet)
        
        return independent_bets
    
    def _calculate_portfolio_risk_score(self, 
                                      opportunities: List[BettingOpportunity],
                                      correlation_matrix: np.ndarray) -> float:
        """Calculate overall portfolio risk score."""
        if len(opportunities) == 0:
            return 0.0
        
        # Weight by suggested stake amounts
        stakes = np.array([opp.stake_suggested for opp in opportunities])
        total_stake = np.sum(stakes)
        
        if total_stake == 0:
            return 0.0
        
        # Normalize stakes to weights
        weights = stakes / total_stake
        
        # Calculate portfolio variance using correlation matrix
        portfolio_variance = np.dot(weights, np.dot(correlation_matrix, weights))
        
        # Convert to risk score (0-1 scale)
        risk_score = min(1.0, portfolio_variance)
        
        return risk_score
    
    async def _recommend_positions(self, 
                                 opportunities: List[BettingOpportunity],
                                 correlation_matrix: np.ndarray,
                                 risk_score: float) -> Dict[str, float]:
        """Recommend position sizes considering correlations."""
        recommendations = {}
        
        # Apply correlation-based position adjustments
        for i, opp in enumerate(opportunities):
            base_position = opp.stake_suggested
            
            # Calculate correlation adjustment factor
            correlation_penalty = 0.0
            for j, other_opp in enumerate(opportunities):
                if i != j:
                    corr_value = abs(correlation_matrix[i, j])
                    if corr_value > self.min_correlation_threshold:
                        # Reduce position size for correlated bets
                        correlation_penalty += corr_value * other_opp.stake_suggested
            
            # Apply penalty (reduce position for highly correlated bets)
            adjustment_factor = max(0.1, 1.0 - (correlation_penalty * 0.5))
            adjusted_position = base_position * adjustment_factor
            
            # Ensure we don't exceed maximum portfolio risk
            max_allowed = self.max_portfolio_risk
            final_position = min(adjusted_position, max_allowed)
            
            bet_key = f"{opp.game_id}_{opp.market_type}_{opp.bet_type}"
            recommendations[bet_key] = final_position
        
        return recommendations
    
    def visualize_correlations(self, 
                             result: CorrelationResult,
                             save_path: Optional[str] = None) -> None:
        """Create correlation visualization."""
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        mask = np.triu(np.ones_like(result.correlation_matrix, dtype=bool))
        sns.heatmap(
            result.correlation_df,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            fmt='.2f',
            cbar_kws={'label': 'Correlation Coefficient'}
        )
        
        plt.title('Portfolio Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Betting Opportunities', fontsize=12)
        plt.ylabel('Betting Opportunities', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"📊 Correlation heatmap saved to {save_path}")
        
        plt.show()
    
    def apply_dimensionality_reduction(self, 
                                     result: CorrelationResult,
                                     n_components: int = 2) -> Dict[str, Any]:
        """Apply PCA for correlation visualization."""
        # Apply PCA to correlation matrix
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(result.correlation_matrix)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            reduced_data[:, 0], 
            reduced_data[:, 1],
            c=range(len(reduced_data)),
            cmap='viridis',
            alpha=0.7,
            s=100
        )
        
        # Add labels
        for i, label in enumerate(result.correlation_df.index):
            plt.annotate(
                label.split('_')[-1],  # Show only bet type
                (reduced_data[i, 0], reduced_data[i, 1]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8
            )
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title('Portfolio Correlation - PCA Visualization')
        plt.colorbar(scatter, label='Bet Index')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return {
            'pca_model': pca,
            'reduced_data': reduced_data,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'total_variance_explained': sum(pca.explained_variance_ratio_)
        }
    
    def _generate_cache_key(self, opportunities: List[BettingOpportunity]) -> str:
        """Generate cache key for opportunities."""
        # Create hash based on opportunity characteristics
        key_parts = []
        for opp in sorted(opportunities, key=lambda x: x.game_id):
            key_parts.append(f"{opp.game_id}_{opp.market_type}_{opp.bet_type}")
        return "_".join(key_parts)
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached result is still valid."""
        if cache_key not in self.correlation_cache:
            return False
        
        if self.cache_timestamp is None:
            return False
        
        age_minutes = (datetime.now() - self.cache_timestamp).total_seconds() / 60
        return age_minutes < self.cache_ttl_minutes
    
    def get_correlation_summary(self, result: CorrelationResult) -> Dict[str, Any]:
        """Get summary statistics for correlation analysis."""
        correlation_values = result.correlation_matrix[
            np.triu_indices_from(result.correlation_matrix, k=1)
        ]
        
        return {
            'total_opportunities': len(result.correlation_df),
            'high_correlations_count': len(result.high_correlations),
            'independent_bets_count': len(result.independent_bets),
            'correlated_groups_count': len(result.correlated_groups),
            'portfolio_risk_score': result.risk_score,
            'correlation_stats': {
                'mean': float(np.mean(correlation_values)),
                'std': float(np.std(correlation_values)),
                'min': float(np.min(correlation_values)),
                'max': float(np.max(correlation_values)),
                'median': float(np.median(correlation_values))
            },
            'risk_level': self._classify_risk_level(result.risk_score)
        }
    
    def _classify_risk_level(self, risk_score: float) -> str:
        """Classify portfolio risk level."""
        if risk_score < 0.3:
            return "LOW"
        elif risk_score < 0.6:
            return "MEDIUM"
        elif risk_score < 0.8:
            return "HIGH"
        else:
            return "VERY_HIGH"

# Example usage and testing
async def main():
    """Example usage of Portfolio Correlation Analyzer."""
    
    # Create sample betting opportunities
    opportunities = [
        BettingOpportunity(
            game_id="NFL_2024_W3_KC_CHI",
            team_home="Kansas City Chiefs",
            team_away="Chicago Bears", 
            market_type="h2h",
            bet_type="home",
            odds=1.85,
            probability=0.60,
            expected_value=0.11,
            confidence=0.75,
            stake_suggested=0.015,
            game_time=datetime.now() + timedelta(days=2),
            league="NFL",
            features={"home_advantage": 0.65, "weather_impact": 0.1}
        ),
        BettingOpportunity(
            game_id="NFL_2024_W3_KC_CHI",
            team_home="Kansas City Chiefs", 
            team_away="Chicago Bears",
            market_type="spread",
            bet_type="home",
            odds=1.91,
            probability=0.58,
            expected_value=0.08,
            confidence=0.70,
            stake_suggested=0.012,
            game_time=datetime.now() + timedelta(days=2),
            league="NFL",
            features={"spread_value": -7.5, "line_movement": 0.5}
        ),
        BettingOpportunity(
            game_id="NFL_2024_W3_DAL_NYG",
            team_home="New York Giants",
            team_away="Dallas Cowboys",
            market_type="h2h", 
            bet_type="away",
            odds=2.10,
            probability=0.52,
            expected_value=0.092,
            confidence=0.68,
            stake_suggested=0.014,
            game_time=datetime.now() + timedelta(days=3),
            league="NFL",
            features={"rivalry_factor": 0.8, "injury_impact": 0.2}
        )
    ]
    
    # Initialize analyzer
    analyzer = PortfolioCorrelationAnalyzer(
        max_correlation_threshold=0.7,
        max_portfolio_risk=0.02
    )
    
    # Analyze correlations
    result = await analyzer.analyze_portfolio_correlations(opportunities)
    
    # Print results
    summary = analyzer.get_correlation_summary(result)
    print("\n🔗 Portfolio Correlation Analysis Results:")
    print(f"   Total opportunities: {summary['total_opportunities']}")
    print(f"   High correlations: {summary['high_correlations_count']}")
    print(f"   Independent bets: {summary['independent_bets_count']}")
    print(f"   Portfolio risk: {summary['risk_level']} ({result.risk_score:.3f})")
    
    print(f"\n📊 Recommended positions:")
    for bet_key, position in result.recommended_positions.items():
        print(f"   {bet_key}: {position:.1%}")
    
    # Visualize correlations
    analyzer.visualize_correlations(result)

if __name__ == "__main__":
    asyncio.run(main())
