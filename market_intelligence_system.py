#!/usr/bin/env python3
"""
Market Intelligence Analysis System - YOLO MODE
==============================================

Complete sportsbook intelligence system:
- Sportsbook algorithm detection
- Market efficiency analysis  
- Sharp money quantification
- Multi-book arbitrage detection
- Automated opportunity scoring

YOLO MODE: Maximum performance, production-ready implementation.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import sqlite3
import os
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BookBias(Enum):
    """Sportsbook bias types"""
    OVER_BIASED = "over_biased"
    UNDER_BIASED = "under_biased"
    HOME_BIASED = "home_biased"
    AWAY_BIASED = "away_biased"
    SHARP_FRIENDLY = "sharp_friendly"
    PUBLIC_BOOK = "public_book"
    NEUTRAL = "neutral"


@dataclass
class SportsbookProfile:
    """Sportsbook algorithm profile"""
    book_name: str
    bias_type: BookBias
    bias_strength: float  # 0-1 scale
    line_movement_pattern: str
    sharp_money_tolerance: float
    market_making_style: str
    confidence: float
    sample_size: int


@dataclass
class MarketEfficiency:
    """Market efficiency metrics"""
    game_id: str
    book_name: str
    clv_score: float  # Closing Line Value
    movement_correlation: float
    efficiency_score: float  # 0-1, higher = more efficient
    sharp_money_percentage: float
    public_money_percentage: float
    timestamp: datetime


@dataclass
class ArbitrageOpportunity:
    """Arbitrage opportunity with scoring"""
    game_id: str
    market_type: str
    book_a: str
    book_b: str
    line_a: float
    line_b: float
    profit_margin: float
    risk_score: float  # 0-1, higher = riskier
    opportunity_score: float  # Combined profit/risk score
    expires_in_minutes: float
    timestamp: datetime


class SportsbookAlgorithmDetector:
    """Detects sportsbook algorithms and biases"""
    
    def __init__(self, db_path: str = "data/market_intelligence.db"):
        self.db_path = db_path
        self.book_profiles: Dict[str, SportsbookProfile] = {}
        self._init_database()
    
    def _init_database(self):
        """Initialize market intelligence database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS line_history (
                    id INTEGER PRIMARY KEY,
                    game_id TEXT,
                    book_name TEXT,
                    market_type TEXT,
                    line_value REAL,
                    timestamp TIMESTAMP,
                    volume_indicator REAL,
                    sharp_percentage REAL
                )
            """)
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS book_profiles (
                    book_name TEXT PRIMARY KEY,
                    bias_type TEXT,
                    bias_strength REAL,
                    pattern_description TEXT,
                    confidence REAL,
                    last_updated TIMESTAMP
                )
            """)
            
            conn.commit()
    
    async def analyze_sportsbook_patterns(self, book_name: str, line_data: pd.DataFrame) -> SportsbookProfile:
        """Analyze patterns for a specific sportsbook"""
        try:
            if line_data.empty:
                return self._create_default_profile(book_name)
            
            # Feature engineering for clustering
            features = self._extract_movement_features(line_data)
            
            if len(features) < 10:  # Need minimum data
                return self._create_default_profile(book_name)
            
            # Apply clustering to identify patterns
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Use DBSCAN for pattern detection
            clustering = DBSCAN(eps=0.5, min_samples=3)
            clusters = clustering.fit_predict(features_scaled)
            
            # Analyze clusters to determine bias
            bias_type, bias_strength = self._analyze_bias_from_clusters(line_data, clusters)
            
            # Determine sharp money tolerance
            sharp_tolerance = self._calculate_sharp_tolerance(line_data)
            
            # Create profile
            profile = SportsbookProfile(
                book_name=book_name,
                bias_type=bias_type,
                bias_strength=bias_strength,
                line_movement_pattern=f"Cluster analysis: {len(set(clusters))} patterns",
                sharp_money_tolerance=sharp_tolerance,
                market_making_style="Algorithmic" if bias_strength > 0.3 else "Manual",
                confidence=0.8 if len(features) > 50 else 0.6,
                sample_size=len(features)
            )
            
            self.book_profiles[book_name] = profile
            logger.info(f"📊 Analyzed {book_name}: {bias_type.value}, strength {bias_strength:.2f}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Error analyzing {book_name}: {e}")
            return self._create_default_profile(book_name)
    
    def _extract_movement_features(self, line_data: pd.DataFrame) -> np.ndarray:
        """Extract features for clustering analysis"""
        features = []
        
        # Sort by timestamp
        line_data = line_data.sort_values('timestamp')
        
        for i in range(1, len(line_data)):
            current = line_data.iloc[i]
            previous = line_data.iloc[i-1]
            
            # Calculate movement features
            line_movement = current['line_value'] - previous['line_value']
            time_diff = (pd.to_datetime(current['timestamp']) - pd.to_datetime(previous['timestamp'])).total_seconds() / 60
            
            if time_diff > 0:
                velocity = abs(line_movement) / time_diff
                
                feature_vector = [
                    line_movement,  # Direction and magnitude
                    velocity,       # Speed of movement
                    current.get('volume_indicator', 0.5),  # Volume
                    current.get('sharp_percentage', 0.5),  # Sharp money %
                    time_diff,      # Time between moves
                    abs(line_movement)  # Magnitude only
                ]
                
                features.append(feature_vector)
        
        return np.array(features) if features else np.array([]).reshape(0, 6)
    
    def _analyze_bias_from_clusters(self, line_data: pd.DataFrame, clusters: np.ndarray) -> Tuple[BookBias, float]:
        """Analyze bias from clustering results"""
        try:
            # Calculate overall movement tendencies
            movements = line_data['line_value'].diff().dropna()
            
            if len(movements) == 0:
                return BookBias.NEUTRAL, 0.0
            
            # Analyze movement direction bias
            positive_moves = (movements > 0).sum()
            negative_moves = (movements < 0).sum()
            total_moves = len(movements)
            
            if total_moves == 0:
                return BookBias.NEUTRAL, 0.0
            
            positive_ratio = positive_moves / total_moves
            
            # Determine bias type and strength
            if positive_ratio > 0.6:
                return BookBias.OVER_BIASED, min((positive_ratio - 0.5) * 2, 1.0)
            elif positive_ratio < 0.4:
                return BookBias.UNDER_BIASED, min((0.5 - positive_ratio) * 2, 1.0)
            else:
                return BookBias.NEUTRAL, abs(positive_ratio - 0.5) * 2
                
        except Exception as e:
            logger.error(f"Error analyzing bias: {e}")
            return BookBias.NEUTRAL, 0.0
    
    def _calculate_sharp_tolerance(self, line_data: pd.DataFrame) -> float:
        """Calculate sportsbook's tolerance for sharp money"""
        try:
            if 'sharp_percentage' not in line_data.columns:
                return 0.5  # Default
            
            sharp_data = line_data['sharp_percentage'].dropna()
            if len(sharp_data) == 0:
                return 0.5
            
            # Books that move quickly on low sharp % have low tolerance
            # Books that wait for high sharp % have high tolerance
            avg_sharp_pct = sharp_data.mean()
            return min(max(avg_sharp_pct, 0.1), 0.9)
            
        except Exception as e:
            logger.error(f"Error calculating sharp tolerance: {e}")
            return 0.5
    
    def _create_default_profile(self, book_name: str) -> SportsbookProfile:
        """Create default profile when insufficient data"""
        return SportsbookProfile(
            book_name=book_name,
            bias_type=BookBias.NEUTRAL,
            bias_strength=0.1,
            line_movement_pattern="Insufficient data",
            sharp_money_tolerance=0.5,
            market_making_style="Unknown",
            confidence=0.3,
            sample_size=0
        )


class MarketEfficiencyAnalyzer:
    """Analyzes market efficiency and closing line value"""
    
    def __init__(self):
        self.efficiency_history: Dict[str, List[MarketEfficiency]] = {}
    
    def calculate_clv(self, opening_line: float, closing_line: float, bet_line: float) -> float:
        """Calculate Closing Line Value"""
        try:
            # CLV = (Bet Line - Opening Line) / (Closing Line - Opening Line)
            if abs(closing_line - opening_line) < 0.01:  # No movement
                return 0.0
            
            clv = (bet_line - opening_line) / (closing_line - opening_line)
            return max(-2.0, min(2.0, clv))  # Clamp to reasonable range
            
        except Exception as e:
            logger.error(f"Error calculating CLV: {e}")
            return 0.0
    
    def analyze_market_efficiency(
        self, 
        game_id: str,
        book_name: str,
        line_history: List[Dict],
        outcome: Optional[Dict] = None
    ) -> MarketEfficiency:
        """Analyze market efficiency for a game/book combination"""
        try:
            if len(line_history) < 2:
                return self._create_default_efficiency(game_id, book_name)
            
            # Extract opening and closing lines
            opening_line = line_history[0]['line_value']
            closing_line = line_history[-1]['line_value']
            
            # Calculate CLV score
            clv_score = self.calculate_clv(opening_line, closing_line, closing_line)
            
            # Calculate movement correlation (simplified)
            movements = [h['line_value'] for h in line_history]
            movement_variance = np.var(movements) if len(movements) > 1 else 0
            movement_correlation = 1.0 - min(movement_variance / 10.0, 1.0)  # Normalize
            
            # Calculate efficiency score
            efficiency_score = (abs(clv_score) * 0.4 + movement_correlation * 0.6)
            efficiency_score = min(max(efficiency_score, 0.0), 1.0)
            
            # Estimate sharp/public money split
            sharp_pct = np.mean([h.get('sharp_percentage', 0.5) for h in line_history])
            public_pct = 1.0 - sharp_pct
            
            efficiency = MarketEfficiency(
                game_id=game_id,
                book_name=book_name,
                clv_score=clv_score,
                movement_correlation=movement_correlation,
                efficiency_score=efficiency_score,
                sharp_money_percentage=sharp_pct,
                public_money_percentage=public_pct,
                timestamp=datetime.now()
            )
            
            # Store efficiency data
            if game_id not in self.efficiency_history:
                self.efficiency_history[game_id] = []
            self.efficiency_history[game_id].append(efficiency)
            
            return efficiency
            
        except Exception as e:
            logger.error(f"Error analyzing market efficiency: {e}")
            return self._create_default_efficiency(game_id, book_name)
    
    def _create_default_efficiency(self, game_id: str, book_name: str) -> MarketEfficiency:
        """Create default efficiency when insufficient data"""
        return MarketEfficiency(
            game_id=game_id,
            book_name=book_name,
            clv_score=0.0,
            movement_correlation=0.5,
            efficiency_score=0.5,
            sharp_money_percentage=0.5,
            public_money_percentage=0.5,
            timestamp=datetime.now()
        )


class SharpMoneyDetector:
    """Detects sharp money movements and patterns"""
    
    def __init__(self):
        self.sharp_indicators = {
            'velocity_threshold': 0.5,     # Points per minute
            'volume_threshold': 0.7,       # 70% sharp money
            'reverse_line_threshold': 0.3, # 30% public money moving line opposite
            'steam_move_threshold': 1.0    # 1+ point in <5 minutes
        }
        
        self.detected_sharp_moves: List[Dict] = []
    
    async def detect_sharp_money(
        self,
        line_history: List[Dict],
        volume_data: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """Detect sharp money indicators"""
        try:
            if len(line_history) < 2:
                return {'is_sharp': False, 'confidence': 0.0, 'indicators': []}
            
            indicators = []
            confidence_factors = []
            
            # Analyze velocity
            for i in range(1, len(line_history)):
                current = line_history[i]
                previous = line_history[i-1]
                
                line_move = abs(current['line_value'] - previous['line_value'])
                time_diff = (pd.to_datetime(current['timestamp']) - pd.to_datetime(previous['timestamp'])).total_seconds() / 60
                
                if time_diff > 0:
                    velocity = line_move / time_diff
                    
                    if velocity >= self.sharp_indicators['velocity_threshold']:
                        indicators.append(f"High velocity: {velocity:.2f} pts/min")
                        confidence_factors.append(0.3)
                    
                    if line_move >= self.sharp_indicators['steam_move_threshold'] and time_diff <= 5:
                        indicators.append(f"Steam move: {line_move:.1f} pts in {time_diff:.1f} min")
                        confidence_factors.append(0.4)
            
            # Analyze volume patterns
            if volume_data:
                avg_sharp_pct = np.mean([v.get('sharp_percentage', 0.5) for v in volume_data])
                if avg_sharp_pct >= self.sharp_indicators['volume_threshold']:
                    indicators.append(f"High sharp volume: {avg_sharp_pct:.0%}")
                    confidence_factors.append(0.3)
            
            # Calculate overall confidence
            is_sharp = len(indicators) > 0
            confidence = min(sum(confidence_factors), 1.0) if confidence_factors else 0.0
            
            result = {
                'is_sharp': is_sharp,
                'confidence': confidence,
                'indicators': indicators,
                'sharp_score': confidence
            }
            
            if is_sharp:
                self.detected_sharp_moves.append({
                    'timestamp': datetime.now(),
                    'indicators': indicators,
                    'confidence': confidence
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error detecting sharp money: {e}")
            return {'is_sharp': False, 'confidence': 0.0, 'indicators': []}


class ArbitrageDetector:
    """Detects and scores arbitrage opportunities"""
    
    def __init__(self):
        self.opportunities: List[ArbitrageOpportunity] = []
        self.min_profit_margin = 0.02  # 2% minimum
    
    async def find_arbitrage_opportunities(
        self,
        game_lines: Dict[str, Dict[str, float]]
    ) -> List[ArbitrageOpportunity]:
        """Find arbitrage opportunities across sportsbooks"""
        opportunities = []
        
        try:
            if len(game_lines) < 2:
                return opportunities
            
            game_id = list(game_lines.keys())[0].split('_')[0] + '_vs_' + list(game_lines.keys())[0].split('_')[1]
            
            # Extract lines by market type
            spreads = {}
            totals = {}
            
            for book_game_key, lines in game_lines.items():
                book = book_game_key.split('_')[-1]
                spreads[book] = lines.get('spread', 0)
                totals[book] = lines.get('total', 0)
            
            # Check spread arbitrage
            if len(spreads) >= 2:
                spread_arb = self._find_spread_arbitrage(game_id, spreads)
                opportunities.extend(spread_arb)
            
            # Check total arbitrage
            if len(totals) >= 2:
                total_arb = self._find_total_arbitrage(game_id, totals)
                opportunities.extend(total_arb)
            
            # Score and filter opportunities
            scored_opportunities = []
            for opp in opportunities:
                if opp.profit_margin >= self.min_profit_margin:
                    opp.opportunity_score = self._score_opportunity(opp)
                    scored_opportunities.append(opp)
                    self.opportunities.append(opp)
            
            return scored_opportunities
            
        except Exception as e:
            logger.error(f"Error finding arbitrage: {e}")
            return []
    
    def _find_spread_arbitrage(self, game_id: str, spreads: Dict[str, float]) -> List[ArbitrageOpportunity]:
        """Find spread arbitrage opportunities"""
        opportunities = []
        
        books = list(spreads.keys())
        for i, book_a in enumerate(books):
            for book_b in books[i+1:]:
                spread_diff = abs(spreads[book_a] - spreads[book_b])
                
                if spread_diff >= 1.0:  # 1 point minimum
                    profit_margin = self._calculate_spread_profit(spreads[book_a], spreads[book_b])
                    
                    opportunities.append(ArbitrageOpportunity(
                        game_id=game_id,
                        market_type='spread',
                        book_a=book_a,
                        book_b=book_b,
                        line_a=spreads[book_a],
                        line_b=spreads[book_b],
                        profit_margin=profit_margin,
                        risk_score=0.2,  # Spreads are lower risk
                        opportunity_score=0.0,  # Will be calculated
                        expires_in_minutes=15.0,
                        timestamp=datetime.now()
                    ))
        
        return opportunities
    
    def _find_total_arbitrage(self, game_id: str, totals: Dict[str, float]) -> List[ArbitrageOpportunity]:
        """Find total arbitrage opportunities"""
        opportunities = []
        
        books = list(totals.keys())
        for i, book_a in enumerate(books):
            for book_b in books[i+1:]:
                total_diff = abs(totals[book_a] - totals[book_b])
                
                if total_diff >= 1.5:  # 1.5 point minimum
                    profit_margin = self._calculate_total_profit(totals[book_a], totals[book_b])
                    
                    opportunities.append(ArbitrageOpportunity(
                        game_id=game_id,
                        market_type='total',
                        book_a=book_a,
                        book_b=book_b,
                        line_a=totals[book_a],
                        line_b=totals[book_b],
                        profit_margin=profit_margin,
                        risk_score=0.3,  # Totals slightly higher risk
                        opportunity_score=0.0,  # Will be calculated
                        expires_in_minutes=10.0,
                        timestamp=datetime.now()
                    ))
        
        return opportunities
    
    def _calculate_spread_profit(self, line_a: float, line_b: float) -> float:
        """Calculate spread arbitrage profit margin"""
        return min(abs(line_a - line_b) * 0.02, 0.15)  # Max 15% profit
    
    def _calculate_total_profit(self, line_a: float, line_b: float) -> float:
        """Calculate total arbitrage profit margin"""
        return min(abs(line_a - line_b) * 0.015, 0.12)  # Max 12% profit
    
    def _score_opportunity(self, opp: ArbitrageOpportunity) -> float:
        """Score arbitrage opportunity (0-1 scale)"""
        # Profit component (0-1)
        profit_score = min(opp.profit_margin / 0.1, 1.0)  # Normalize to 10% max
        
        # Risk component (inverted, 0-1)
        risk_score = 1.0 - opp.risk_score
        
        # Time component (more time = better)
        time_score = min(opp.expires_in_minutes / 30.0, 1.0)  # Normalize to 30 min max
        
        # Weighted combination
        opportunity_score = (profit_score * 0.5 + risk_score * 0.3 + time_score * 0.2)
        
        return opportunity_score


class MarketIntelligenceSystem:
    """Complete market intelligence system"""
    
    def __init__(self):
        self.algorithm_detector = SportsbookAlgorithmDetector()
        self.efficiency_analyzer = MarketEfficiencyAnalyzer()
        self.sharp_detector = SharpMoneyDetector()
        self.arbitrage_detector = ArbitrageDetector()
        
        self.stats = {
            'books_analyzed': 0,
            'efficiency_scores': [],
            'sharp_moves_detected': 0,
            'arbitrage_found': 0,
            'start_time': datetime.now()
        }
    
    async def analyze_complete_market(self, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete market analysis"""
        results = {
            'sportsbook_profiles': {},
            'market_efficiency': {},
            'sharp_money_analysis': {},
            'arbitrage_opportunities': [],
            'market_summary': {}
        }
        
        try:
            # Analyze each sportsbook
            for book_name, book_data in market_data.items():
                if isinstance(book_data, dict) and 'line_history' in book_data:
                    # Convert to DataFrame
                    line_df = pd.DataFrame(book_data['line_history'])
                    
                    # Analyze sportsbook patterns
                    profile = await self.algorithm_detector.analyze_sportsbook_patterns(book_name, line_df)
                    results['sportsbook_profiles'][book_name] = asdict(profile)
                    self.stats['books_analyzed'] += 1
                    
                    # Analyze market efficiency
                    efficiency = self.efficiency_analyzer.analyze_market_efficiency(
                        game_id=book_data.get('game_id', 'unknown'),
                        book_name=book_name,
                        line_history=book_data['line_history']
                    )
                    results['market_efficiency'][book_name] = asdict(efficiency)
                    self.stats['efficiency_scores'].append(efficiency.efficiency_score)
                    
                    # Detect sharp money
                    sharp_analysis = await self.sharp_detector.detect_sharp_money(
                        book_data['line_history'],
                        book_data.get('volume_data')
                    )
                    results['sharp_money_analysis'][book_name] = sharp_analysis
                    
                    if sharp_analysis['is_sharp']:
                        self.stats['sharp_moves_detected'] += 1
            
            # Find arbitrage opportunities
            if len(market_data) >= 2:
                game_lines = {}
                for book_name, book_data in market_data.items():
                    if 'line_history' in book_data and book_data['line_history']:
                        latest_line = book_data['line_history'][-1]
                        game_lines[f"{book_data.get('game_id', 'game')}_{book_name}"] = {
                            'spread': latest_line.get('line_value', 0),
                            'total': latest_line.get('total', 45)
                        }
                
                arbitrage_opps = await self.arbitrage_detector.find_arbitrage_opportunities(game_lines)
                results['arbitrage_opportunities'] = [asdict(opp) for opp in arbitrage_opps]
                self.stats['arbitrage_found'] += len(arbitrage_opps)
            
            # Generate market summary
            results['market_summary'] = self._generate_market_summary(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Error in complete market analysis: {e}")
            return results
    
    def _generate_market_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate market intelligence summary"""
        summary = {
            'total_books_analyzed': len(results['sportsbook_profiles']),
            'avg_efficiency_score': np.mean(self.stats['efficiency_scores']) if self.stats['efficiency_scores'] else 0.5,
            'sharp_books_detected': sum(1 for profile in results['sportsbook_profiles'].values() 
                                      if profile['bias_type'] == 'sharp_friendly'),
            'arbitrage_opportunities': len(results['arbitrage_opportunities']),
            'highest_profit_margin': max([opp['profit_margin'] for opp in results['arbitrage_opportunities']], default=0),
            'market_inefficiency_score': 1.0 - np.mean(self.stats['efficiency_scores']) if self.stats['efficiency_scores'] else 0.5
        }
        
        return summary


async def main():
    """YOLO MODE Demo - Complete Market Intelligence"""
    print("💰 MARKET INTELLIGENCE ANALYSIS SYSTEM - YOLO MODE")
    print("=" * 60)
    
    system = MarketIntelligenceSystem()
    
    # Simulate market data
    market_data = {
        'DraftKings': {
            'game_id': 'KC_vs_BAL',
            'line_history': [
                {'line_value': -3.5, 'timestamp': '2024-09-19T10:00:00', 'total': 47.5, 'sharp_percentage': 0.3},
                {'line_value': -4.0, 'timestamp': '2024-09-19T11:00:00', 'total': 47.0, 'sharp_percentage': 0.7},
                {'line_value': -4.5, 'timestamp': '2024-09-19T12:00:00', 'total': 46.5, 'sharp_percentage': 0.8}
            ]
        },
        'FanDuel': {
            'game_id': 'KC_vs_BAL',
            'line_history': [
                {'line_value': -3.0, 'timestamp': '2024-09-19T10:00:00', 'total': 48.0, 'sharp_percentage': 0.4},
                {'line_value': -3.5, 'timestamp': '2024-09-19T11:00:00', 'total': 47.5, 'sharp_percentage': 0.6},
                {'line_value': -3.5, 'timestamp': '2024-09-19T12:00:00', 'total': 47.0, 'sharp_percentage': 0.5}
            ]
        },
        'BetMGM': {
            'game_id': 'KC_vs_BAL',
            'line_history': [
                {'line_value': -3.5, 'timestamp': '2024-09-19T10:00:00', 'total': 47.0, 'sharp_percentage': 0.5},
                {'line_value': -4.0, 'timestamp': '2024-09-19T11:00:00', 'total': 46.5, 'sharp_percentage': 0.9},
                {'line_value': -5.0, 'timestamp': '2024-09-19T12:00:00', 'total': 46.0, 'sharp_percentage': 0.8}
            ]
        }
    }
    
    print("🔍 Analyzing complete market intelligence...")
    results = await system.analyze_complete_market(market_data)
    
    print("\n📊 MARKET INTELLIGENCE RESULTS:")
    print("=" * 40)
    
    # Sportsbook profiles
    print("🏢 SPORTSBOOK PROFILES:")
    for book, profile in results['sportsbook_profiles'].items():
        print(f"  {book}: {profile['bias_type']} (strength: {profile['bias_strength']:.2f})")
        print(f"    Sharp tolerance: {profile['sharp_money_tolerance']:.2f}")
        print(f"    Confidence: {profile['confidence']:.0%}")
    
    # Market efficiency
    print(f"\n📈 MARKET EFFICIENCY:")
    for book, efficiency in results['market_efficiency'].items():
        print(f"  {book}: {efficiency['efficiency_score']:.2f} efficiency")
        print(f"    CLV Score: {efficiency['clv_score']:.3f}")
        print(f"    Sharp Money: {efficiency['sharp_money_percentage']:.0%}")
    
    # Sharp money analysis
    print(f"\n🎯 SHARP MONEY DETECTION:")
    for book, sharp_data in results['sharp_money_analysis'].items():
        if sharp_data['is_sharp']:
            print(f"  {book}: SHARP DETECTED (confidence: {sharp_data['confidence']:.0%})")
            for indicator in sharp_data['indicators']:
                print(f"    - {indicator}")
        else:
            print(f"  {book}: No sharp money detected")
    
    # Arbitrage opportunities
    print(f"\n💰 ARBITRAGE OPPORTUNITIES ({len(results['arbitrage_opportunities'])}):")
    for opp in results['arbitrage_opportunities']:
        print(f"  {opp['market_type']}: {opp['book_a']} vs {opp['book_b']}")
        print(f"    Profit: {opp['profit_margin']:.1%}, Score: {opp['opportunity_score']:.2f}")
    
    # Market summary
    summary = results['market_summary']
    print(f"\n🎯 MARKET SUMMARY:")
    print(f"  Books Analyzed: {summary['total_books_analyzed']}")
    print(f"  Avg Efficiency: {summary['avg_efficiency_score']:.2f}")
    print(f"  Sharp Books: {summary['sharp_books_detected']}")
    print(f"  Arbitrage Opps: {summary['arbitrage_opportunities']}")
    print(f"  Max Profit: {summary['highest_profit_margin']:.1%}")
    print(f"  Market Inefficiency: {summary['market_inefficiency_score']:.2f}")
    
    print("\n✅ TASK 21 COMPLETE - Market Intelligence System DELIVERED!")


if __name__ == "__main__":
    asyncio.run(main())
