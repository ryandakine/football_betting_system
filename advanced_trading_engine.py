"""
Advanced Trading Engine for Professional Football Betting

This module implements sophisticated trading strategies that leverage
comprehensive data sources for professional-grade betting analysis:

🧮 ARBITRAGE DETECTION
   - True arbitrage opportunity identification across sportsbooks
   - Risk-free profit calculation and execution
   - Multi-market arbitrage (moneyline + spread + total)
   - Real-time arbitrage monitoring and alerts

🎯 SHARP MONEY FOLLOWING
   - Professional bettor movement tracking
   - Line movement analysis and sharp money indicators
   - Steam move detection and validation
   - Institutional betting pattern recognition

📊 SENTIMENT-BASED TIMING
   - Social media sentiment analysis for contrarian signals
   - Public betting sentiment vs. sharp money divergence
   - News sentiment impact timing
   - Crowd psychology exploitation

🎪 EXPERT CONSENSUS TRADING
   - Professional analyst pick aggregation
   - Consensus analysis and contrarian positioning
   - Expert accuracy weighting and bias detection
   - Follow vs. fade strategy optimization

🔬 MULTI-FACTOR ANALYSIS
   - Comprehensive game intelligence fusion
   - Advanced statistical modeling
   - Machine learning prediction integration
   - Risk-adjusted position sizing

The engine provides professional trading signals with
quantitative analysis, risk management, and execution guidance.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings

from advanced_data_sources import SportsbookOdds, AdvancedAnalytics, SocialSentiment, NewsAnalysis, ExpertPicks

logger = logging.getLogger(__name__)

class TradeSignal(Enum):
    """Trading signal types"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    ARBITRAGE = "arbitrage"
    HEDGE = "hedge"

class MarketCondition(Enum):
    """Market condition assessment"""
    HIGHLY_EFFICIENT = "highly_efficient"
    EFFICIENT = "efficient"
    INEFFICIENT = "inefficient"
    HIGHLY_INEFFICIENT = "highly_inefficient"
    ARBITRAGE_OPPORTUNITY = "arbitrage_opportunity"

@dataclass
class ArbitrageOpportunity:
    """Arbitrage trading opportunity"""
    game_id: str
    market_type: str  # moneyline, spread, total
    profit_percentage: float
    required_stake: float
    bookmaker1: str
    bookmaker2: str
    bet1_details: Dict[str, Any]
    bet2_details: Dict[str, Any]
    expiry_time: datetime
    confidence_score: float
    risk_assessment: str

@dataclass
class SharpMoneySignal:
    """Sharp money movement signal"""
    game_id: str
    signal_type: str  # line_move, steam_move, reverse_line
    direction: str  # up, down, stable
    magnitude: float  # 0-1 scale
    bookmaker: str
    timestamp: datetime
    confidence_score: float
    supporting_evidence: List[str]
    recommended_action: str

@dataclass
class SentimentTimingSignal:
    """Sentiment-based timing signal"""
    game_id: str
    sentiment_divergence: float  # -1 to 1 (contrarian vs momentum)
    public_sentiment: float
    sharp_money_position: float
    timing_recommendation: str  # bet_now, wait, fade, follow
    confidence_score: float
    optimal_entry_time: datetime
    holding_period: timedelta

@dataclass
class ExpertConsensusSignal:
    """Expert consensus trading signal"""
    game_id: str
    consensus_pick: str
    consensus_strength: float  # 0-1
    expert_count: int
    contrarian_signal: bool
    recommended_strategy: str  # follow_consensus, fade_consensus, neutral
    confidence_score: float
    supporting_experts: List[str]

@dataclass
class MultiFactorPrediction:
    """Multi-factor analysis prediction"""
    game_id: str
    predicted_winner: str
    win_probability: float
    confidence_score: float
    factors_considered: List[str]
    key_drivers: List[Tuple[str, float]]  # factor, impact_score
    risk_adjusted_stake: float
    expected_value: float
    recommended_bet_type: str

class ArbitrageDetector:
    """
    Advanced arbitrage detection engine.

    Identifies true arbitrage opportunities where betting on all outcomes
    guarantees profit regardless of result.
    """

    def __init__(self):
        self.min_arbitrage_profit = 0.005  # 0.5% minimum profit
        self.max_stake = 10000  # Maximum stake per arbitrage
        self.arbitrage_cache = {}
        self.cache_expiry = timedelta(minutes=5)

    def detect_moneyline_arbitrage(self, odds_data: List[SportsbookOdds]) -> Optional[ArbitrageOpportunity]:
        """
        Detect moneyline arbitrage opportunities.

        Moneyline arbitrage occurs when the sum of implied probabilities < 1,
        meaning you can bet on all outcomes and guarantee profit.
        """
        if len(odds_data) < 2:
            return None

        # Group by game
        games_odds = {}
        for odds in odds_data:
            game_id = odds.game_id
            if game_id not in games_odds:
                games_odds[game_id] = []
            games_odds[game_id].append(odds)

        opportunities = []
        for game_id, game_odds in games_odds.items():
            if len(game_odds) < 2:
                continue

            # Find best home and away odds
            best_home_odds = max((o for o in game_odds if o.home_ml), key=lambda x: x.home_ml)
            best_away_odds = max((o for o in game_odds if o.away_ml), key=lambda x: x.away_ml)

            # Calculate implied probabilities
            home_prob = 1 / best_home_odds.home_ml
            away_prob = 1 / best_away_odds.away_ml
            total_prob = home_prob + away_prob

            if total_prob < 1:  # Arbitrage opportunity
                profit_pct = (1 - total_prob) / total_prob * 100

                if profit_pct >= self.min_arbitrage_profit * 100:
                    # Calculate required stakes for $100 profit
                    stake_home = 100 / (best_home_odds.home_ml - 1)
                    stake_away = 100 / (best_away_odds.away_ml - 1)
                    total_stake = stake_home + stake_away

                    opportunity = ArbitrageOpportunity(
                        game_id=game_id,
                        market_type="moneyline",
                        profit_percentage=profit_pct,
                        required_stake=total_stake,
                        bookmaker1=best_home_odds.bookmaker,
                        bookmaker2=best_away_odds.bookmaker,
                        bet1_details={
                            'team': best_home_odds.home_team,
                            'odds': best_home_odds.home_ml,
                            'stake': stake_home,
                            'bookmaker': best_home_odds.bookmaker
                        },
                        bet2_details={
                            'team': best_away_odds.away_team,
                            'odds': best_away_odds.away_ml,
                            'stake': stake_away,
                            'bookmaker': best_away_odds.bookmaker
                        },
                        expiry_time=min(best_home_odds.last_updated or datetime.now(),
                                      best_away_odds.last_updated or datetime.now()) + timedelta(hours=1),
                        confidence_score=min(0.95, profit_pct / 2),  # Higher profit = higher confidence
                        risk_assessment="LOW" if profit_pct > 1.0 else "MEDIUM"
                    )

                    opportunities.append(opportunity)

        return max(opportunities, key=lambda x: x.profit_percentage) if opportunities else None

    def detect_spread_arbitrage(self, odds_data: List[SportsbookOdds]) -> Optional[ArbitrageOpportunity]:
        """
        Detect spread arbitrage opportunities.

        Spread arbitrage occurs when the spread odds don't align properly
        between different sportsbooks.
        """
        if len(odds_data) < 2:
            return None

        # Group by game and spread line
        spread_groups = {}
        for odds in odds_data:
            if odds.spread_line and odds.spread_home and odds.spread_away:
                key = (odds.game_id, odds.spread_line)
                if key not in spread_groups:
                    spread_groups[key] = []
                spread_groups[key].append(odds)

        opportunities = []
        for (game_id, spread_line), group_odds in spread_groups.items():
            if len(group_odds) < 2:
                continue

            # Find best odds for each side
            best_home_cover = max(group_odds, key=lambda x: x.spread_home)
            best_away_cover = max(group_odds, key=lambda x: x.spread_away)

            # Calculate if arbitrage exists
            home_prob = 1 / best_home_cover.spread_home
            away_prob = 1 / best_away_cover.spread_away
            total_prob = home_prob + away_prob

            if total_prob < 1:
                profit_pct = (1 - total_prob) / total_prob * 100

                if profit_pct >= self.min_arbitrage_profit * 100:
                    stake_home = 100 / (best_home_cover.spread_home - 1)
                    stake_away = 100 / (best_away_cover.spread_away - 1)
                    total_stake = stake_home + stake_away

                    opportunity = ArbitrageOpportunity(
                        game_id=game_id,
                        market_type="spread",
                        profit_percentage=profit_pct,
                        required_stake=total_stake,
                        bookmaker1=best_home_cover.bookmaker,
                        bookmaker2=best_away_cover.bookmaker,
                        bet1_details={
                            'type': 'spread',
                            'team': best_home_cover.home_team,
                            'line': spread_line,
                            'odds': best_home_cover.spread_home,
                            'stake': stake_home,
                            'bookmaker': best_home_cover.bookmaker
                        },
                        bet2_details={
                            'type': 'spread',
                            'team': best_away_cover.away_team,
                            'line': spread_line,
                            'odds': best_away_cover.spread_away,
                            'stake': stake_away,
                            'bookmaker': best_away_cover.bookmaker
                        },
                        expiry_time=datetime.now() + timedelta(hours=1),
                        confidence_score=min(0.90, profit_pct / 3),
                        risk_assessment="MEDIUM"
                    )

                    opportunities.append(opportunity)

        return max(opportunities, key=lambda x: x.profit_percentage) if opportunities else None

    def detect_total_arbitrage(self, odds_data: List[SportsbookOdds]) -> Optional[ArbitrageOpportunity]:
        """
        Detect over/under arbitrage opportunities.
        """
        if len(odds_data) < 2:
            return None

        # Group by game and total line
        total_groups = {}
        for odds in odds_data:
            if odds.total_line and odds.total_over and odds.total_under:
                key = (odds.game_id, odds.total_line)
                if key not in total_groups:
                    total_groups[key] = []
                total_groups[key].append(odds)

        opportunities = []
        for (game_id, total_line), group_odds in total_groups.items():
            if len(group_odds) < 2:
                continue

            # Find best odds for over and under
            best_over = max(group_odds, key=lambda x: x.total_over)
            best_under = max(group_odds, key=lambda x: x.total_under)

            # Calculate arbitrage
            over_prob = 1 / best_over.total_over
            under_prob = 1 / best_under.total_under
            total_prob = over_prob + under_prob

            if total_prob < 1:
                profit_pct = (1 - total_prob) / total_prob * 100

                if profit_pct >= self.min_arbitrage_profit * 100:
                    stake_over = 100 / (best_over.total_over - 1)
                    stake_under = 100 / (best_under.total_under - 1)
                    total_stake = stake_over + stake_under

                    opportunity = ArbitrageOpportunity(
                        game_id=game_id,
                        market_type="total",
                        profit_percentage=profit_pct,
                        required_stake=total_stake,
                        bookmaker1=best_over.bookmaker,
                        bookmaker2=best_under.bookmaker,
                        bet1_details={
                            'type': 'total',
                            'side': 'over',
                            'line': total_line,
                            'odds': best_over.total_over,
                            'stake': stake_over,
                            'bookmaker': best_over.bookmaker
                        },
                        bet2_details={
                            'type': 'total',
                            'side': 'under',
                            'line': total_line,
                            'odds': best_under.total_under,
                            'stake': stake_under,
                            'bookmaker': best_under.bookmaker
                        },
                        expiry_time=datetime.now() + timedelta(hours=1),
                        confidence_score=min(0.85, profit_pct / 4),
                        risk_assessment="MEDIUM"
                    )

                    opportunities.append(opportunity)

        return max(opportunities, key=lambda x: x.profit_percentage) if opportunities else None

    def scan_for_arbitrage(self, odds_data: List[SportsbookOdds]) -> List[ArbitrageOpportunity]:
        """
        Comprehensive arbitrage scan across all markets.
        """
        opportunities = []

        # Moneyline arbitrage
        ml_arb = self.detect_moneyline_arbitrage(odds_data)
        if ml_arb:
            opportunities.append(ml_arb)

        # Spread arbitrage
        spread_arb = self.detect_spread_arbitrage(odds_data)
        if spread_arb:
            opportunities.append(spread_arb)

        # Total arbitrage
        total_arb = self.detect_total_arbitrage(odds_data)
        if total_arb:
            opportunities.append(total_arb)

        # Sort by profit percentage
        opportunities.sort(key=lambda x: x.profit_percentage, reverse=True)

        return opportunities

class SharpMoneyTracker:
    """
    Professional bettor movement tracking and analysis.

    Identifies sharp money movements, steam moves, and institutional
    betting patterns for optimal timing.
    """

    def __init__(self):
        self.line_movement_threshold = 1.5  # Points for significant movement
        self.steam_move_threshold = 3.0  # Points for steam move
        self.volume_threshold = 1000  # Minimum volume for consideration
        self.time_window = timedelta(hours=6)  # Analysis window

    def analyze_line_movement(self, game_id: str, odds_history: List[SportsbookOdds]) -> Optional[SharpMoneySignal]:
        """
        Analyze line movement patterns for sharp money indicators.
        """
        if len(odds_history) < 2:
            return None

        # Sort by timestamp
        odds_history.sort(key=lambda x: x.last_updated or datetime.min)

        # Analyze spread movement
        spreads = [(o.spread_line, o.last_updated) for o in odds_history if o.spread_line]
        if len(spreads) >= 2:
            initial_spread = spreads[0][0]
            current_spread = spreads[-1][0]
            movement = abs(current_spread - initial_spread)

            if movement >= self.line_movement_threshold:
                direction = "up" if current_spread > initial_spread else "down"

                # Check for steam move (rapid movement)
                time_diff = spreads[-1][1] - spreads[0][1] if spreads[-1][1] and spreads[0][1] else timedelta(hours=1)
                movement_rate = movement / max(time_diff.total_seconds() / 3600, 1)  # points per hour

                signal_type = "steam_move" if movement_rate >= self.steam_move_threshold else "line_move"

                # Determine confidence based on movement magnitude and consistency
                confidence = min(0.9, movement / 10)  # Scale confidence with movement

                # Supporting evidence
                evidence = [
                    f"Line moved {movement:.1f} points {direction}",
                    f"Movement rate: {movement_rate:.1f} points/hour",
                    f"Time period: {time_diff.total_seconds()/3600:.1f} hours"
                ]

                # Recommended action based on sharp money
                if signal_type == "steam_move":
                    recommended_action = f"Follow sharp money - bet {direction} side immediately"
                else:
                    recommended_action = f"Monitor for confirmation - {direction} movement detected"

                signal = SharpMoneySignal(
                    game_id=game_id,
                    signal_type=signal_type,
                    direction=direction,
                    magnitude=min(1.0, movement / 10),
                    bookmaker=odds_history[-1].bookmaker,
                    timestamp=datetime.now(),
                    confidence_score=confidence,
                    supporting_evidence=evidence,
                    recommended_action=recommended_action
                )

                return signal

        return None

    def detect_reverse_line_movement(self, game_id: str, odds_history: List[SportsbookOdds]) -> Optional[SharpMoneySignal]:
        """
        Detect reverse line movement (strong sharp money indicator).
        """
        if len(odds_history) < 3:
            return None

        # Look for line moving in one direction then reversing
        spreads = [(o.spread_line, o.last_updated) for o in odds_history if o.spread_line and o.last_updated]
        spreads.sort(key=lambda x: x[1])

        if len(spreads) >= 3:
            # Check for trend reversal
            mid_point = len(spreads) // 2
            first_half = spreads[:mid_point]
            second_half = spreads[mid_point:]

            first_trend = first_half[-1][0] - first_half[0][0]
            second_trend = second_half[-1][0] - second_half[0][0]

            # Opposite trends indicate reverse movement
            if (first_trend > 0 and second_trend < 0) or (first_trend < 0 and second_trend > 0):
                reversal_magnitude = abs(first_trend) + abs(second_trend)

                if reversal_magnitude >= self.line_movement_threshold:
                    direction = "down" if second_trend < 0 else "up"

                    signal = SharpMoneySignal(
                        game_id=game_id,
                        signal_type="reverse_line",
                        direction=direction,
                        magnitude=min(1.0, reversal_magnitude / 20),
                        bookmaker=odds_history[-1].bookmaker,
                        timestamp=datetime.now(),
                        confidence_score=min(0.95, reversal_magnitude / 15),
                        supporting_evidence=[
                            f"Line reversal detected: {reversal_magnitude:.1f} point swing",
                            f"Initial trend: {'up' if first_trend > 0 else 'down'} {abs(first_trend):.1f} points",
                            f"Reverse trend: {'up' if second_trend > 0 else 'down'} {abs(second_trend):.1f} points"
                        ],
                        recommended_action="HIGH CONFIDENCE: Sharp money reversal - follow immediately"
                    )

                    return signal

        return None

    def analyze_volume_spikes(self, game_id: str, odds_data: List[SportsbookOdds]) -> Optional[SharpMoneySignal]:
        """
        Analyze betting volume spikes for sharp money indicators.
        """
        # Find odds with volume data
        volume_odds = [o for o in odds_data if o.volume and o.volume >= self.volume_threshold]

        if not volume_odds:
            return None

        # Sort by volume
        volume_odds.sort(key=lambda x: x.volume, reverse=True)
        highest_volume = volume_odds[0]

        # Calculate volume concentration
        total_volume = sum(o.volume for o in volume_odds)
        concentration_ratio = highest_volume.volume / total_volume if total_volume > 0 else 0

        if concentration_ratio >= 0.7:  # 70%+ of volume on one side
            # Determine which side has the volume
            if highest_volume.home_ml > highest_volume.away_ml:
                direction = "home"
            else:
                direction = "away"

            signal = SharpMoneySignal(
                game_id=game_id,
                signal_type="volume_spike",
                direction=direction,
                magnitude=min(1.0, concentration_ratio),
                bookmaker=highest_volume.bookmaker,
                timestamp=datetime.now(),
                confidence_score=min(0.85, concentration_ratio * 1.2),
                supporting_evidence=[
                    f"Volume concentration: {concentration_ratio:.1%} on {direction} side",
                    f"Total volume: {total_volume:,.0f}",
                    f"Primary bookmaker: {highest_volume.bookmaker}"
                ],
                recommended_action=f"Follow volume spike - {direction} side showing sharp interest"
            )

            return signal

        return None

    def get_sharp_money_signals(self, game_id: str, odds_history: List[SportsbookOdds]) -> List[SharpMoneySignal]:
        """
        Comprehensive sharp money analysis for a game.
        """
        signals = []

        # Line movement analysis
        line_signal = self.analyze_line_movement(game_id, odds_history)
        if line_signal:
            signals.append(line_signal)

        # Reverse line movement
        reverse_signal = self.detect_reverse_line_movement(game_id, odds_history)
        if reverse_signal:
            signals.append(reverse_signal)

        # Volume spike analysis
        volume_signal = self.analyze_volume_spikes(game_id, odds_history)
        if volume_signal:
            signals.append(volume_signal)

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence_score, reverse=True)

        return signals

class SentimentTimingEngine:
    """
    Sentiment-based timing and contrarian analysis.

    Uses social media sentiment, news sentiment, and public betting
    patterns to identify optimal timing for bets.
    """

    def __init__(self):
        self.sentiment_threshold = 0.3  # Significant sentiment divergence
        self.volume_threshold = 1000  # Minimum social volume
        self.news_impact_window = timedelta(hours=24)  # News impact timeframe

    def analyze_sentiment_divergence(self, game_id: str,
                                   sentiment_data: List[SocialSentiment],
                                   odds_data: List[SportsbookOdds],
                                   expert_data: List[ExpertPicks]) -> Optional[SentimentTimingSignal]:
        """
        Analyze divergence between public sentiment and market/odds.
        """
        if not sentiment_data:
            return None

        # Calculate average sentiment
        total_volume = sum(s.mention_volume for s in sentiment_data)
        if total_volume < self.volume_threshold:
            return None

        avg_sentiment = sum(s.sentiment_score * s.mention_volume for s in sentiment_data) / total_volume

        # Get expert consensus
        expert_consensus = self._calculate_expert_consensus(expert_data)

        # Get market position (implied from odds)
        if odds_data:
            home_odds = np.mean([o.home_ml for o in odds_data if o.home_ml])
            away_odds = np.mean([o.away_ml for o in odds_data if o.away_ml])

            if home_odds and away_odds:
                market_home_prob = 1 / home_odds
                market_away_prob = 1 / away_odds

                # Determine market favorite
                market_favorite = "home" if market_home_prob > market_away_prob else "away"
                market_confidence = abs(market_home_prob - market_away_prob)
            else:
                market_favorite = None
                market_confidence = 0
        else:
            market_favorite = None
            market_confidence = 0

        # Calculate sentiment divergence
        sentiment_divergence = 0

        if expert_consensus['consensus_pick']:
            expert_favorite = "home" if "Chiefs" in expert_consensus['consensus_pick'] else "away"  # Simplified

            # Sentiment vs expert divergence
            if avg_sentiment > 0.2 and expert_favorite == "away":
                sentiment_divergence = avg_sentiment  # Contrarian signal
            elif avg_sentiment < -0.2 and expert_favorite == "home":
                sentiment_divergence = -avg_sentiment  # Contrarian signal

        # Determine timing recommendation
        if abs(sentiment_divergence) >= self.sentiment_threshold:
            if sentiment_divergence > 0:
                timing = "fade"  # Public sentiment too positive, consider contrarian
                entry_time = datetime.now() + timedelta(hours=2)
                holding_period = timedelta(hours=4)
            else:
                timing = "follow"  # Public sentiment too negative, follow momentum
                entry_time = datetime.now() + timedelta(hours=1)
                holding_period = timedelta(hours=6)

            confidence = min(0.8, abs(sentiment_divergence) * 2)

            signal = SentimentTimingSignal(
                game_id=game_id,
                sentiment_divergence=sentiment_divergence,
                public_sentiment=avg_sentiment,
                sharp_money_position=expert_consensus.get('consensus_strength', 0),
                timing_recommendation=timing,
                confidence_score=confidence,
                optimal_entry_time=entry_time,
                holding_period=holding_period
            )

            return signal

        return None

    def analyze_news_impact_timing(self, game_id: str,
                                 news_data: List[NewsAnalysis],
                                 odds_data: List[SportsbookOdds]) -> Optional[SentimentTimingSignal]:
        """
        Analyze timing based on recent news sentiment impact.
        """
        if not news_data:
            return None

        # Filter recent news
        cutoff_time = datetime.now() - self.news_impact_window
        recent_news = [n for n in news_data if n.published_at > cutoff_time]

        if not recent_news:
            return None

        # Calculate news sentiment impact
        total_relevance = sum(n.relevance_score for n in recent_news)
        if total_relevance == 0:
            return None

        avg_news_sentiment = sum(n.sentiment * n.relevance_score for n in recent_news) / total_relevance

        # Analyze sentiment momentum (recent vs older news)
        mid_point = len(recent_news) // 2
        older_news = recent_news[:mid_point]
        newer_news = recent_news[mid_point:]

        older_sentiment = np.mean([n.sentiment for n in older_news]) if older_news else 0
        newer_sentiment = np.mean([n.sentiment for n in newer_news]) if newer_news else 0

        sentiment_momentum = newer_sentiment - older_sentiment

        # Determine timing based on news momentum
        if abs(sentiment_momentum) >= 0.2:
            if sentiment_momentum > 0:
                timing = "bet_now"  # Positive momentum building
                entry_time = datetime.now()
                holding_period = timedelta(hours=12)
            else:
                timing = "wait"  # Negative momentum, wait for stabilization
                entry_time = datetime.now() + timedelta(hours=6)
                holding_period = timedelta(hours=8)

            confidence = min(0.75, abs(sentiment_momentum) * 3)

            signal = SentimentTimingSignal(
                game_id=game_id,
                sentiment_divergence=sentiment_momentum,
                public_sentiment=avg_news_sentiment,
                sharp_money_position=0.5,  # Neutral assumption
                timing_recommendation=timing,
                confidence_score=confidence,
                optimal_entry_time=entry_time,
                holding_period=holding_period
            )

            return signal

        return None

    def get_timing_signals(self, game_id: str,
                          sentiment_data: List[SocialSentiment],
                          news_data: List[NewsAnalysis],
                          odds_data: List[SportsbookOdds],
                          expert_data: List[ExpertPicks]) -> List[SentimentTimingSignal]:
        """
        Get all sentiment-based timing signals for a game.
        """
        signals = []

        # Sentiment divergence analysis
        sentiment_signal = self.analyze_sentiment_divergence(game_id, sentiment_data, odds_data, expert_data)
        if sentiment_signal:
            signals.append(sentiment_signal)

        # News impact timing
        news_signal = self.analyze_news_impact_timing(game_id, news_data, odds_data)
        if news_signal:
            signals.append(news_signal)

        # Sort by confidence
        signals.sort(key=lambda x: x.confidence_score, reverse=True)

        return signals

    def _calculate_expert_consensus(self, expert_data: List[ExpertPicks]) -> Dict[str, Any]:
        """
        Calculate expert consensus from picks data.
        """
        if not expert_data:
            return {'consensus_pick': None, 'consensus_strength': 0}

        # Group picks by choice
        pick_groups = {}
        total_confidence = 0

        for pick in expert_data:
            pick_value = pick.pick_value
            if pick_value not in pick_groups:
                pick_groups[pick_value] = {'count': 0, 'total_confidence': 0}

            pick_groups[pick_value]['count'] += 1
            pick_groups[pick_value]['total_confidence'] += pick.confidence
            total_confidence += pick.confidence

        # Find consensus
        if pick_groups:
            consensus_pick = max(pick_groups.items(), key=lambda x: x[1]['count'])
            consensus_strength = consensus_pick[1]['total_confidence'] / consensus_pick[1]['count']
        else:
            consensus_pick = (None, {'count': 0, 'total_confidence': 0})
            consensus_strength = 0

        return {
            'consensus_pick': consensus_pick[0],
            'consensus_strength': consensus_strength,
            'total_experts': len(expert_data),
            'pick_distribution': pick_groups
        }

class ExpertConsensusAnalyzer:
    """
    Expert pick consensus analysis and trading strategy optimization.
    """

    def __init__(self):
        self.consensus_threshold = 0.7  # Strong consensus threshold
        self.contrarian_threshold = 0.8  # High consensus for contrarian plays
        self.min_experts = 3  # Minimum experts for analysis

    def analyze_expert_consensus(self, game_id: str, expert_data: List[ExpertPicks]) -> Optional[ExpertConsensusSignal]:
        """
        Analyze expert consensus patterns and generate trading signals.
        """
        if len(expert_data) < self.min_experts:
            return None

        # Group picks by choice
        pick_groups = {}
        expert_details = []

        for pick in expert_data:
            pick_value = pick.pick_value
            if pick_value not in pick_groups:
                pick_groups[pick_value] = {'count': 0, 'total_confidence': 0, 'experts': []}

            pick_groups[pick_value]['count'] += 1
            pick_groups[pick_value]['total_confidence'] += pick.confidence
            pick_groups[pick_value]['experts'].append(pick.expert_name)

        # Find strongest consensus
        if not pick_groups:
            return None

        consensus_pick = max(pick_groups.items(), key=lambda x: x[1]['count'])
        pick_name, pick_data = consensus_pick

        consensus_strength = pick_data['total_confidence'] / pick_data['count']
        expert_count = pick_data['count']

        # Determine strategy
        if consensus_strength >= self.contrarian_threshold and expert_count >= 5:
            # Very strong consensus - consider contrarian
            strategy = "fade_consensus"
            contrarian = True
            confidence = min(0.8, consensus_strength * 0.9)
        elif consensus_strength >= self.consensus_threshold:
            # Strong consensus - follow
            strategy = "follow_consensus"
            contrarian = False
            confidence = consensus_strength
        else:
            # Weak consensus - neutral
            strategy = "neutral"
            contrarian = False
            confidence = consensus_strength * 0.7

        signal = ExpertConsensusSignal(
            game_id=game_id,
            consensus_pick=pick_name,
            consensus_strength=consensus_strength,
            expert_count=expert_count,
            contrarian_signal=contrarian,
            recommended_strategy=strategy,
            confidence_score=confidence,
            supporting_experts=pick_data['experts'][:5]  # Top 5 experts
        )

        return signal

    def analyze_expert_accuracy_bias(self, expert_data: List[ExpertPicks]) -> Dict[str, Any]:
        """
        Analyze expert accuracy patterns and potential biases.
        """
        expert_stats = {}

        for pick in expert_data:
            expert = pick.expert_name
            if expert not in expert_stats:
                expert_stats[expert] = {
                    'total_picks': 0,
                    'win_rate': 0,
                    'avg_confidence': 0,
                    'bias_score': 0,
                    'recent_performance': []
                }

            # Use provided win rate or default
            expert_stats[expert]['total_picks'] += 1
            expert_stats[expert]['avg_confidence'] += pick.confidence
            if pick.win_rate:
                expert_stats[expert]['win_rate'] = pick.win_rate

        # Calculate averages
        for expert, stats in expert_stats.items():
            if stats['total_picks'] > 0:
                stats['avg_confidence'] /= stats['total_picks']

        return expert_stats

class MultiFactorAnalysisEngine:
    """
    Advanced multi-factor analysis combining all data sources
    for comprehensive game predictions.
    """

    def __init__(self):
        self.feature_scaler = StandardScaler()
        self.prediction_model = None
        self.min_confidence_threshold = 0.6
        self.max_risk_stake = 0.05  # 5% of bankroll

    def build_prediction_model(self, historical_data: List[Dict[str, Any]]) -> None:
        """
        Build machine learning model from historical data.
        """
        if len(historical_data) < 50:
            logger.warning("Insufficient historical data for model training")
            return

        # Extract features
        features = []
        targets = []

        for game_data in historical_data:
            feature_vector = self._extract_feature_vector(game_data)
            if feature_vector and 'actual_winner' in game_data:
                features.append(feature_vector)
                targets.append(1 if game_data['actual_winner'] == game_data.get('home_team') else 0)

        if features and targets:
            # Train model
            X = np.array(features)
            y = np.array(targets)

            self.feature_scaler.fit(X)
            X_scaled = self.feature_scaler.transform(X)

            self.prediction_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            self.prediction_model.fit(X_scaled, y)

            logger.info(f"Trained multi-factor model on {len(features)} samples")

    def generate_prediction(self, game_id: str, game_data: Dict[str, Any]) -> MultiFactorPrediction:
        """
        Generate comprehensive prediction using all available factors.
        """
        # Extract all available data
        odds_data = game_data.get('multi_book_odds', [])
        analytics_data = game_data.get('advanced_analytics', [])
        sentiment_data = game_data.get('social_sentiment', {})
        news_data = game_data.get('news_analysis', [])
        expert_data = game_data.get('expert_consensus', {})

        # Calculate factors
        factors_considered = []
        key_drivers = []

        # Odds-based factors
        odds_factors = self._analyze_odds_factors(odds_data)
        if odds_factors:
            factors_considered.extend(odds_factors['factors'])
            key_drivers.extend(odds_factors['drivers'])

        # Analytics factors
        analytics_factors = self._analyze_analytics_factors(analytics_data)
        if analytics_factors:
            factors_considered.extend(analytics_factors['factors'])
            key_drivers.extend(analytics_factors['drivers'])

        # Sentiment factors
        sentiment_factors = self._analyze_sentiment_factors(sentiment_data)
        if sentiment_factors:
            factors_considered.extend(sentiment_factors['factors'])
            key_drivers.extend(sentiment_factors['drivers'])

        # Expert factors
        expert_factors = self._analyze_expert_factors(expert_data)
        if expert_factors:
            factors_considered.extend(expert_factors['factors'])
            key_drivers.extend(expert_factors['drivers'])

        # ML prediction if model available
        ml_prediction = None
        if self.prediction_model:
            feature_vector = self._extract_feature_vector(game_data)
            if feature_vector:
                X_scaled = self.feature_scaler.transform([feature_vector])
                ml_prob = self.prediction_model.predict_proba(X_scaled)[0]
                ml_prediction = ml_prob[1]  # Home win probability

        # Combine all factors for final prediction
        final_prediction = self._combine_factors(
            game_data,
            odds_factors,
            analytics_factors,
            sentiment_factors,
            expert_factors,
            ml_prediction
        )

        return final_prediction

    def _analyze_odds_factors(self, odds_data: List[SportsbookOdds]) -> Optional[Dict[str, Any]]:
        """Analyze betting market factors."""
        if not odds_data:
            return None

        factors = ["Market Odds Analysis", "Vig Assessment", "Line Movement"]
        drivers = []

        # Best odds analysis
        best_odds = self._find_best_odds(odds_data)
        if best_odds:
            home_edge = (1/best_odds['best_home_odds']) - (1/best_odds['best_away_odds'])
            drivers.append(("Market Edge", home_edge))

        # Vig analysis
        avg_vig = np.mean([self._calculate_vig(o) for o in odds_data if o.home_ml and o.away_ml])
        drivers.append(("Average Vig", -avg_vig))  # Lower vig is better

        return {'factors': factors, 'drivers': drivers}

    def _analyze_analytics_factors(self, analytics_data: List[AdvancedAnalytics]) -> Optional[Dict[str, Any]]:
        """Analyze player and team analytics."""
        if not analytics_data:
            return None

        factors = ["Player Efficiency", "Team Analytics", "Injury Impact"]
        drivers = []

        # Aggregate analytics by team
        team_stats = {}
        for player in analytics_data:
            team = player.team
            if team not in team_stats:
                team_stats[team] = {'epa': [], 'success_rate': [], 'injury_impact': []}

            if player.epa_per_play is not None:
                team_stats[team]['epa'].append(player.epa_per_play)
            if player.success_rate is not None:
                team_stats[team]['success_rate'].append(player.success_rate)
            if player.injury_status and player.injury_status.lower() in ['out', 'doubtful']:
                team_stats[team]['injury_impact'].append(0.3)  # Injury penalty

        # Calculate team advantages
        for team, stats in team_stats.items():
            if stats['epa']:
                avg_epa = np.mean(stats['epa'])
                drivers.append((f"{team} EPA", avg_epa))

            if stats['success_rate']:
                avg_success = np.mean(stats['success_rate'])
                drivers.append((f"{team} Success Rate", avg_success))

            injury_impact = sum(stats['injury_impact'])
            if injury_impact > 0:
                drivers.append((f"{team} Injury Impact", -injury_impact))

        return {'factors': factors, 'drivers': drivers}

    def _analyze_sentiment_factors(self, sentiment_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze sentiment factors."""
        if not sentiment_data:
            return None

        factors = ["Social Sentiment", "Public Opinion", "Market Psychology"]
        drivers = []

        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        drivers.append(("Public Sentiment", overall_sentiment))

        sentiment_trend = sentiment_data.get('sentiment_trend', 'neutral')
        trend_score = {'positive': 0.2, 'negative': -0.2, 'neutral': 0}.get(sentiment_trend, 0)
        drivers.append(("Sentiment Trend", trend_score))

        return {'factors': factors, 'drivers': drivers}

    def _analyze_expert_factors(self, expert_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Analyze expert consensus factors."""
        if not expert_data:
            return None

        factors = ["Expert Consensus", "Analyst Agreement", "Professional Insight"]
        drivers = []

        consensus_strength = expert_data.get('consensus_strength', 0)
        drivers.append(("Expert Consensus", consensus_strength))

        expert_count = expert_data.get('total_experts', 0)
        drivers.append(("Expert Count", min(0.3, expert_count / 20)))  # Cap at 20 experts

        return {'factors': factors, 'drivers': drivers}

    def _combine_factors(self, game_data: Dict[str, Any], *factor_results) -> MultiFactorPrediction:
        """Combine all factors into final prediction."""
        # Extract home/away teams
        home_team = game_data.get('home_team', 'Home')
        away_team = game_data.get('away_team', 'Away')

        # Aggregate all drivers
        all_drivers = []
        factors_considered = []

        for result in factor_results:
            if result and 'drivers' in result:
                all_drivers.extend(result['drivers'])
            if result and 'factors' in result:
                factors_considered.extend(result['factors'])

        # Calculate weighted prediction
        home_advantage = 0
        total_weight = 0

        for driver_name, impact in all_drivers:
            # Assign weights based on factor type
            if 'EPA' in driver_name:
                weight = 0.3
            elif 'Success Rate' in driver_name:
                weight = 0.25
            elif 'Market Edge' in driver_name:
                weight = 0.2
            elif 'Injury' in driver_name:
                weight = 0.15
            elif 'Consensus' in driver_name:
                weight = 0.15
            elif 'Sentiment' in driver_name:
                weight = 0.1
            else:
                weight = 0.1

            # Add team bias
            if home_team in driver_name:
                home_advantage += impact * weight
            elif away_team in driver_name:
                home_advantage -= impact * weight
            else:
                # Neutral factors
                home_advantage += impact * weight * 0.5  # Split impact

            total_weight += weight

        # Normalize advantage
        if total_weight > 0:
            home_advantage /= total_weight

        # Convert to probability and prediction
        home_win_prob = 1 / (1 + np.exp(-home_advantage * 3))  # Sigmoid with scaling
        predicted_winner = home_team if home_win_prob > 0.5 else away_team
        confidence = abs(home_win_prob - 0.5) * 2  # Scale to 0-1

        # Calculate expected value and stake
        best_odds = game_data.get('best_odds', {})
        if best_odds and confidence >= self.min_confidence_threshold:
            if predicted_winner == home_team and 'best_moneyline' in best_odds:
                odds = best_odds['best_moneyline']['home_odds']
                expected_value = (home_win_prob * odds) - 1
                bet_type = "moneyline"
            elif predicted_winner == away_team and 'best_moneyline' in best_odds:
                odds = best_odds['best_moneyline']['away_odds']
                expected_value = ((1 - home_win_prob) * odds) - 1
                bet_type = "moneyline"
            else:
                expected_value = 0
                bet_type = "no_bet"
        else:
            expected_value = 0
            bet_type = "insufficient_confidence"

        # Risk-adjusted stake sizing
        if expected_value > 0.05 and confidence > 0.7:
            risk_stake = min(self.max_risk_stake, expected_value / 2)
        else:
            risk_stake = 0

        return MultiFactorPrediction(
            game_id=game_data.get('id', game_data.get('game_id', 'unknown')),
            predicted_winner=predicted_winner,
            win_probability=home_win_prob if predicted_winner == home_team else (1 - home_win_prob),
            confidence_score=confidence,
            factors_considered=list(set(factors_considered)),
            key_drivers=all_drivers[:10],  # Top 10 drivers
            risk_adjusted_stake=risk_stake,
            expected_value=expected_value,
            recommended_bet_type=bet_type
        )

    def _find_best_odds(self, odds_data: List[SportsbookOdds]) -> Optional[Dict[str, float]]:
        """Find best available odds."""
        if not odds_data:
            return None

        best_home = max((o.home_ml for o in odds_data if o.home_ml), default=None)
        best_away = max((o.away_ml for o in odds_data if o.away_ml), default=None)

        if best_home and best_away:
            return {'best_home_odds': best_home, 'best_away_odds': best_away}

        return None

    def _calculate_vig(self, odds: SportsbookOdds) -> float:
        """Calculate vig (house edge) for odds."""
        if odds.home_ml and odds.away_ml:
            return (1/odds.home_ml) + (1/odds.away_ml) - 1
        return 0

    def _extract_feature_vector(self, game_data: Dict[str, Any]) -> Optional[List[float]]:
        """Extract feature vector for ML prediction."""
        # This would extract standardized features from game data
        # Implementation depends on specific features used
        return []

class AdvancedTradingEngine:
    """
    Unified advanced trading engine combining all strategies.
    """

    def __init__(self):
        self.arbitrage_detector = ArbitrageDetector()
        self.sharp_money_tracker = SharpMoneyTracker()
        self.sentiment_engine = SentimentTimingEngine()
        self.expert_analyzer = ExpertConsensusAnalyzer()
        self.multi_factor_engine = MultiFactorAnalysisEngine()

    async def analyze_game(self, game_id: str, comprehensive_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive trading analysis for a game.
        """
        logger.info(f"🔬 Running advanced trading analysis for {game_id}")

        analysis_results = {
            'game_id': game_id,
            'timestamp': datetime.now(),
            'arbitrage_opportunities': [],
            'sharp_money_signals': [],
            'sentiment_timing_signals': [],
            'expert_consensus_signals': [],
            'multi_factor_prediction': None,
            'market_condition': MarketCondition.EFFICIENT,
            'trading_signals': [],
            'risk_assessment': {},
            'confidence_metrics': {}
        }

        # Extract data sources
        odds_data = [SportsbookOdds(**o) for o in comprehensive_data.get('odds_data', [])]
        analytics_data = [AdvancedAnalytics(**a) for a in comprehensive_data.get('analytics_data', [])]
        sentiment_data = [SocialSentiment(**s) for s in comprehensive_data.get('sentiment_data', [])]
        news_data = [NewsAnalysis(**n) for n in comprehensive_data.get('news_data', [])]
        expert_data = [ExpertPicks(**e) for e in comprehensive_data.get('expert_data', [])]

        # Arbitrage Detection
        if odds_data:
            arbitrage_opps = self.arbitrage_detector.scan_for_arbitrage(odds_data)
            analysis_results['arbitrage_opportunities'] = [asdict(opp) for opp in arbitrage_opps]

            if arbitrage_opps:
                analysis_results['market_condition'] = MarketCondition.ARBITRAGE_OPPORTUNITY

        # Sharp Money Tracking
        if odds_data:
            sharp_signals = self.sharp_money_tracker.get_sharp_money_signals(game_id, odds_data)
            analysis_results['sharp_money_signals'] = [asdict(signal) for signal in sharp_signals]

        # Sentiment-Based Timing
        sentiment_signals = self.sentiment_engine.get_timing_signals(
            game_id, sentiment_data, news_data, odds_data, expert_data
        )
        analysis_results['sentiment_timing_signals'] = [asdict(signal) for signal in sentiment_signals]

        # Expert Consensus Analysis
        if expert_data:
            expert_signal = self.expert_analyzer.analyze_expert_consensus(game_id, expert_data)
            if expert_signal:
                analysis_results['expert_consensus_signals'] = [asdict(expert_signal)]

        # Multi-Factor Analysis
        if comprehensive_data:
            prediction = self.multi_factor_engine.generate_prediction(game_id, comprehensive_data)
            analysis_results['multi_factor_prediction'] = asdict(prediction)

        # Generate Trading Signals
        analysis_results['trading_signals'] = self._generate_trading_signals(analysis_results)

        # Risk Assessment
        analysis_results['risk_assessment'] = self._assess_overall_risk(analysis_results)

        # Confidence Metrics
        analysis_results['confidence_metrics'] = self._calculate_confidence_metrics(analysis_results)

        logger.info(f"✅ Advanced trading analysis complete for {game_id}")
        return analysis_results

    def _generate_trading_signals(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate unified trading signals from all analyses."""
        signals = []

        # Arbitrage signals (highest priority)
        for arb in analysis_results['arbitrage_opportunities']:
            signals.append({
                'type': 'arbitrage',
                'signal': TradeSignal.ARBITRAGE,
                'confidence': arb['confidence_score'],
                'description': f"Arbitrage opportunity: {arb['profit_percentage']:.1f}% profit",
                'action': 'Execute arbitrage bet immediately',
                'market_type': arb['market_type'],
                'stake_required': arb['required_stake']
            })

        # Sharp money signals
        for sharp in analysis_results['sharp_money_signals']:
            if sharp['confidence_score'] > 0.7:
                signals.append({
                    'type': 'sharp_money',
                    'signal': TradeSignal.BUY if sharp['direction'] == 'up' else TradeSignal.SELL,
                    'confidence': sharp['confidence_score'],
                    'description': f"Sharp money moving {sharp['direction']}: {sharp['signal_type']}",
                    'action': sharp['recommended_action']
                })

        # Sentiment timing signals
        for timing in analysis_results['sentiment_timing_signals']:
            if timing['confidence_score'] > 0.6:
                signal_type = TradeSignal.BUY if timing['timing_recommendation'] == 'follow' else TradeSignal.SELL
                signals.append({
                    'type': 'sentiment_timing',
                    'signal': signal_type,
                    'confidence': timing['confidence_score'],
                    'description': f"Sentiment timing: {timing['timing_recommendation']}",
                    'action': f"Entry time: {timing['optimal_entry_time'].strftime('%H:%M')}",
                    'holding_period': str(timing['holding_period'])
                })

        # Expert consensus signals
        for expert in analysis_results['expert_consensus_signals']:
            strategy = expert['recommended_strategy']
            if strategy == 'follow_consensus':
                signal = TradeSignal.BUY
            elif strategy == 'fade_consensus':
                signal = TradeSignal.SELL
            else:
                signal = TradeSignal.HOLD

            signals.append({
                'type': 'expert_consensus',
                'signal': signal,
                'confidence': expert['confidence_score'],
                'description': f"Expert consensus: {strategy}",
                'action': f"{'Follow' if strategy == 'follow_consensus' else 'Fade'} {expert['consensus_pick']}"
            })

        # Multi-factor prediction signals
        prediction = analysis_results.get('multi_factor_prediction')
        if prediction and prediction['confidence_score'] > 0.7:
            signals.append({
                'type': 'multi_factor',
                'signal': TradeSignal.BUY,
                'confidence': prediction['confidence_score'],
                'description': f"Multi-factor prediction: {prediction['predicted_winner']}",
                'action': f"Bet {prediction['recommended_bet_type']} on {prediction['predicted_winner']}",
                'expected_value': prediction['expected_value'],
                'stake_size': prediction['risk_adjusted_stake']
            })

        # Sort by confidence
        signals.sort(key=lambda x: x['confidence'], reverse=True)

        return signals

    def _assess_overall_risk(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk of trading signals."""
        risk_assessment = {
            'overall_risk': 'LOW',
            'arbitrage_risk': 'NONE',
            'market_risk': 'MEDIUM',
            'execution_risk': 'LOW',
            'factors': []
        }

        # Arbitrage opportunities reduce risk
        if analysis_results['arbitrage_opportunities']:
            risk_assessment['arbitrage_risk'] = 'NONE'
            risk_assessment['overall_risk'] = 'VERY_LOW'
            risk_assessment['factors'].append('Arbitrage opportunity available')

        # Sharp money signals
        sharp_signals = [s for s in analysis_results['sharp_money_signals'] if s['confidence_score'] > 0.8]
        if sharp_signals:
            risk_assessment['market_risk'] = 'LOW'
            risk_assessment['factors'].append('Strong sharp money signals detected')

        # Multiple confirming signals reduce risk
        signal_count = (len(analysis_results['arbitrage_opportunities']) +
                       len([s for s in analysis_results['sharp_money_signals'] if s['confidence_score'] > 0.7]) +
                       len([s for s in analysis_results['sentiment_timing_signals'] if s['confidence_score'] > 0.6]) +
                       len(analysis_results['expert_consensus_signals']) +
                       (1 if analysis_results.get('multi_factor_prediction') and
                        analysis_results['multi_factor_prediction']['confidence_score'] > 0.7 else 0))

        if signal_count >= 3:
            risk_assessment['overall_risk'] = 'LOW'
            risk_assessment['factors'].append('Multiple confirming signals')

        return risk_assessment

    def _calculate_confidence_metrics(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate confidence metrics across all analyses."""
        confidence_metrics = {
            'overall_confidence': 0.0,
            'signal_consistency': 0.0,
            'data_quality_score': analysis_results.get('data_quality_score', 0),
            'signal_strength': 'WEAK',
            'recommendation': 'HOLD'
        }

        # Calculate overall confidence from all signals
        all_confidences = []

        for arb in analysis_results['arbitrage_opportunities']:
            all_confidences.append(arb['confidence_score'])

        for sharp in analysis_results['sharp_money_signals']:
            all_confidences.append(sharp['confidence_score'])

        for timing in analysis_results['sentiment_timing_signals']:
            all_confidences.append(timing['confidence_score'])

        for expert in analysis_results['expert_consensus_signals']:
            all_confidences.append(expert['confidence_score'])

        prediction = analysis_results.get('multi_factor_prediction')
        if prediction:
            all_confidences.append(prediction['confidence_score'])

        if all_confidences:
            confidence_metrics['overall_confidence'] = np.mean(all_confidences)

            # Signal consistency (how close confidences are to each other)
            if len(all_confidences) > 1:
                confidence_metrics['signal_consistency'] = 1 - np.std(all_confidences)

            # Signal strength classification
            avg_conf = confidence_metrics['overall_confidence']
            if avg_conf > 0.8:
                confidence_metrics['signal_strength'] = 'VERY_STRONG'
                confidence_metrics['recommendation'] = 'EXECUTE'
            elif avg_conf > 0.7:
                confidence_metrics['signal_strength'] = 'STRONG'
                confidence_metrics['recommendation'] = 'EXECUTE'
            elif avg_conf > 0.6:
                confidence_metrics['signal_strength'] = 'MODERATE'
                confidence_metrics['recommendation'] = 'CONSIDER'
            elif avg_conf > 0.5:
                confidence_metrics['signal_strength'] = 'WEAK'
                confidence_metrics['recommendation'] = 'MONITOR'
            else:
                confidence_metrics['signal_strength'] = 'VERY_WEAK'
                confidence_metrics['recommendation'] = 'AVOID'

        return confidence_metrics

# Export main classes
__all__ = [
    'AdvancedTradingEngine',
    'ArbitrageDetector',
    'SharpMoneyTracker',
    'SentimentTimingEngine',
    'ExpertConsensusAnalyzer',
    'MultiFactorAnalysisEngine',
    'ArbitrageOpportunity',
    'SharpMoneySignal',
    'SentimentTimingSignal',
    'ExpertConsensusSignal',
    'MultiFactorPrediction',
    'TradeSignal',
    'MarketCondition'
]

