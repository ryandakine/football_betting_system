#!/usr/bin/env python3
"""
Unified WNBA Analyzer - Main System
===================================

Combines ALL features for WNBA betting:
- CloudGPU AI ensemble
- 5-AI Council with specialized agents
- Game prioritization
- Social & weather analysis
- Parlay optimization
- Real-time monitoring
- Performance tracking
- Backtesting
"""

import asyncio
import logging
import json
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

# Core dependencies
from unified_nfl_intelligence_system import UnifiedNFLIntelligenceSystem
from football_odds_fetcher import FootballOddsFetcher
from api_config import get_api_keys
from performance_tracker import PerformanceTracker

# CloudGPU AI (HuggingFace)
try:
    from huggingface_cloud_gpu import CloudGPUAIEnsemble, CloudGPUConfig
    HAS_CLOUDGPU = True
except ImportError:
    HAS_CLOUDGPU = False
    logging.warning("CloudGPU AI not available - using standard analysis")

# Local modules
from crew_prediction_integration import CrewPredictionEngine
from meta_learner import MetaLearner

from wnba_system.game_prioritization import GamePrioritizer
from wnba_system.social_weather_analyzer import CombinedSocialWeatherAnalyzer
from wnba_system.parlay_optimizer import ParlayOptimizer
from wnba_system.realtime_monitor import RealTimeMonitor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UnifiedWNBAAnalyzer:
    """
    Master analyzer combining all WNBA betting features.
    """

    def __init__(self, bankroll: float = 50000.0):
        self.bankroll = bankroll

        # Core analysis system
        self.system = UnifiedNFLIntelligenceSystem(bankroll)

        # Initialize Odds API for WNBA
        api_keys = get_api_keys()
        odds_api_key = api_keys.get('odds_api')
        if not odds_api_key:
            raise ValueError("Odds API key not found")
        self.odds_fetcher = FootballOddsFetcher(
            api_key=odds_api_key,
            sport_key='basketball_wnba',  # WNBA
            markets=['h2h','spreads','totals']
        )

        # Performance tracking
        self.bet_tracker = PerformanceTracker(initial_bankroll=bankroll)

        # Specialized analyzers
        self.prioritizer = GamePrioritizer()
        self.social_weather = CombinedSocialWeatherAnalyzer()
        self.parlay_optimizer = ParlayOptimizer()
        self.realtime_monitor = RealTimeMonitor()

        # Optional advanced modules
        try:
            self.crew_engine = CrewPredictionEngine()
        except Exception as exc:
            logger.warning(f"⚠️ Crew prediction engine unavailable: {exc}")
            self.crew_engine = None

        try:
            self.meta_learner = MetaLearner()
            self.meta_learner.train_if_needed()
        except Exception as exc:
            logger.warning(f"⚠️ Meta learner disabled: {exc}")
            self.meta_learner = None

        # AI Council agents (integrated inline for simplicity)
        self.ai_council = FiveAICouncil()

        # CloudGPU ensemble (if available)
        if HAS_CLOUDGPU:
            self.cloudgpu_ensemble = CloudGPUAIEnsemble()
            logger.info("✅ CloudGPU AI Ensemble enabled")
        else:
            self.cloudgpu_ensemble = None

        logger.info("🏀 Unified WNBA Analyzer initialized")

    async def run_complete_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline."""
        print("\n🏀 UNIFIED WNBA ANALYSIS")
        print("=" * 70)

        # Step 1: Fetch real games
        print("📊 Fetching games from Odds API...")
        all_games = await self.fetch_all_games()
        print(f"   ✅ Fetched {len(all_games)} games")

        # Step 2: Prioritize games
        print("🎯 Prioritizing games...")
        prioritized_games = self.prioritizer.optimize_processing_order(all_games)
        print(f"   ✅ Prioritized by edge potential")

        # Step 3: Analyze each game with ALL systems
        print("🤖 Running comprehensive analysis...")
        results = []
        high_edge_games = []

        for i, game in enumerate(prioritized_games[:15], 1):  # WNBA has fewer games
            print(f"   [{i}/15] {game['away_team']} @ {game['home_team']}")

            # Enhanced analysis with all features
            result = await self.analyze_single_game(game)
            results.append(result)

            if result.get('total_edge', 0) > 0.08:
                high_edge_games.append(result)

        # Step 4: Generate parlays
        print("🎰 Optimizing parlays...")
        parlays = self.parlay_optimizer.optimize_parlays(results, self.bankroll)
        print(f"   ✅ Generated {len(parlays)} optimized parlays")

        # Step 5: Start real-time monitoring (optional)
        # await self.realtime_monitor.start_monitoring(high_edge_games, max_cycles=1)

        return {
            'total_games': len(all_games),
            'games_analyzed': len(results),
            'high_edge_games': high_edge_games,
            'parlays': parlays[:5],  # Top 5
            'summary': self.generate_summary(results, high_edge_games, parlays)
        }

    async def fetch_all_games(self) -> List[Dict[str, Any]]:
        """Fetch real games from Odds API."""
        try:
            async with self.odds_fetcher as fetcher:
                odds = await fetcher.get_all_odds_with_props()

            games = []
            spreads = {b.game_id: b for b in odds.spread_bets}
            totals = {b.game_id: b for b in odds.total_bets}

            for g in odds.games:
                spread_val = spreads.get(g.game_id).home_spread if spreads.get(g.game_id) else 0.0
                total_val = totals.get(g.game_id).total_points if totals.get(g.game_id) else 0.0

                games.append({
                    'game_id': g.game_id,
                    'home_team': g.home_team,
                    'away_team': g.away_team,
                    'spread': spread_val,
                    'total': total_val,
                    'league': 'WNBA',
                    'game_type': 'WNBA',
                    'edge_value': 0.0,  # Will calculate
                    'confidence': 0.0,  # Will calculate
                })

            return games
        except Exception as e:
            logger.error(f"Error fetching games: {e}")
            return []

    async def analyze_single_game(self, game: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze single game with ALL systems."""

        game_id = game.get('game_id', 'unknown')

        # 1. Base unified analysis (already includes agent influence + edge)
        base_result = await self.system.run_unified_analysis(game)
        base_edge = base_result.get('total_edge', 0.0)
        agent_metrics = base_result.get('agent_metrics', {}) or {}

        # 2. Social & weather enhancement
        social_weather_context = await self.social_weather.analyze_game_context(game)
        social_edge = social_weather_context.get('edge_adjustment', 0.0) if social_weather_context else 0.0

        # 3. Optional crew/official context
        crew_context = self._compute_crew_context(game, base_result)
        crew_edge = crew_context.get('crew_adjustment', 0.0)

        # 4. CloudGPU AI (if available)
        cloudgpu_insight = None
        if self.cloudgpu_ensemble:
            try:
                cloudgpu_insight = await self.cloudgpu_ensemble.analyze_game(game)
            except Exception as exc:
                logger.debug(f"CloudGPU analysis unavailable for {game_id}: {exc}")

        # 5. WNBA 5-AI council consensus with full context
        council_context = CouncilInput(
            game=game,
            base_result=base_result,
            social_weather=social_weather_context,
            agent_metrics=agent_metrics,
            crew_context=crew_context,
            cloudgpu_insight=cloudgpu_insight,
        )
        council_recommendation = await self.ai_council.get_consensus(council_context)

        # 6. Meta-learner calibration
        meta_result = self._apply_meta_learning(
            game,
            base_result,
            council_recommendation,
            social_weather_context,
            cloudgpu_insight
        )
        meta_probability = meta_result.get('meta_probability')

        # Combine all insights
        combined_edge = base_edge + social_edge + crew_edge
        base_confidence = base_result.get('unified_recommendation', {}).get('combined_confidence', 0.5)
        council_confidence = council_recommendation.get('consensus_confidence', 0.5)
        combined_confidence = (base_confidence * 0.55) + (council_confidence * 0.35)
        if meta_probability is not None:
            combined_confidence = combined_confidence * 0.6 + meta_probability * 0.4

        penalty_bias = agent_metrics.get('penalty_bias_home')
        if isinstance(penalty_bias, (int, float)) and penalty_bias not in (0, 1):
            combined_edge *= float(penalty_bias)

        final_action = base_result.get('unified_recommendation', {}).get('action', 'PASS')
        council_action = council_recommendation.get('consensus_action', 'PASS')
        if council_action and council_action not in {'PASS', 'MONITOR'}:
            final_action = f"{final_action} + {council_action}" if final_action != 'PASS' else council_action

        # Aggregate strategic signals
        signals = set(council_recommendation.get('signals', []))
        signals.update(agent_metrics.get('rules_triggered', []) or [])
        signals.update(agent_metrics.get('strategy_signals', []) or [])
        if crew_context.get('signals'):
            signals.update(crew_context['signals'])
        if social_weather_context and social_weather_context.get('weather_impact', {}).get('recommendations'):
            signals.update(social_weather_context['weather_impact']['recommendations'])

        bet_tracking_id = self._track_recommendation(game, final_action, combined_edge, combined_confidence)

        enhanced_recommendation = {
            'final_action': final_action,
            'base_action': base_result.get('unified_recommendation', {}).get('action', 'PASS'),
            'council_action': council_action,
            'confidence': combined_confidence,
            'edge': combined_edge,
            'meta_probability': meta_probability,
            'signals': sorted(signals),
            'bet_tracking_id': bet_tracking_id,
        }

        return {
            **game,
            'total_edge': combined_edge,
            'base_total_edge': base_edge,
            'confidence': combined_confidence,
            'baseline_recommendation': base_result.get('unified_recommendation'),
            'unified_recommendation': enhanced_recommendation,
            'ai_council': council_recommendation,
            'social_weather': social_weather_context,
            'cloudgpu_insight': cloudgpu_insight,
            'agent_metrics': agent_metrics,
            'crew_context': crew_context,
            'meta_learning': meta_result,
        }

    def generate_summary(self, results: List[Dict], high_edge: List[Dict], parlays: List) -> Dict[str, Any]:
        """Generate analysis summary."""
        avg_edge = sum(r.get('total_edge', 0) for r in results) / len(results) if results else 0

        return {
            'games_analyzed': len(results),
            'high_edge_opportunities': len(high_edge),
            'average_edge': avg_edge,
            'parlays_generated': len(parlays),
            'best_single_game': max(results, key=lambda x: x.get('total_edge', 0)) if results else None,
        }

    def _compute_crew_context(self, game: Dict[str, Any], base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate officiating bias impact if crew data/model available."""
        if not self.crew_engine:
            return {'crew_adjustment': 0.0, 'notes': 'Crew prediction engine unavailable'}

        crew_name = (
            game.get('referee')
            or game.get('crew')
            or game.get('referee_crew')
            or base_result.get('legacy_analysis', {}).get('referee_profile', {}).get('crew_name')
        )
        home_team = game.get('home_team')
        away_team = game.get('away_team')

        if not crew_name or not home_team or not away_team:
            return {'crew_adjustment': 0.0, 'notes': 'No crew data supplied'}

        try:
            week = int(game.get('week', 0) or 0)
            year = int(game.get('season', datetime.utcnow().year) or datetime.utcnow().year)
            home_bias = self.crew_engine.predict_margin(crew_name, home_team, week=week, year=year)
            away_bias = self.crew_engine.predict_margin(crew_name, away_team, week=week, year=year)
            net_bias = home_bias - away_bias
            adjustment = net_bias * 0.01  # convert predicted margin to edge delta
            signals = []
            if abs(net_bias) >= 4:
                direction = 'HOME' if net_bias > 0 else 'AWAY'
                signals.append(f'CREW_FAVOR_{direction}')
            return {
                'crew_name': crew_name,
                'home_margin': home_bias,
                'away_margin': away_bias,
                'net_margin': net_bias,
                'crew_adjustment': adjustment,
                'signals': signals,
            }
        except Exception as exc:
            logger.debug(f"Crew adjustment unavailable for {crew_name}: {exc}")
            return {'crew_adjustment': 0.0, 'notes': 'Crew model error'}

    def _apply_meta_learning(
        self,
        game: Dict[str, Any],
        base_result: Dict[str, Any],
        council: Dict[str, Any],
        social_weather: Optional[Dict[str, Any]],
        cloudgpu: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Blend meta-learner probability if available."""
        if not self.meta_learner:
            return {'meta_probability': None, 'features': {}}

        base_rec = base_result.get('unified_recommendation', {}) or {}
        features = {
            'base_confidence': float(base_rec.get('combined_confidence', 0.5)),
            'base_edge': float(base_result.get('total_edge', 0.0)),
            'council_confidence': float(council.get('consensus_confidence', 0.5)),
            'council_action_is_bet': 1.0 if (council.get('consensus_action') or '').startswith('BET') else 0.0,
            'cloud_confidence': float((cloudgpu or {}).get('confidence', 0.0) or 0.0),
        }

        if social_weather:
            features['social_weather_score'] = float(social_weather.get('combined_impact_score', 0.5))
            home_sent = social_weather.get('home_sentiment', {}).get('sentiment_score', 0.5)
            away_sent = social_weather.get('away_sentiment', {}).get('sentiment_score', 0.5)
            features['sentiment_gap'] = float(home_sent) - float(away_sent)
            features['weather_edge_adjustment'] = float(social_weather.get('edge_adjustment', 0.0))
        else:
            features['social_weather_score'] = 0.5
            features['sentiment_gap'] = 0.0
            features['weather_edge_adjustment'] = 0.0

        meta_probability = None
        try:
            meta_probability = self.meta_learner.predict(features)
        except Exception as exc:
            logger.debug(f"Meta learner unable to score {game.get('game_id')}: {exc}")

        try:
            self.meta_learner.log_prediction(
                game_id=game.get('game_id', 'unknown'),
                prediction_time=datetime.utcnow().isoformat(),
                features=features,
                base_probability=features['base_confidence'],
                ensemble_probability=features['council_confidence'],
                meta_probability=meta_probability,
                source='wnba_unified_analyzer',
            )
        except Exception as exc:
            logger.debug(f"Meta learner logging failed: {exc}")

        return {
            'meta_probability': meta_probability,
            'features': features,
        }

    def _track_recommendation(self, game: Dict[str, Any], action: str, edge: float, confidence: float) -> Optional[str]:
        """Log recommendation into performance tracker and return bet id."""
        normalized_action = (action or '').upper()
        if normalized_action in {'', 'PASS', 'MONITOR'}:
            return None
        if edge <= 0 or confidence <= 0:
            return None

        try:
            stake = max(0.0, min(self.bankroll * 0.025, self.bankroll * abs(edge)))
            if stake == 0:
                return None
            bet_data = {
                'game_id': game.get('game_id', 'unknown'),
                'sport_type': 'wnba',
                'bet_type': 'spread',
                'selection': f"{game.get('away_team', 'AWAY')} @ {game.get('home_team', 'HOME')}",
                'odds': -110,
                'stake': stake,
                'expected_value': edge,
                'confidence': confidence,
            }
            return self.bet_tracker.track_bet(bet_data)
        except Exception as exc:
            logger.debug(f"Performance tracker logging skipped: {exc}")
            return None


@dataclass
class CouncilInput:
    """Context bundle passed into 5-AI council agents."""
    game: Dict[str, Any]
    base_result: Dict[str, Any]
    social_weather: Optional[Dict[str, Any]] = None
    agent_metrics: Optional[Dict[str, Any]] = None
    crew_context: Optional[Dict[str, Any]] = None
    cloudgpu_insight: Optional[Dict[str, Any]] = None


class FiveAICouncil:
    """5-AI Council with specialized agents for WNBA."""

    def __init__(self):
        self.agents = {
            'game_context': GameContextAgent(),
            'line_movement': LineMovementAgent(),
            'weather_social': WeatherSocialAgent(),
            'team_expert': TeamExpertAgent(),
            'player_analysis': PlayerAnalysisAgent(),
        }

    async def get_consensus(self, context: CouncilInput) -> Dict[str, Any]:
        """Get consensus recommendation from all agents."""
        votes: Dict[str, Dict[str, Any]] = {}
        aggregated_signals: List[str] = []
        strategy_counts: Dict[str, int] = {}
        confidences: List[float] = []

        for name, agent in self.agents.items():
            response = await agent.analyze(context)
            votes[name] = response
            recommendation = response.get('recommendation', 'PASS')
            strategy_counts[recommendation] = strategy_counts.get(recommendation, 0) + 1
            confidences.append(response.get('confidence', 0.0))
            aggregated_signals.extend(response.get('signals', []))

        if strategy_counts:
            consensus_action = max(strategy_counts.items(), key=lambda x: (x[1], x[0]))[0]
        else:
            consensus_action = 'PASS'

        consensus_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confidence_range = (
            min(confidences) if confidences else 0.0,
            max(confidences) if confidences else 0.0,
        )

        return {
            'consensus_action': consensus_action,
            'consensus_confidence': consensus_confidence,
            'confidence_range': confidence_range,
            'agent_votes': votes,
            'signals': sorted(set(aggregated_signals)),
            'strategy_distribution': strategy_counts,
        }


# Lightweight agent implementations specific to WNBA
class GameContextAgent:
    async def analyze(self, context: CouncilInput) -> Dict[str, Any]:
        base_rec = context.base_result.get('unified_recommendation', {}) if context.base_result else {}
        edge = context.base_result.get('total_edge', 0.0) if context.base_result else 0.0
        cloud_confidence = (context.cloudgpu_insight or {}).get('confidence', 0.0) if context.cloudgpu_insight else 0.0

        recommendation = base_rec.get('action', 'PASS')
        confidence = float(base_rec.get('combined_confidence', 0.5))
        signals: List[str] = []

        if edge >= 0.12:
            signals.append('EDGE_ELITE')
            confidence = max(confidence, 0.72)
        elif edge <= 0.03:
            recommendation = 'PASS'

        if cloud_confidence:
            confidence = max(confidence, cloud_confidence)
            if cloud_confidence >= 0.7:
                signals.append('CLOUDGPU_ALIGNMENT')

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
        }

class LineMovementAgent:
    async def analyze(self, context: CouncilInput) -> Dict[str, Any]:
        realtime = context.base_result.get('realtime_analysis', {}) if context.base_result else {}
        market = realtime.get('market_intelligence', {}) if realtime else {}
        sharp_books = market.get('sharp_detection', 0)
        arbitrage = market.get('arbitrage_opportunities', 0)

        recommendation = 'PASS'
        confidence = 0.5
        signals: List[str] = []

        if sharp_books >= 2:  # WNBA has fewer books
            recommendation = 'FOLLOW_SHARP'
            confidence = 0.65
            signals.append('MARKET_SHARP_SPIKE')
        if arbitrage:
            signals.append('MARKET_ARBITRAGE_WINDOW')

        stream_confidence = realtime.get('stream_processing', {}).get('confidence')
        if stream_confidence:
            confidence = max(confidence, float(stream_confidence))

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
        }

class WeatherSocialAgent:
    async def analyze(self, context: CouncilInput) -> Dict[str, Any]:
        if not context.social_weather:
            return {'recommendation': 'PASS', 'confidence': 0.5, 'signals': []}

        combined = context.social_weather.get('combined_impact_score', 0.5)
        home_sent = context.social_weather.get('home_sentiment', {}).get('sentiment_score', 0.5)
        away_sent = context.social_weather.get('away_sentiment', {}).get('sentiment_score', 0.5)
        sentiment_gap = home_sent - away_sent

        recommendation = 'PASS'
        confidence = 0.5
        signals: List[str] = []

        # WNBA specific social sentiment is important
        if sentiment_gap >= 0.12:
            signals.append('HOME_FAN_SURGE')
            recommendation = 'BET_HOME'
            confidence = 0.60
        elif sentiment_gap <= -0.12:
            signals.append('AWAY_PUBLIC_MOMENTUM')
            recommendation = 'BET_AWAY'
            confidence = 0.60

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
        }

class TeamExpertAgent:
    async def analyze(self, context: CouncilInput) -> Dict[str, Any]:
        """WNBA team-specific analysis."""
        recommendation = 'PASS'
        confidence = 0.48
        signals: List[str] = []

        # Check for home/away patterns
        if context.base_result.get('total_edge', 0) >= 0.08:
            recommendation = 'BET_VALUE'
            confidence = 0.60
            signals.append('VALUE_OPPORTUNITY')

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
        }

class PlayerAnalysisAgent:
    async def analyze(self, context: CouncilInput) -> Dict[str, Any]:
        """WNBA player-specific analysis (injuries, rest, etc.)."""
        metrics = context.agent_metrics or {}

        recommendation = 'PASS'
        confidence = 0.5
        signals: List[str] = []

        # Check for injury/rest impacts
        scandal = float(metrics.get('scandal_score', 0.0) or 0.0)
        if scandal >= 0.6:
            recommendation = 'LIMIT_STAKE'
            confidence = 0.68
            signals.append('PLAYER_AVAILABILITY_CONCERN')

        return {
            'recommendation': recommendation,
            'confidence': confidence,
            'signals': signals,
        }


async def main():
    """Main entry point."""
    analyzer = UnifiedWNBAAnalyzer(bankroll=50000.0)
    results = await analyzer.run_complete_analysis()

    print("\n" + "=" * 70)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 70)
    print(json.dumps(results['summary'], indent=2))


if __name__ == "__main__":
    asyncio.run(main())
