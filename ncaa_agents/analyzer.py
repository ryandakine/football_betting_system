#!/usr/bin/env python3
"""
Analysis Agent - Runs backtests and evaluates system performance
"""

import json
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Agent responsible for analyzing system performance"""

    def __init__(self, config):
        self.config = config

    async def run_backtest(self, seasons, min_edge=0.03, min_confidence=0.60):
        """Run backtest using the improved backtester"""
        logger.info(f"Running backtest for seasons: {seasons}")

        try:
            # Use the improved backtester
            from backtest_ncaa_improved import ImprovedNCAABacktester

            backtester = ImprovedNCAABacktester(
                bankroll=self.config.bankroll,
                unit_size=100,
                min_edge=min_edge,
                min_confidence=min_confidence
            )

            bets = backtester.run_backtest(seasons=seasons)

            if not bets:
                return None

            # Calculate results
            wins = sum(1 for b in bets if b['won'])
            total_profit = sum(b['profit'] for b in bets)
            win_rate = wins / len(bets) if bets else 0
            roi = (total_profit / self.config.bankroll) * 100

            # Statistical test
            import numpy as np
            from scipy import stats
            profits = [b['profit'] for b in bets]
            t_stat, p_value = stats.ttest_1samp(profits, 0)

            return {
                'total_bets': len(bets),
                'wins': wins,
                'losses': len(bets) - wins,
                'win_rate': win_rate,
                'total_profit': total_profit,
                'roi': roi,
                't_stat': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05,
                'bets': bets
            }

        except Exception as e:
            logger.error(f"Backtest failed: {e}")
            return None

    def analyze_conference_performance(self, bets):
        """Analyze performance by conference"""
        from collections import defaultdict

        by_conference = defaultdict(lambda: {'bets': 0, 'wins': 0, 'profit': 0})

        for bet in bets:
            conf = bet.get('conference', 'Unknown')
            by_conference[conf]['bets'] += 1
            if bet['won']:
                by_conference[conf]['wins'] += 1
            by_conference[conf]['profit'] += bet['profit']

        # Calculate win rates and ROI
        results = []
        for conf, data in by_conference.items():
            if data['bets'] >= 5:  # Minimum sample size
                win_rate = data['wins'] / data['bets']
                roi = (data['profit'] / self.config.bankroll) * 100
                results.append({
                    'conference': conf,
                    'bets': data['bets'],
                    'wins': data['wins'],
                    'win_rate': win_rate,
                    'profit': data['profit'],
                    'roi': roi
                })

        # Sort by profit
        results.sort(key=lambda x: x['profit'], reverse=True)
        return results

    def generate_recommendations(self, backtest_results):
        """Generate actionable recommendations based on backtest"""
        recommendations = []

        roi = backtest_results.get('roi', 0)
        win_rate = backtest_results.get('win_rate', 0)
        p_value = backtest_results.get('p_value', 1)

        # ROI analysis
        if roi > 15:
            recommendations.append({
                'type': 'action',
                'priority': 'high',
                'message': 'System is highly profitable! Consider live betting with small units.'
            })
        elif roi > 5:
            recommendations.append({
                'type': 'action',
                'priority': 'medium',
                'message': 'System shows promise. Collect more data before live betting.'
            })
        elif roi < 0:
            recommendations.append({
                'type': 'warning',
                'priority': 'high',
                'message': 'System is losing money. DO NOT use for live betting.'
            })

        # Statistical significance
        if p_value >= 0.05:
            recommendations.append({
                'type': 'warning',
                'priority': 'high',
                'message': f'Results not statistically significant (p={p_value:.3f}). Need more data.'
            })

        # Win rate analysis
        if win_rate < 0.524:
            recommendations.append({
                'type': 'optimization',
                'priority': 'high',
                'message': 'Win rate below break-even. Increase min_confidence threshold.'
            })

        return recommendations
