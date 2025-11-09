#!/usr/bin/env python3
"""
Performance Tracker Agent - Tracks results of placed bets
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class PerformanceTrackerAgent:
    """Agent responsible for tracking bet performance"""

    def __init__(self, config):
        self.config = config
        self.bets_file = config.results_dir / "placed_bets.json"
        self.results_file = config.results_dir / "bet_results.json"

    async def check_recent_bets(self, days=7):
        """Check results of bets placed in last N days"""
        logger.info(f"Checking bets from last {days} days...")

        try:
            # Load placed bets
            placed_bets = self._load_placed_bets()

            if not placed_bets:
                logger.info("No placed bets found")
                return {
                    'completed_bets': 0,
                    'wins': 0,
                    'losses': 0,
                    'profit': 0,
                    'win_rate': 0
                }

            # Get game results
            completed_bets = []
            wins = 0
            losses = 0
            total_profit = 0

            for bet in placed_bets:
                if bet.get('result_checked'):
                    continue  # Already checked

                game_id = bet.get('game_id')
                if not game_id:
                    continue

                # Check if game is completed
                result = await self._check_game_result(game_id, bet)

                if result:
                    # Update bet with result
                    bet['result_checked'] = True
                    bet['won'] = result['won']
                    bet['actual_profit'] = result['profit']

                    completed_bets.append(bet)

                    if result['won']:
                        wins += 1
                    else:
                        losses += 1

                    total_profit += result['profit']

            # Save updated bets
            if completed_bets:
                self._save_bet_results(completed_bets)
                self._update_placed_bets(placed_bets)

            win_rate = wins / len(completed_bets) if completed_bets else 0

            return {
                'completed_bets': len(completed_bets),
                'wins': wins,
                'losses': losses,
                'profit': total_profit,
                'win_rate': win_rate
            }

        except Exception as e:
            logger.error(f"Result checking failed: {e}")
            return {
                'completed_bets': 0,
                'wins': 0,
                'losses': 0,
                'profit': 0,
                'win_rate': 0
            }

    def _load_placed_bets(self):
        """Load bets that have been placed"""
        if not self.bets_file.exists():
            return []

        with open(self.bets_file) as f:
            return json.load(f)

    def _update_placed_bets(self, bets):
        """Update placed bets file"""
        with open(self.bets_file, 'w') as f:
            json.dump(bets, f, indent=2, default=str)

    def _save_bet_results(self, completed_bets):
        """Save completed bet results"""
        # Load existing results
        results = []
        if self.results_file.exists():
            with open(self.results_file) as f:
                results = json.load(f)

        # Add new results
        results.extend(completed_bets)

        # Save
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Saved {len(completed_bets)} bet results")

    async def _check_game_result(self, game_id, bet):
        """Check result of a specific game"""
        try:
            # Load current season games
            season = bet.get('season', datetime.now().year)
            games_file = self.config.data_dir / f"ncaaf_{season}_games.json"

            if not games_file.exists():
                return None

            with open(games_file) as f:
                games = json.load(f)

            # Find the game
            game = next((g for g in games if str(g.get('id')) == str(game_id)), None)

            if not game:
                return None

            # Check if completed
            if not game.get('completed'):
                return None

            # Determine winner
            home_score = game.get('home_points', game.get('home_score', 0))
            away_score = game.get('away_points', game.get('away_score', 0))

            if home_score is None or away_score is None:
                return None

            actual_winner = 'home' if home_score > away_score else 'away'

            # Check if our pick was correct
            our_pick = bet.get('pick')
            won = (our_pick == actual_winner)

            # Calculate profit
            stake = bet.get('recommended_stake', 0)
            profit = stake * 0.909 if won else -stake

            return {
                'won': won,
                'profit': profit,
                'home_score': home_score,
                'away_score': away_score,
                'actual_winner': actual_winner
            }

        except Exception as e:
            logger.error(f"Failed to check game {game_id}: {e}")
            return None

    def log_bet_placement(self, pick, amount_bet):
        """Log when a bet is actually placed"""
        # Load placed bets
        placed_bets = self._load_placed_bets()

        # Add new bet
        bet_record = {
            **pick,
            'amount_bet': amount_bet,
            'placed_at': datetime.now().isoformat(),
            'result_checked': False
        }

        placed_bets.append(bet_record)

        # Save
        self._update_placed_bets(placed_bets)

        logger.info(f"Logged bet: {pick['predicted_winner']} ${amount_bet}")

    def get_performance_summary(self, days=30):
        """Get performance summary for last N days"""
        if not self.results_file.exists():
            return None

        with open(self.results_file) as f:
            results = json.load(f)

        # Filter by date
        cutoff = datetime.now() - timedelta(days=days)
        recent = [r for r in results if datetime.fromisoformat(r['placed_at']) > cutoff]

        if not recent:
            return None

        wins = sum(1 for r in recent if r.get('won'))
        total_profit = sum(r.get('actual_profit', 0) for r in recent)
        total_staked = sum(r.get('amount_bet', 0) for r in recent)

        return {
            'period_days': days,
            'total_bets': len(recent),
            'wins': wins,
            'losses': len(recent) - wins,
            'win_rate': wins / len(recent) if recent else 0,
            'total_profit': total_profit,
            'total_staked': total_staked,
            'roi': (total_profit / total_staked * 100) if total_staked > 0 else 0
        }
