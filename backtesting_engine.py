"""
Backtesting Engine - Test betting strategies against historical data
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import logging
from dataclasses import dataclass, asdict
import statistics

logger = logging.getLogger(__name__)

@dataclass
class HistoricalGame:
    """Represents a completed historical game"""
    game_id: str
    home_team: str
    away_team: str
    game_date: str
    final_score_home: int
    final_score_away: int
    winner: str
    spread_result: str  # home_cover, away_cover, push
    total_result: str   # over, under, push
    odds_moneyline_home: float
    odds_moneyline_away: float
    odds_spread_home: float
    odds_spread_away: float
    spread_line: float
    odds_total_over: float
    odds_total_under: float
    total_line: float

@dataclass
class BacktestBet:
    """A bet placed during backtesting"""
    game_id: str
    bet_type: str  # moneyline, spread, total
    selection: str  # team name or over/under
    odds: float
    stake: float
    result: str  # win, loss, push
    payout: float
    timestamp: str

@dataclass
class BacktestResult:
    """Results of a backtesting run"""
    strategy_name: str
    total_bets: int
    wins: int
    losses: int
    pushes: int
    win_rate: float
    total_staked: float
    total_payout: float
    net_profit: float
    roi_percentage: float
    max_drawdown: float
    sharpe_ratio: float
    kelly_suggestions: List[Dict]
    monthly_returns: List[Dict]
    bet_distribution: Dict[str, int]

@dataclass
class BacktestStrategy:
    """Defines a betting strategy for backtesting"""
    name: str
    description: str
    bet_criteria: Callable[[HistoricalGame], Optional[Dict]]  # Returns bet details or None
    stake_calculator: Callable[[BacktestResult, float], float]  # Calculates stake size
    risk_management: Dict[str, Any]  # Risk parameters

class BacktestingEngine:
    """Engine for testing betting strategies against historical data"""

    def __init__(self, data_directory: str = None):
        self.data_directory = data_directory or os.path.join(os.path.dirname(__file__), 'historical_data')
        self.historical_games = []
        self.strategies = {}

        # Ensure data directory exists
        os.makedirs(self.data_directory, exist_ok=True)

        # Load historical data
        self._load_historical_data()

        # Register built-in strategies
        self._register_builtin_strategies()

    def _load_historical_data(self):
        """Load historical game data from files"""
        data_file = os.path.join(self.data_directory, 'historical_games.json')

        if os.path.exists(data_file):
            try:
                with open(data_file, 'r') as f:
                    data = json.load(f)
                    self.historical_games = [HistoricalGame(**game) for game in data.get('games', [])]
                    logger.info(f"Loaded {len(self.historical_games)} historical games")
            except Exception as e:
                logger.error(f"Error loading historical data: {e}")
                self.historical_games = []
        else:
            logger.info("No historical data file found, starting with empty dataset")
            self.historical_games = []

    def save_historical_data(self):
        """Save historical games to disk"""
        data_file = os.path.join(self.data_directory, 'historical_games.json')

        data = {
            'games': [asdict(game) for game in self.historical_games],
            'last_updated': datetime.now().isoformat(),
            'total_games': len(self.historical_games)
        }

        try:
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved {len(self.historical_games)} historical games")
        except Exception as e:
            logger.error(f"Error saving historical data: {e}")

    def add_historical_game(self, game: HistoricalGame):
        """Add a single historical game to the dataset"""
        # Check for duplicates
        existing_ids = {g.game_id for g in self.historical_games}
        if game.game_id not in existing_ids:
            self.historical_games.append(game)
            logger.info(f"Added historical game: {game.game_id}")
        else:
            logger.warning(f"Game {game.game_id} already exists, skipping")

    def import_games_from_season(self, season: int, sport: str = 'nfl'):
        """Import all games from a season (mock implementation)"""
        # In a real implementation, this would fetch from sports APIs
        # For now, we'll generate some mock historical data

        mock_games = self._generate_mock_historical_games(season, sport)
        for game in mock_games:
            self.add_historical_game(game)

        self.save_historical_data()
        logger.info(f"Imported {len(mock_games)} games from {season} {sport.upper()} season")

    def _generate_mock_historical_games(self, season: int, sport: str) -> List[HistoricalGame]:
        """Generate mock historical games for testing"""
        games = []

        # NFL teams for mock data
        nfl_teams = [
            'Kansas City Chiefs', 'Buffalo Bills', 'Philadelphia Eagles', 'San Francisco 49ers',
            'Detroit Lions', 'Cleveland Browns', 'Dallas Cowboys', 'Miami Dolphins',
            'Jacksonville Jaguars', 'New England Patriots', 'Pittsburgh Steelers',
            'Tennessee Titans', 'Indianapolis Colts', 'Cincinnati Bengals',
            'Seattle Seahawks', 'Arizona Cardinals', 'Atlanta Falcons', 'Carolina Panthers',
            'New Orleans Saints', 'Tampa Bay Buccaneers', 'Green Bay Packers', 'Chicago Bears',
            'Minnesota Vikings', 'New York Giants', 'New York Jets', 'Las Vegas Raiders',
            'Los Angeles Chargers', 'Los Angeles Rams', 'Denver Broncos', 'Baltimore Ravens'
        ]

        import random

        # Generate games for the season (16 weeks + playoffs)
        for week in range(1, 18):  # Regular season + playoffs
            week_games = min(16, len(nfl_teams) // 2)  # Max 16 games per week

            used_teams = set()
            for _ in range(week_games):
                # Pick two unused teams
                available_teams = [t for t in nfl_teams if t not in used_teams]
                if len(available_teams) < 2:
                    break

                home_team = random.choice(available_teams)
                available_teams.remove(home_team)
                away_team = random.choice(available_teams)
                available_teams.remove(away_team)

                used_teams.add(home_team)
                used_teams.add(away_team)

                # Generate realistic scores and odds
                home_score = random.randint(13, 45)
                away_score = random.randint(10, 42)

                winner = home_team if home_score > away_score else away_team

                # Generate spread (typically 1-14 points)
                spread_line = random.uniform(1, 14)
                if winner == home_team:
                    spread_result = 'home_cover' if home_score - away_score > spread_line else 'away_cover'
                else:
                    spread_result = 'away_cover' if away_score - home_score > spread_line else 'home_cover'

                # Generate total (typically 37-55)
                total_line = random.uniform(37, 55)
                total_points = home_score + away_score
                total_result = 'over' if total_points > total_line else 'under'

                # Generate realistic odds
                favorite_odds = random.uniform(1.2, 3.0)
                underdog_odds = random.uniform(1.8, 6.0)

                if random.choice([True, False]):
                    odds_moneyline_home, odds_moneyline_away = favorite_odds, underdog_odds
                else:
                    odds_moneyline_home, odds_moneyline_away = underdog_odds, favorite_odds

                # Spread odds
                odds_spread_home = random.uniform(1.85, 2.1)
                odds_spread_away = random.uniform(1.85, 2.1)

                # Total odds
                odds_total_over = random.uniform(1.85, 2.0)
                odds_total_under = random.uniform(1.85, 2.0)

                game_date = f"{season}-{week:02d}-{random.randint(1, 7):02d}"

                game = HistoricalGame(
                    game_id=f"{sport}_{season}_week{week}_{home_team.replace(' ', '_')}_{away_team.replace(' ', '_')}",
                    home_team=home_team,
                    away_team=away_team,
                    game_date=game_date,
                    final_score_home=home_score,
                    final_score_away=away_score,
                    winner=winner,
                    spread_result=spread_result,
                    total_result=total_result,
                    odds_moneyline_home=round(odds_moneyline_home, 2),
                    odds_moneyline_away=round(odds_moneyline_away, 2),
                    odds_spread_home=round(odds_spread_home, 2),
                    odds_spread_away=round(odds_spread_away, 2),
                    spread_line=round(spread_line, 1),
                    odds_total_over=round(odds_total_over, 2),
                    odds_total_under=round(odds_total_under, 2),
                    total_line=round(total_line, 1)
                )

                games.append(game)

        return games

    def _register_builtin_strategies(self):
        """Register built-in betting strategies"""

        # Strategy 1: Favorite Moneyline
        def favorite_ml_criteria(game: HistoricalGame) -> Optional[Dict]:
            # Bet on the favorite if odds are reasonable (< 2.5)
            if game.odds_moneyline_home < game.odds_moneyline_away and game.odds_moneyline_home <= 2.5:
                return {
                    'bet_type': 'moneyline',
                    'selection': game.home_team,
                    'odds': game.odds_moneyline_home
                }
            elif game.odds_moneyline_away < game.odds_moneyline_home and game.odds_moneyline_away <= 2.5:
                return {
                    'bet_type': 'moneyline',
                    'selection': game.away_team,
                    'odds': game.odds_moneyline_away
                }
            return None

        # Strategy 2: Underdog Moneyline
        def underdog_ml_criteria(game: HistoricalGame) -> Optional[Dict]:
            # Bet on the underdog if odds are reasonable (> 2.0)
            if game.odds_moneyline_home > game.odds_moneyline_away and game.odds_moneyline_home >= 2.0:
                return {
                    'bet_type': 'moneyline',
                    'selection': game.home_team,
                    'odds': game.odds_moneyline_home
                }
            elif game.odds_moneyline_away > game.odds_moneyline_home and game.odds_moneyline_away >= 2.0:
                return {
                    'bet_type': 'moneyline',
                    'selection': game.away_team,
                    'odds': game.odds_moneyline_away
                }
            return None

        # Strategy 3: Overs (high totals)
        def overs_criteria(game: HistoricalGame) -> Optional[Dict]:
            if game.total_line >= 45:  # Only bet overs on high total games
                return {
                    'bet_type': 'total',
                    'selection': 'over',
                    'odds': game.odds_total_over
                }
            return None

        # Strategy 4: Home Underdog
        def home_underdog_criteria(game: HistoricalGame) -> Optional[Dict]:
            if game.odds_moneyline_home > game.odds_moneyline_away and game.odds_moneyline_home <= 3.0:
                return {
                    'bet_type': 'moneyline',
                    'selection': game.home_team,
                    'odds': game.odds_moneyline_home
                }
            return None

        def flat_stake_calculator(result: BacktestResult, base_stake: float) -> float:
            return base_stake

        def percentage_stake_calculator(result: BacktestResult, base_stake: float) -> float:
            # Use percentage of current bankroll
            current_bankroll = 1000 + result.net_profit  # Start with $1000
            return max(1, current_bankroll * base_stake)  # Minimum $1 bet

        strategies = [
            BacktestStrategy(
                name="Favorite ML",
                description="Bet on moneyline favorites with odds <= 2.5",
                bet_criteria=favorite_ml_criteria,
                stake_calculator=flat_stake_calculator,
                risk_management={'max_bets_per_day': 5, 'max_consecutive_losses': 3}
            ),
            BacktestStrategy(
                name="Underdog ML",
                description="Bet on moneyline underdogs with odds >= 2.0",
                bet_criteria=underdog_ml_criteria,
                stake_calculator=flat_stake_calculator,
                risk_management={'max_bets_per_day': 3, 'max_consecutive_losses': 2}
            ),
            BacktestStrategy(
                name="High Total Overs",
                description="Bet overs on games with total >= 45",
                bet_criteria=overs_criteria,
                stake_calculator=flat_stake_calculator,
                risk_management={'max_bets_per_day': 4, 'max_consecutive_losses': 3}
            ),
            BacktestStrategy(
                name="Home Underdog",
                description="Bet on home underdogs with odds <= 3.0",
                bet_criteria=home_underdog_criteria,
                stake_calculator=percentage_stake_calculator,
                risk_management={'max_bets_per_day': 3, 'max_consecutive_losses': 2}
            )
        ]

        for strategy in strategies:
            self.register_strategy(strategy)

    def register_strategy(self, strategy: BacktestStrategy):
        """Register a betting strategy"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Registered strategy: {strategy.name}")

    def run_backtest(self, strategy_name: str, start_date: str = None, end_date: str = None,
                    initial_bankroll: float = 1000.0, base_stake: float = 10.0) -> BacktestResult:
        """Run a backtest for a specific strategy"""

        if strategy_name not in self.strategies:
            raise ValueError(f"Strategy '{strategy_name}' not found")

        strategy = self.strategies[strategy_name]

        # Filter games by date range
        filtered_games = self._filter_games_by_date(start_date, end_date)

        if not filtered_games:
            raise ValueError("No games found in the specified date range")

        # Sort games by date
        filtered_games.sort(key=lambda g: g.game_date)

        # Run the backtest
        bets = []
        current_bankroll = initial_bankroll
        consecutive_losses = 0
        max_drawdown = 0
        peak_bankroll = initial_bankroll
        daily_pnl = {}

        for game in filtered_games:
            # Check risk management
            if self._should_skip_bet(strategy, bets, consecutive_losses):
                continue

            # Get bet criteria
            bet_info = strategy.bet_criteria(game)
            if not bet_info:
                continue

            # Calculate stake
            stake = strategy.stake_calculator(
                BacktestResult(
                    strategy_name=strategy_name,
                    total_bets=len(bets),
                    wins=sum(1 for b in bets if b.result == 'win'),
                    losses=sum(1 for b in bets if b.result == 'loss'),
                    pushes=sum(1 for b in bets if b.result == 'push'),
                    win_rate=0,
                    total_staked=sum(b.stake for b in bets),
                    total_payout=sum(b.payout for b in bets if b.result == 'win'),
                    net_profit=sum(b.payout for b in bets if b.result == 'win') - sum(b.stake for b in bets),
                    roi_percentage=0,
                    max_drawdown=0,
                    sharpe_ratio=0,
                    kelly_suggestions=[],
                    monthly_returns=[],
                    bet_distribution={}
                ),
                base_stake
            )

            # Determine bet result
            result, payout = self._calculate_bet_result(game, bet_info, stake)

            # Create bet record
            bet = BacktestBet(
                game_id=game.game_id,
                bet_type=bet_info['bet_type'],
                selection=bet_info['selection'],
                odds=bet_info['odds'],
                stake=stake,
                result=result,
                payout=payout,
                timestamp=datetime.now().isoformat()
            )

            bets.append(bet)

            # Update tracking
            if result == 'win':
                current_bankroll += payout
                consecutive_losses = 0
            elif result == 'loss':
                current_bankroll -= stake
                consecutive_losses += 1
            # Push doesn't change bankroll or consecutive losses

            # Track daily P&L
            game_date = game.game_date.split('T')[0] if 'T' in game.game_date else game.game_date
            if game_date not in daily_pnl:
                daily_pnl[game_date] = 0
            if result == 'win':
                daily_pnl[game_date] += payout
            elif result == 'loss':
                daily_pnl[game_date] -= stake

            # Track drawdown
            peak_bankroll = max(peak_bankroll, current_bankroll)
            current_drawdown = peak_bankroll - current_bankroll
            max_drawdown = max(max_drawdown, current_drawdown)

        # Calculate final results
        return self._calculate_backtest_results(strategy_name, bets, initial_bankroll, max_drawdown, daily_pnl)

    def _filter_games_by_date(self, start_date: str = None, end_date: str = None) -> List[HistoricalGame]:
        """Filter games by date range"""
        if not start_date and not end_date:
            return self.historical_games

        filtered = []
        for game in self.historical_games:
            game_date = game.game_date

            if start_date and game_date < start_date:
                continue
            if end_date and game_date > end_date:
                continue

            filtered.append(game)

        return filtered

    def _should_skip_bet(self, strategy: BacktestStrategy, bets: List[BacktestBet], consecutive_losses: int) -> bool:
        """Check if we should skip this bet based on risk management"""
        risk = strategy.risk_management

        # Check consecutive losses
        if 'max_consecutive_losses' in risk and consecutive_losses >= risk['max_consecutive_losses']:
            return True

        # Check daily bet limit
        today = datetime.now().date().isoformat()
        todays_bets = sum(1 for bet in bets if bet.timestamp.startswith(today))
        if 'max_bets_per_day' in risk and todays_bets >= risk['max_bets_per_day']:
            return True

        return False

    def _calculate_bet_result(self, game: HistoricalGame, bet_info: Dict, stake: float) -> tuple[str, float]:
        """Calculate the result of a bet"""
        bet_type = bet_info['bet_type']
        selection = bet_info['selection']
        odds = bet_info['odds']

        if bet_type == 'moneyline':
            if selection == game.winner:
                return 'win', stake * (odds - 1)
            else:
                return 'loss', 0

        elif bet_type == 'spread':
            if selection == game.home_team and game.spread_result == 'home_cover':
                return 'win', stake * (odds - 1)
            elif selection == game.away_team and game.spread_result == 'away_cover':
                return 'win', stake * (odds - 1)
            elif game.spread_result == 'push':
                return 'push', stake  # Return stake on push
            else:
                return 'loss', 0

        elif bet_type == 'total':
            if selection == 'over' and game.total_result == 'over':
                return 'win', stake * (odds - 1)
            elif selection == 'under' and game.total_result == 'under':
                return 'win', stake * (odds - 1)
            elif game.total_result == 'push':
                return 'push', stake  # Return stake on push
            else:
                return 'loss', 0

        return 'loss', 0

    def _calculate_backtest_results(self, strategy_name: str, bets: List[BacktestBet],
                                  initial_bankroll: float, max_drawdown: float,
                                  daily_pnl: Dict[str, float]) -> BacktestResult:
        """Calculate comprehensive backtest results"""

        if not bets:
            return BacktestResult(
                strategy_name=strategy_name,
                total_bets=0,
                wins=0,
                losses=0,
                pushes=0,
                win_rate=0,
                total_staked=0,
                total_payout=0,
                net_profit=0,
                roi_percentage=0,
                max_drawdown=0,
                sharpe_ratio=0,
                kelly_suggestions=[],
                monthly_returns=[],
                bet_distribution={}
            )

        wins = sum(1 for b in bets if b.result == 'win')
        losses = sum(1 for b in bets if b.result == 'loss')
        pushes = sum(1 for b in bets if b.result == 'push')

        total_bets = len(bets)
        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        total_staked = sum(b.stake for b in bets)
        total_payout = sum(b.payout for b in bets if b.result == 'win')
        net_profit = total_payout - total_staked

        roi_percentage = (net_profit / total_staked * 100) if total_staked > 0 else 0

        # Calculate Sharpe ratio (risk-adjusted return)
        daily_returns = list(daily_pnl.values())
        if len(daily_returns) > 1:
            avg_return = statistics.mean(daily_returns)
            std_return = statistics.stdev(daily_returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        else:
            sharpe_ratio = 0

        # Kelly Criterion suggestions
        kelly_suggestions = self._calculate_kelly_suggestions(win_rate, avg_odds=bets[0].odds if bets else 2.0)

        # Monthly returns (simplified)
        monthly_returns = self._calculate_monthly_returns(daily_pnl)

        # Bet distribution
        bet_distribution = {}
        for bet in bets:
            key = f"{bet.bet_type}_{bet.selection}"
            bet_distribution[key] = bet_distribution.get(key, 0) + 1

        return BacktestResult(
            strategy_name=strategy_name,
            total_bets=total_bets,
            wins=wins,
            losses=losses,
            pushes=pushes,
            win_rate=win_rate,
            total_staked=round(total_staked, 2),
            total_payout=round(total_payout, 2),
            net_profit=round(net_profit, 2),
            roi_percentage=round(roi_percentage, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            kelly_suggestions=kelly_suggestions,
            monthly_returns=monthly_returns,
            bet_distribution=bet_distribution
        )

    def _calculate_kelly_suggestions(self, win_rate: float, avg_odds: float) -> List[Dict]:
        """Calculate Kelly Criterion bet sizing suggestions"""
        if win_rate <= 0 or win_rate >= 1:
            return []

        # Simplified Kelly: f = (bp - q) / b
        # where b = odds - 1, p = win_rate, q = 1 - p
        b = avg_odds - 1
        kelly_fraction = (win_rate * b - (1 - win_rate)) / b

        suggestions = []
        for fraction in [0.25, 0.5, 0.75, 1.0]:
            stake_fraction = max(0, kelly_fraction * fraction)
            suggestions.append({
                'kelly_fraction': round(stake_fraction, 3),
                'description': f"{fraction*100:.0f}% Kelly"
            })

        return suggestions

    def _calculate_monthly_returns(self, daily_pnl: Dict[str, float]) -> List[Dict]:
        """Calculate monthly return statistics"""
        # Group by month
        monthly_data = {}
        for date_str, pnl in daily_pnl.items():
            try:
                date = datetime.fromisoformat(date_str)
                month_key = f"{date.year}-{date.month:02d}"
                if month_key not in monthly_data:
                    monthly_data[month_key] = []
                monthly_data[month_key].append(pnl)
            except:
                continue

        monthly_returns = []
        for month, pnls in monthly_data.items():
            total_pnl = sum(pnls)
            monthly_returns.append({
                'month': month,
                'total_pnl': round(total_pnl, 2),
                'days': len(pnls)
            })

        return monthly_returns

    def compare_strategies(self, strategy_names: List[str], **backtest_kwargs) -> Dict[str, BacktestResult]:
        """Compare multiple strategies"""
        results = {}

        for strategy_name in strategy_names:
            if strategy_name in self.strategies:
                try:
                    result = self.run_backtest(strategy_name, **backtest_kwargs)
                    results[strategy_name] = result
                except Exception as e:
                    logger.error(f"Error backtesting {strategy_name}: {e}")

        return results

    def get_available_strategies(self) -> List[str]:
        """Get list of available strategy names"""
        return list(self.strategies.keys())

    def get_strategy_info(self, strategy_name: str) -> Optional[Dict]:
        """Get information about a specific strategy"""
        if strategy_name not in self.strategies:
            return None

        strategy = self.strategies[strategy_name]
        return {
            'name': strategy.name,
            'description': strategy.description,
            'risk_management': strategy.risk_management
        }
