#!/usr/bin/env python3
"""
NCAA Betting Agent System
Autonomous agent for collecting data, analyzing games, and generating predictions
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Configuration for NCAA betting agent"""
    # API Keys
    cfb_api_key: str
    odds_api_key: Optional[str] = None

    # Data paths
    data_dir: Path = Path("data/football/historical/ncaaf")
    results_dir: Path = Path("data/agents/ncaa/results")

    # Betting parameters
    bankroll: float = 10000.0
    min_edge: float = 0.03
    min_confidence: float = 0.60
    max_bet_percent: float = 0.15

    # Agent behavior
    auto_collect_data: bool = True
    auto_generate_picks: bool = True
    auto_track_results: bool = True

    # Scheduling
    run_on_days: List[str] = None  # ["tuesday", "wednesday", "saturday"]

    def __post_init__(self):
        if self.run_on_days is None:
            self.run_on_days = ["tuesday", "wednesday", "saturday"]
        self.data_dir = Path(self.data_dir)
        self.results_dir = Path(self.results_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)


class NCAABettingAgent:
    """
    Main orchestrator agent for NCAA betting system
    Coordinates data collection, analysis, and prediction sub-agents
    """

    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = {
            'last_run': None,
            'total_bets': 0,
            'total_wins': 0,
            'current_bankroll': config.bankroll,
            'last_data_update': None,
            'season': self._get_current_season()
        }
        self._load_state()

        # Initialize sub-agents
        from ncaa_agents.data_collector import DataCollectorAgent
        from ncaa_agents.analyzer import AnalysisAgent
        from ncaa_agents.predictor import PredictionAgent
        from ncaa_agents.tracker import PerformanceTrackerAgent

        self.data_collector = DataCollectorAgent(config)
        self.analyzer = AnalysisAgent(config)
        self.predictor = PredictionAgent(config)
        self.tracker = PerformanceTrackerAgent(config)

    def _get_current_season(self) -> int:
        """Determine current NCAA season"""
        now = datetime.now()
        # NCAA season runs Aug-Jan
        if now.month >= 8:
            return now.year
        else:
            return now.year - 1

    def _load_state(self):
        """Load agent state from disk"""
        state_file = self.config.results_dir / "agent_state.json"
        if state_file.exists():
            with open(state_file) as f:
                saved_state = json.load(f)
                self.state.update(saved_state)
            logger.info(f"Loaded agent state: {self.state['total_bets']} bets placed")

    def _save_state(self):
        """Save agent state to disk"""
        state_file = self.config.results_dir / "agent_state.json"
        with open(state_file, 'w') as f:
            json.dump(self.state, f, indent=2, default=str)
        logger.info("Agent state saved")

    async def run_daily_cycle(self):
        """
        Main agent loop - runs daily tasks
        """
        logger.info("="*70)
        logger.info("NCAA BETTING AGENT - DAILY CYCLE")
        logger.info("="*70)

        today = datetime.now()
        day_name = today.strftime("%A").lower()

        logger.info(f"Today: {today.strftime('%Y-%m-%d %A')}")
        logger.info(f"Current Season: {self.state['season']}")
        logger.info(f"Current Bankroll: ${self.state['current_bankroll']:,.2f}")

        # Task 1: Update data if needed (weekly)
        if self.config.auto_collect_data:
            await self._update_data()

        # Task 2: Run backtest on new data (weekly)
        if day_name in ["monday", "tuesday"]:
            await self._run_backtest()

        # Task 3: Generate picks for upcoming games
        if self.config.auto_generate_picks:
            picks = await self._generate_picks()
            if picks:
                self._display_picks(picks)
                self._save_picks(picks)

        # Task 4: Track results from recent bets
        if self.config.auto_track_results:
            await self._track_recent_results()

        # Update state
        self.state['last_run'] = datetime.now().isoformat()
        self._save_state()

        logger.info("\n✅ Daily cycle complete!")

    async def _update_data(self):
        """Update game data and SP+ ratings"""
        logger.info("\n📊 Checking for data updates...")

        # Check if we need to update (once per week)
        last_update = self.state.get('last_data_update')
        if last_update:
            last_update_date = datetime.fromisoformat(last_update)
            days_since_update = (datetime.now() - last_update_date).days

            if days_since_update < 7:
                logger.info(f"Data updated {days_since_update} days ago - skipping")
                return

        logger.info("Updating NCAA data...")
        result = await self.data_collector.collect_current_season()

        if result['success']:
            self.state['last_data_update'] = datetime.now().isoformat()
            logger.info(f"✅ Collected {result['games_count']} games")
        else:
            logger.error(f"❌ Data collection failed: {result['error']}")

    async def _run_backtest(self):
        """Run backtest to validate system performance"""
        logger.info("\n📈 Running backtest...")

        results = await self.analyzer.run_backtest(
            seasons=[self.state['season']],
            min_edge=self.config.min_edge,
            min_confidence=self.config.min_confidence
        )

        if results:
            logger.info(f"Backtest Results:")
            logger.info(f"  ROI: {results['roi']:.2f}%")
            logger.info(f"  Win Rate: {results['win_rate']:.1%}")
            logger.info(f"  P-Value: {results['p_value']:.4f}")

            # Save backtest results
            backtest_file = self.config.results_dir / f"backtest_{datetime.now().strftime('%Y%m%d')}.json"
            with open(backtest_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

    async def _generate_picks(self) -> List[Dict]:
        """Generate betting picks for upcoming games"""
        logger.info("\n🎯 Generating picks for upcoming games...")

        # Get upcoming games (next 7 days)
        picks = await self.predictor.generate_picks(
            days_ahead=7,
            min_edge=self.config.min_edge,
            min_confidence=self.config.min_confidence,
            current_bankroll=self.state['current_bankroll']
        )

        logger.info(f"Generated {len(picks)} picks")
        return picks

    def _display_picks(self, picks: List[Dict]):
        """Display generated picks"""
        if not picks:
            logger.info("No picks generated (no games meet criteria)")
            return

        logger.info("\n" + "="*70)
        logger.info("🏈 BETTING PICKS")
        logger.info("="*70)

        for i, pick in enumerate(picks, 1):
            logger.info(f"\n{i}. {pick['away_team']} @ {pick['home_team']}")
            logger.info(f"   Date: {pick['date']}")
            logger.info(f"   Pick: {pick['pick']} ({pick['predicted_winner']})")
            logger.info(f"   Edge: {pick['edge']:.1%}")
            logger.info(f"   Confidence: {pick['confidence']:.1%}")
            logger.info(f"   Recommended Bet: ${pick['recommended_stake']:.2f}")
            logger.info(f"   Reasoning: {pick.get('reasoning', 'SP+ advantage')}")

    def _save_picks(self, picks: List[Dict]):
        """Save picks to file"""
        if not picks:
            return

        date_str = datetime.now().strftime('%Y%m%d')
        picks_file = self.config.results_dir / f"picks_{date_str}.json"

        with open(picks_file, 'w') as f:
            json.dump(picks, f, indent=2, default=str)

        logger.info(f"\n💾 Picks saved to: {picks_file}")

    async def _track_recent_results(self):
        """Track results of recent bets"""
        logger.info("\n📊 Tracking recent results...")

        results = await self.tracker.check_recent_bets(days=7)

        if results['completed_bets']:
            logger.info(f"\nRecent Results ({len(results['completed_bets'])} bets):")
            logger.info(f"  Wins: {results['wins']}")
            logger.info(f"  Losses: {results['losses']}")
            logger.info(f"  Win Rate: {results['win_rate']:.1%}")
            logger.info(f"  Profit: ${results['profit']:,.2f}")

            # Update state
            self.state['total_bets'] += results['completed_bets']
            self.state['total_wins'] += results['wins']
            self.state['current_bankroll'] += results['profit']

    async def manual_mode(self):
        """
        Interactive mode for manual agent operation
        """
        print("\n" + "="*70)
        print("NCAA BETTING AGENT - MANUAL MODE")
        print("="*70)
        print("\nCommands:")
        print("  1. update    - Update game data")
        print("  2. backtest  - Run backtest")
        print("  3. picks     - Generate picks for upcoming games")
        print("  4. results   - Check recent bet results")
        print("  5. status    - Show agent status")
        print("  6. run       - Run full daily cycle")
        print("  7. exit      - Exit")

        while True:
            cmd = input("\n> ").strip().lower()

            if cmd in ['1', 'update']:
                await self._update_data()
            elif cmd in ['2', 'backtest']:
                await self._run_backtest()
            elif cmd in ['3', 'picks']:
                picks = await self._generate_picks()
                self._display_picks(picks)
            elif cmd in ['4', 'results']:
                await self._track_recent_results()
            elif cmd in ['5', 'status']:
                self._show_status()
            elif cmd in ['6', 'run']:
                await self.run_daily_cycle()
            elif cmd in ['7', 'exit', 'quit']:
                print("Goodbye!")
                break
            else:
                print(f"Unknown command: {cmd}")

    def _show_status(self):
        """Display agent status"""
        print("\n" + "="*70)
        print("AGENT STATUS")
        print("="*70)
        print(f"Season: {self.state['season']}")
        print(f"Bankroll: ${self.state['current_bankroll']:,.2f}")
        print(f"Total Bets: {self.state['total_bets']}")
        print(f"Total Wins: {self.state['total_wins']}")
        if self.state['total_bets'] > 0:
            win_rate = self.state['total_wins'] / self.state['total_bets']
            print(f"Win Rate: {win_rate:.1%}")
        print(f"Last Run: {self.state.get('last_run', 'Never')}")
        print(f"Last Data Update: {self.state.get('last_data_update', 'Never')}")


async def main():
    """Main entry point"""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Load configuration
    config = AgentConfig(
        cfb_api_key=os.getenv("CFB_DATA_API_KEY", ""),
        odds_api_key=os.getenv("ODDS_API_KEY"),
        bankroll=float(os.getenv("NCAA_BANKROLL", "10000")),
        min_edge=float(os.getenv("NCAA_MIN_EDGE", "0.03")),
        min_confidence=float(os.getenv("NCAA_MIN_CONFIDENCE", "0.60"))
    )

    if not config.cfb_api_key:
        logger.error("CFB_DATA_API_KEY not found in environment!")
        return

    # Create agent
    agent = NCAABettingAgent(config)

    # Check if running in automated or manual mode
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        await agent.manual_mode()
    else:
        # Run daily cycle
        await agent.run_daily_cycle()


if __name__ == "__main__":
    asyncio.run(main())
