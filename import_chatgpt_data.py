#!/usr/bin/env python3
"""
Import ChatGPT Agent Mode Data - Integrates ChatGPT-collected data into betting system

WHY THIS EXISTS:
ChatGPT Agent Mode collects referee data, backtest verifications, etc.
This script imports that data into our automated betting system.

DESIGN PHILOSOPHY: Investment → System
- ChatGPT = Data collection agent (browses web, researches)
- This script = Data integration layer
- Betting system = Execution agent (uses the data)

USAGE:
    python import_chatgpt_data.py --referee-intel referee_intelligence.json
    python import_chatgpt_data.py --weekly-refs week_11_referees.json
    python import_chatgpt_data.py --backtest backtest_verification.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List


class ChatGPTDataImporter:
    """Imports data collected by ChatGPT Agent Mode"""

    def __init__(self):
        self.data_dir = Path(__file__).parent / "data"
        self.data_dir.mkdir(exist_ok=True)

    def import_referee_intelligence(self, source_file: Path):
        """
        Import expanded referee intelligence database from ChatGPT.

        Args:
            source_file: JSON file with referee data from ChatGPT
        """
        print(f"📥 Importing referee intelligence from {source_file.name}...")

        with open(source_file, 'r') as f:
            chatgpt_data = json.load(f)

        # Validate structure
        if 'referees' not in chatgpt_data:
            print("❌ Invalid format: Missing 'referees' key")
            return

        # Load existing intelligence (if any)
        intel_file = self.data_dir / "referee_intelligence.json"
        if intel_file.exists():
            with open(intel_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {"referees": {}}

        # Merge data (ChatGPT data takes precedence)
        for ref in chatgpt_data['referees']:
            ref_name = ref['name']
            existing['referees'][ref_name] = ref
            print(f"   ✅ Imported: {ref_name}")
            print(f"      - Edge: {ref.get('betting_edge', 'Unknown')}")
            print(f"      - Pace: {ref.get('pace', 'Unknown')}")

        # Save merged data
        with open(intel_file, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"\n✅ Saved to {intel_file}")
        print(f"   Total referees in database: {len(existing['referees'])}")

    def import_weekly_referees(self, source_file: Path):
        """
        Import current week's referee assignments from ChatGPT.

        Args:
            source_file: JSON file with this week's referee assignments
        """
        print(f"📥 Importing weekly referee assignments from {source_file.name}...")

        with open(source_file, 'r') as f:
            chatgpt_data = json.load(f)

        # Validate structure
        if 'assignments' not in chatgpt_data:
            print("❌ Invalid format: Missing 'assignments' key")
            return

        # Convert to referee_cache.json format
        cache = {}
        for assignment in chatgpt_data['assignments']:
            game = assignment['game']
            referee = assignment['referee']
            cache[game] = referee
            print(f"   ✅ {game:20s} → {referee}")

        # Save to cache
        cache_file = self.data_dir / "referee_cache.json"
        cache_data = {
            "week": chatgpt_data.get('week', 'Unknown'),
            "season": chatgpt_data.get('season', 2024),
            "assignments": cache
        }

        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2)

        print(f"\n✅ Saved to {cache_file}")
        print(f"   Week {chatgpt_data.get('week')} assignments imported")

    def import_backtest_verification(self, source_file: Path):
        """
        Import verified backtest results from ChatGPT.

        Args:
            source_file: JSON file with verified game results
        """
        print(f"📥 Importing backtest verification from {source_file.name}...")

        with open(source_file, 'r') as f:
            chatgpt_data = json.load(f)

        # Validate structure
        if 'verified_games' not in chatgpt_data:
            print("❌ Invalid format: Missing 'verified_games' key")
            return

        # Load existing backtest data
        backtest_file = self.data_dir / "backtest_verified.json"
        if backtest_file.exists():
            with open(backtest_file, 'r') as f:
                existing = json.load(f)
        else:
            existing = {"games": []}

        # Add verified games
        for game in chatgpt_data['verified_games']:
            game_id = f"{game['season']}_W{game['week']}_{game['game']}"

            # Check if already exists
            exists = any(
                g.get('id') == game_id
                for g in existing['games']
            )

            if not exists:
                game['id'] = game_id
                existing['games'].append(game)
                print(f"   ✅ {game['game']:20s} - {game['final_score']}")
            else:
                print(f"   ⏭️  {game['game']:20s} - Already verified")

        # Save merged data
        with open(backtest_file, 'w') as f:
            json.dump(existing, f, indent=2)

        print(f"\n✅ Saved to {backtest_file}")
        print(f"   Total verified games: {len(existing['games'])}")

    def import_historical_referee_data(self, source_file: Path):
        """
        Import historical referee data from ChatGPT (2020-2024).

        Args:
            source_file: JSON file with historical referee assignments
        """
        print(f"📥 Importing historical referee data from {source_file.name}...")

        with open(source_file, 'r') as f:
            chatgpt_data = json.load(f)

        # Validate structure
        if 'games' not in chatgpt_data or 'season' not in chatgpt_data:
            print("❌ Invalid format: Missing 'games' or 'season' key")
            return

        season = chatgpt_data['season']
        history_file = self.data_dir / f"referee_history_{season}.json"

        # Save historical data
        with open(history_file, 'w') as f:
            json.dump(chatgpt_data, f, indent=2)

        print(f"\n✅ Saved to {history_file}")
        print(f"   Season {season}: {len(chatgpt_data['games'])} games imported")

        # Analyze for betting patterns
        self._analyze_referee_patterns(chatgpt_data)

    def _analyze_referee_patterns(self, historical_data: Dict):
        """Analyze historical data for betting patterns."""
        print("\n📊 REFEREE PATTERN ANALYSIS:")

        # Group games by referee
        by_referee = {}
        for game in historical_data['games']:
            ref = game.get('referee')
            if not ref:
                continue

            if ref not in by_referee:
                by_referee[ref] = {
                    'games': 0,
                    'total_points': 0,
                    'penalties': 0
                }

            by_referee[ref]['games'] += 1
            by_referee[ref]['total_points'] += game.get('total_points', 0)
            by_referee[ref]['penalties'] += game.get('penalties', {}).get('total', 0)

        # Calculate averages
        for ref, stats in sorted(by_referee.items()):
            if stats['games'] < 5:  # Need at least 5 games
                continue

            avg_points = stats['total_points'] / stats['games']
            avg_penalties = stats['penalties'] / stats['games']

            print(f"\n   {ref}:")
            print(f"      Games: {stats['games']}")
            print(f"      Avg Points: {avg_points:.1f}")
            print(f"      Avg Penalties: {avg_penalties:.1f}")

            # Suggest betting edge
            if avg_points > 48:
                print(f"      💡 Betting Edge: OVER (high-scoring games)")
            elif avg_points < 42:
                print(f"      💡 Betting Edge: UNDER (low-scoring games)")

            if avg_penalties > 14:
                print(f"      💡 Note: High penalty rate (slows pace)")


def main():
    parser = argparse.ArgumentParser(
        description="Import data collected by ChatGPT Agent Mode"
    )
    parser.add_argument(
        "--referee-intel",
        type=Path,
        help="Import referee intelligence JSON"
    )
    parser.add_argument(
        "--weekly-refs",
        type=Path,
        help="Import weekly referee assignments JSON"
    )
    parser.add_argument(
        "--backtest",
        type=Path,
        help="Import backtest verification JSON"
    )
    parser.add_argument(
        "--historical",
        type=Path,
        help="Import historical referee data JSON"
    )

    args = parser.parse_args()

    importer = ChatGPTDataImporter()

    if args.referee_intel:
        if not args.referee_intel.exists():
            print(f"❌ File not found: {args.referee_intel}")
            sys.exit(1)
        importer.import_referee_intelligence(args.referee_intel)

    elif args.weekly_refs:
        if not args.weekly_refs.exists():
            print(f"❌ File not found: {args.weekly_refs}")
            sys.exit(1)
        importer.import_weekly_referees(args.weekly_refs)

    elif args.backtest:
        if not args.backtest.exists():
            print(f"❌ File not found: {args.backtest}")
            sys.exit(1)
        importer.import_backtest_verification(args.backtest)

    elif args.historical:
        if not args.historical.exists():
            print(f"❌ File not found: {args.historical}")
            sys.exit(1)
        importer.import_historical_referee_data(args.historical)

    else:
        parser.print_help()
        print("\n💡 TIP: Use ChatGPT Agent Mode to collect data first!")
        print("   See CHATGPT_AGENT_PROMPTS.md for prompts")


if __name__ == "__main__":
    main()
