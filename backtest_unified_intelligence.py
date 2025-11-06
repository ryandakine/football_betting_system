#!/usr/bin/env python3
"""
Backtest Unified Intelligence System
=====================================
Tests the unified betting intelligence system against historical NFL games
to measure prediction accuracy and profitability.
"""

import argparse
import asyncio
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

from unified_intelligence_system import UnifiedBettingIntelligence

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_primary_pick(recommendation: str) -> Optional[Dict[str, str]]:
    """Infer market and side from the primary recommendation string."""

    if not recommendation or recommendation == "PASS":
        return None

    normalized = recommendation.upper()
    if "UNDER" in normalized:
        return {"market": "total", "side": "under"}
    if "OVER" in normalized:
        return {"market": "total", "side": "over"}
    if "HOME" in normalized and "SPREAD" in normalized:
        return {"market": "spread", "side": "home"}
    if "AWAY" in normalized and "SPREAD" in normalized:
        return {"market": "spread", "side": "away"}
    return None

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/referee_conspiracy")
BACKTEST_DIR = DATA_DIR / "backtests"
BACKTEST_DIR.mkdir(parents=True, exist_ok=True)


class UnifiedIntelligenceBacktester:
    """Backtest the unified intelligence system"""
    
    def __init__(self):
        self.intelligence = UnifiedBettingIntelligence()
        self.results = []
    
    async def backtest_date(self, target_date: date) -> Dict[str, Any]:
        """Backtest predictions for a specific date"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 BACKTESTING: {target_date}")
        logger.info(f"{'='*80}")
        
        # Find odds file for this date
        odds_file = DATA_DIR / f"nfl_odds_{target_date}.json"
        
        if not odds_file.exists():
            logger.warning(f"No odds file found for {target_date}")
            return {
                "date": str(target_date),
                "games": 0,
                "predictions": [],
                "error": "No odds data available"
            }
        
        # Run intelligence analysis
        try:
            recommendations = await self.intelligence.analyze_all_games(odds_file)
            
            # Load actual results if available
            results_file = DATA_DIR / f"game_results_{target_date}.json"
            actual_results = {}
            
            if results_file.exists():
                actual_results = json.loads(results_file.read_text())
                logger.info(f"✅ Found actual results for {target_date}")
            else:
                logger.warning(f"⚠️  No actual results file found for {target_date}")
            
            # Grade predictions
            graded_predictions = self._grade_predictions(recommendations, actual_results)
            
            # Calculate metrics
            metrics = self._calculate_metrics(graded_predictions)
            
            backtest_result = {
                "date": str(target_date),
                "games": len(recommendations),
                "predictions": graded_predictions,
                "metrics": metrics,
            }
            
            self.results.append(backtest_result)
            
            # Print summary
            self._print_day_summary(target_date, metrics)
            
            return backtest_result
        
        except Exception as e:
            logger.error(f"Backtest failed for {target_date}: {e}")
            return {
                "date": str(target_date),
                "games": 0,
                "predictions": [],
                "error": str(e)
            }
    
    def _grade_predictions(
        self,
        recommendations: List[Dict],
        actual_results: Dict[str, Any]
    ) -> List[Dict]:
        """Grade each prediction against actual results"""
        
        graded = []
        
        for rec in recommendations:
            game_id = rec.get("game_id")
            recommendation = rec.get("recommendation", "")
            confidence = float(rec.get("confidence", 0.0) or 0.0)

            actual = actual_results.get(game_id, {})
            actual_total = actual.get("total_score")
            actual_spread = actual.get("spread_result")

            market_total = rec.get("market_lines", {}).get("total")
            market_spread = rec.get("market_lines", {}).get("spread")

            parsed_pick = _parse_primary_pick(recommendation)
            base_stake = max(1.0, confidence * 100.0)
            stake_record = 0.0
            result = "PENDING"
            profit = 0.0

            if parsed_pick and parsed_pick["market"] == "total":
                if actual_total is None or market_total is None:
                    result = "NO_DATA"
                else:
                    if actual_total == market_total:
                        result = "PUSH"
                    elif parsed_pick["side"] == "under":
                        if actual_total < market_total:
                            result = "WIN"
                            stake_record = base_stake
                            profit = base_stake
                        else:
                            result = "LOSS"
                            stake_record = base_stake
                            profit = -base_stake
                    else:  # over
                        if actual_total > market_total:
                            result = "WIN"
                            stake_record = base_stake
                            profit = base_stake
                        else:
                            result = "LOSS"
                            stake_record = base_stake
                            profit = -base_stake
            elif parsed_pick and parsed_pick["market"] == "spread":
                if actual_spread is None or market_spread is None:
                    result = "NO_DATA"
                else:
                    # spread_result is home margin (home_score - away_score)
                    margin = float(actual_spread)
                    target = float(market_spread)
                    if margin == target:
                        result = "PUSH"
                    elif parsed_pick["side"] == "home":
                        if margin > target:
                            result = "WIN"
                            stake_record = base_stake
                            profit = base_stake
                        else:
                            result = "LOSS"
                            stake_record = base_stake
                            profit = -base_stake
                    else:
                        if margin < target:
                            result = "WIN"
                            stake_record = base_stake
                            profit = base_stake
                        else:
                            result = "LOSS"
                            stake_record = base_stake
                            profit = -base_stake

            graded.append(
                {
                    "game_id": game_id,
                    "matchup": rec.get("matchup"),
                    "recommendation": recommendation,
                    "confidence": confidence,
                    "bet_strength": rec.get("bet_strength"),
                    "market_total": market_total,
                    "actual_total": actual_total,
                    "market_spread": market_spread,
                    "actual_spread": actual_spread,
                    "result": result,
                    "profit": profit,
                    "stake": stake_record,
                    "reasoning": rec.get("reasoning", []),
                }
            )

        return graded
    
    def _calculate_metrics(self, predictions: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics"""
        
        # Filter to graded games only
        graded = [p for p in predictions if p["result"] in ["WIN", "LOSS"]]
        
        if not graded:
            return {
                "total_bets": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
                "avg_confidence": 0.0,
            }
        
        wins = len([p for p in graded if p["result"] == "WIN"])
        losses = len([p for p in graded if p["result"] == "LOSS"])
        total_profit = sum(p["profit"] for p in graded)
        total_risk = sum(p.get("stake", 0.0) for p in graded)

        # Filter to strong bets
        strong_bets = [p for p in graded if p.get("stake", 0.0) > 0 and p["bet_strength"] in ["STRONG_BET", "LEAN"]]
        strong_wins = len([p for p in strong_bets if p["result"] == "WIN"]) if strong_bets else 0

        return {
            "total_bets": len(graded),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / len(graded) if graded else 0.0,
            "total_profit": total_profit,
            "roi": (total_profit / total_risk * 100) if total_risk > 0 else 0.0,
            "avg_confidence": sum(p["confidence"] for p in graded) / len(graded) if graded else 0.0,
            "strong_bets": len(strong_bets),
            "strong_wins": strong_wins,
            "strong_win_rate": strong_wins / len(strong_bets) if strong_bets else 0.0,
        }
    
    def _print_day_summary(self, target_date: date, metrics: Dict):
        """Print summary for a single day"""
        
        print(f"\n{'='*80}")
        print(f"📊 {target_date} RESULTS")
        print(f"{'='*80}")
        print(f"Bets: {metrics['total_bets']}")
        print(f"Record: {metrics['wins']}-{metrics['losses']} ({metrics['win_rate']:.1%})")
        print(f"Profit: ${metrics['total_profit']:+.2f}")
        print(f"ROI: {metrics['roi']:+.1f}%")
        print(f"Avg Confidence: {metrics['avg_confidence']:.1%}")
        
        if metrics['strong_bets'] > 0:
            print(f"\n🎯 Strong Bets: {metrics['strong_wins']}/{metrics['strong_bets']} ({metrics['strong_win_rate']:.1%})")
    
    async def backtest_range(self, start_date: date, end_date: date) -> Dict[str, Any]:
        """Backtest across a date range"""
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🔬 BACKTESTING: {start_date} to {end_date}")
        logger.info(f"{'='*80}\n")
        
        current_date = start_date
        
        while current_date <= end_date:
            await self.backtest_date(current_date)
            current_date += timedelta(days=1)
        
        # Calculate overall metrics
        overall = self._calculate_overall_metrics()
        
        # Save results
        self._save_backtest_results(overall, start_date, end_date)
        
        # Print final summary
        self._print_overall_summary(overall)
        
        return overall
    
    def _calculate_overall_metrics(self) -> Dict[str, Any]:
        """Calculate metrics across all dates"""
        
        all_predictions = []
        for result in self.results:
            all_predictions.extend(result.get("predictions", []))
        
        graded = [p for p in all_predictions if p["result"] in ["WIN", "LOSS"]]
        
        if not graded:
            return {
                "total_days": len(self.results),
                "total_bets": 0,
                "record": "0-0",
                "win_rate": 0.0,
                "total_profit": 0.0,
                "roi": 0.0,
            }
        
        wins = len([p for p in graded if p["result"] == "WIN"])
        losses = len([p for p in graded if p["result"] == "LOSS"])
        total_profit = sum(p["profit"] for p in graded)
        total_risk = sum(p.get("stake", 0.0) for p in graded)

        # Breakdown by confidence
        high_conf = [p for p in graded if p["confidence"] >= 0.7]
        high_wins = len([p for p in high_conf if p["result"] == "WIN"]) if high_conf else 0

        strong_bets = [p for p in graded if p.get("stake", 0.0) > 0 and p["bet_strength"] in ["STRONG_BET", "LEAN"]]
        strong_wins = len([p for p in strong_bets if p["result"] == "WIN"]) if strong_bets else 0
        
        return {
            "total_days": len(self.results),
            "total_bets": len(graded),
            "record": f"{wins}-{losses}",
            "win_rate": wins / len(graded) if graded else 0.0,
            "total_profit": total_profit,
            "roi": (total_profit / total_risk * 100) if total_risk > 0 else 0.0,
            "avg_confidence": sum(p["confidence"] for p in graded) / len(graded) if graded else 0.0,
            "high_confidence_bets": len(high_conf),
            "high_confidence_wins": high_wins,
            "high_confidence_win_rate": high_wins / len(high_conf) if high_conf else 0.0,
            "strong_bets": len(strong_bets),
            "strong_wins": strong_wins,
            "strong_win_rate": strong_wins / len(strong_bets) if strong_bets else 0.0,
            "daily_results": self.results,
        }
    
    def _save_backtest_results(self, results: Dict, start_date: date, end_date: date):
        """Save backtest results to file"""
        
        filename = f"backtest_{start_date}_to_{end_date}.json"
        output_file = BACKTEST_DIR / filename
        
        output_file.write_text(json.dumps(results, indent=2))
        logger.info(f"\n💾 Saved backtest results to {output_file}")
    
    def _print_overall_summary(self, metrics: Dict):
        """Print overall backtest summary"""
        
        print(f"\n{'='*80}")
        print(f"🏆 OVERALL BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Days Tested: {metrics['total_days']}")
        print(f"Total Bets: {metrics['total_bets']}")
        print(f"Record: {metrics['record']} ({metrics['win_rate']:.1%})")
        print(f"Total Profit: ${metrics['total_profit']:+.2f}")
        print(f"ROI: {metrics['roi']:+.1f}%")
        
        if metrics.get('avg_confidence'):
            print(f"Avg Confidence: {metrics['avg_confidence']:.1%}")
        
        if metrics.get('high_confidence_bets', 0) > 0:
            print(f"\n📈 High Confidence (≥70%):")
            print(f"   Bets: {metrics['high_confidence_bets']}")
            print(f"   Record: {metrics['high_confidence_wins']}-{metrics['high_confidence_bets'] - metrics['high_confidence_wins']}")
            print(f"   Win Rate: {metrics['high_confidence_win_rate']:.1%}")
        
        if metrics.get('strong_bets', 0) > 0:
            print(f"\n🎯 Strong Bets (STRONG_BET/LEAN):")
            print(f"   Bets: {metrics['strong_bets']}")
            print(f"   Record: {metrics['strong_wins']}-{metrics['strong_bets'] - metrics['strong_wins']}")
            print(f"   Win Rate: {metrics['strong_win_rate']:.1%}")
        
        print(f"\n{'='*80}")


async def main():
    """Run backtest"""
    
    parser = argparse.ArgumentParser(description="Unified intelligence backtester")
    parser.add_argument("--start", type=str, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, help="End date YYYY-MM-DD")
    parser.add_argument("--years", type=int, default=10, help="Number of years to backtest when start/end not provided")
    args = parser.parse_args()

    backtester = UnifiedIntelligenceBacktester()

    if args.start and args.end:
        start_date = date.fromisoformat(args.start)
        end_date = date.fromisoformat(args.end)
    else:
        years = max(1, args.years or 10)
        end_date = date.today()
        start_date = end_date - timedelta(days=365 * years)
    
    print(f"\n🔬 Starting Unified Intelligence Backtest")
    print(f"📅 Date Range: {start_date} to {end_date}\n")
    
    results = await backtester.backtest_range(start_date, end_date)
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(main())
