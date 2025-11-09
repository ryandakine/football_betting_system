#!/usr/bin/env python3
"""
NFL Backtesting Harness - IMPROVED VERSION
===========================================

🐛 BUG FIXES APPLIED:
1. ✅ Fixed American odds conversion (was missing +1.0)
2. ✅ Fixed profit calculation (was double-counting)
3. ✅ Implemented proper Kelly Criterion (was arbitrary multiplier)
4. ✅ Added edge validation (was trusting bad data)
5. ✅ Fixed Sharpe ratio (was over-annualizing)
6. ✅ Added statistical significance testing
7. ✅ Added division/conference analysis

Based on fixes from NCAA backtester bug discovery.
See NFL_BACKTESTER_BUGS.md for detailed bug documentation.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from nfl_system.game_prioritization import NFLGamePrioritizer
from nfl_system.gold_standard_nfl_config import NFLGoldStandardConfig, get_nfl_config

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    seasons: List[str]
    bankroll_start: float = 10000.0
    unit_size: float = 100.0
    max_unit_multiplier: float = 3.0
    max_exposure: float = 0.12
    fractional_kelly: float = 0.25  # 25% Kelly for safety
    include_weather: bool = True
    include_sentiment: bool = True
    random_seed: int = 42
    historical_data_dir: Path = field(
        default_factory=lambda: Path("data/football/historical/nfl")
    )
    historical_file_patterns: Tuple[str, ...] = (
        "nfl_{season}.csv",
        "nfl_{season}.json",
        "season_{season}.csv",
        "season_{season}.json",
    )


class NFLBacktesterImproved:
    """
    Comprehensive NFL backtesting engine with bug fixes.
    
    Critical fixes from NCAA backtester bugs:
    - Proper American odds conversion
    - Correct Kelly Criterion implementation  
    - Edge calculation validation
    - Accurate Sharpe ratio calculation
    """

    def __init__(
        self,
        *,
        nfl_config: Optional[NFLGoldStandardConfig] = None,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        self.config = nfl_config or get_nfl_config()
        self.settings = settings or self._default_settings()

        self.thresholds = self.config.thresholds
        self.bankroll_cfg = self.config.bankroll
        self.prioritizer = NFLGamePrioritizer(self.config)
        self.backtest_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.initialized_at = datetime.utcnow()

        logger.info(
            "🧪 NFL Backtester (IMPROVED) initialized for seasons: %s",
            ", ".join(self.settings.seasons),
        )

    def _default_settings(self) -> BacktestSettings:
        current_year = datetime.utcnow().year
        seasons = [str(year) for year in range(current_year - 6, current_year + 1)]
        return BacktestSettings(seasons=seasons)

    async def run_comprehensive_backtest(
        self,
        analyzer_factory: Optional[Callable[[], Any]] = None,
        seasons: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        seasons_to_run = list(seasons) if seasons else list(self.settings.seasons)
        aggregate: Dict[str, Any] = {
            "seasons_tested": len(seasons_to_run),
            "total_games": 0,
            "total_bets": 0,
            "winning_bets": 0,
            "total_profit": 0.0,
            "season_results": [],
        }

        all_bets: List[Dict[str, Any]] = []

        for season in seasons_to_run:
            season_result = await self._run_season_backtest(analyzer_factory, season)
            aggregate["season_results"].append(season_result)
            aggregate["total_games"] += season_result["games_analyzed"]
            aggregate["total_bets"] += season_result["bets_placed"]
            aggregate["winning_bets"] += season_result["winning_bets"]
            aggregate["total_profit"] += season_result["total_profit"]
            all_bets.extend(season_result.get("bet_history", []))

        aggregate["average_roi"] = self._safe_mean(
            [season["season_roi"] for season in aggregate["season_results"]]
        )
        aggregate["win_rate"] = (
            aggregate["winning_bets"] / aggregate["total_bets"]
            if aggregate["total_bets"] > 0
            else 0.0
        )
        aggregate["aggregate_metrics"] = self._summarize_bets(all_bets)
        aggregate["division_analysis"] = self._analyze_by_division(all_bets)
        aggregate["statistical_significance"] = self._test_statistical_significance(all_bets)
        aggregate["summary"] = self._generate_summary(aggregate)

        self.backtest_results.append(aggregate)
        self.performance_metrics = aggregate["summary"]

        return aggregate

    async def _run_season_backtest(
        self,
        analyzer_factory: Optional[Callable[[], Any]],
        season: str,
    ) -> Dict[str, Any]:
        logger.info("🏈 NFL backtesting season %s", season)
        historical_games = self._load_historical_games(season)
        if not historical_games:
            raise ValueError(
                f"No historical NFL data found for season {season}. "
                "Populate data/football/historical/nfl/ before running the backtester."
            )

        prioritized_games = self.prioritizer.optimize_processing_order(historical_games)
        analyzer = analyzer_factory() if analyzer_factory else None

        bankroll = self.settings.bankroll_start
        bets_placed = 0
        winning_bets = 0
        total_profit = 0.0
        bet_history: List[Dict[str, Any]] = []

        for game in prioritized_games:
            edge, confidence, odds = await self._evaluate_game(game, analyzer)
            actual = game.get("actual_result")
            if actual is None:
                raise ValueError(
                    f"Historical record {game.get('game_id')} missing 'actual_result'. "
                    "Ensure NFL dataset includes actual outcomes (1 for win, 0 for loss)."
                )

            if confidence < self.thresholds.confidence_threshold:
                continue
            if edge <= self.thresholds.min_edge_threshold:
                continue
            if bankroll <= 0:
                break

            # 🐛 FIX #3: Proper Kelly Criterion implementation
            stake = self._calculate_kelly_stake(edge, confidence, odds, bankroll)
            stake = min(stake, bankroll * self.settings.max_exposure)
            stake = min(stake, bankroll)

            # 🐛 FIX #1 & #2: Proper odds conversion and profit calculation
            payout_multiplier = self._american_odds_to_multiplier(odds)
            won = bool(actual)
            if won:
                profit = stake * (payout_multiplier - 1.0)  # Correct: subtract stake
            else:
                profit = -stake

            bankroll += profit
            total_profit += profit
            bets_placed += 1
            if won:
                winning_bets += 1

            bet_history.append(
                {
                    "season": season,
                    "game_id": game.get("game_id"),
                    "matchup": f"{game.get('away_team')} @ {game.get('home_team')}",
                    "division": game.get("division", "Unknown"),
                    "conference": game.get("conference", "Unknown"),
                    "stake": stake,
                    "profit": profit,
                    "edge": edge,
                    "confidence": confidence,
                    "odds": odds,
                    "won": won,
                    "bankroll": bankroll,
                }
            )

        season_roi = (
            (total_profit / self.settings.bankroll_start) * 100
            if self.settings.bankroll_start
            else 0.0
        )

        return {
            "season": season,
            "games_analyzed": len(historical_games),
            "bets_placed": bets_placed,
            "winning_bets": winning_bets,
            "win_rate": (winning_bets / bets_placed) if bets_placed else 0.0,
            "total_profit": total_profit,
            "final_bankroll": bankroll,
            "season_roi": season_roi,
            "bankroll_growth": (
                (bankroll - self.settings.bankroll_start) / self.settings.bankroll_start
                if self.settings.bankroll_start
                else 0.0
            ),
            "bet_history": bet_history,
        }

    async def _evaluate_game(self, game: Dict[str, Any], analyzer: Any) -> Tuple[float, float, int]:
        default_edge = float(game.get("edge_value", 0.0))
        default_confidence = float(game.get("confidence", 0.5))
        odds = int(game.get("odds", -110))

        # 🐛 FIX #4: Validate edge from first principles
        edge, confidence = self._validate_and_recalculate_edge(
            default_edge, default_confidence, odds
        )

        if not analyzer:
            return edge, confidence, odds

        try:
            if hasattr(analyzer, "analyze_game"):
                analysis = analyzer.analyze_game(game)
                if asyncio.iscoroutine(analysis) or asyncio.isfuture(analysis):
                    analysis = await analysis
            elif hasattr(analyzer, "run_unified_analysis"):
                analysis = await analyzer.run_unified_analysis(game)
            elif hasattr(analyzer, "system") and hasattr(analyzer.system, "run_unified_analysis"):
                analysis = await analyzer.system.run_unified_analysis(game)
            else:
                analysis = None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Analyzer evaluation failed for %s: %s", game.get("game_id"), exc, exc_info=True)
            analysis = None

        if not analysis:
            return edge, confidence, odds

        edge = self._extract_float(analysis, ["edge", "edge_value", "total_edge"], edge)
        confidence = self._extract_float(
            analysis, ["confidence", "combined_confidence", "model_confidence"], confidence
        )

        if self.settings.include_weather and hasattr(analyzer, "social_weather"):
            try:
                ctx = await analyzer.social_weather.analyze_game_context({**game, "edge_value": edge})
                edge = ctx.get("edge_adjustment", edge)
                confidence = max(confidence, ctx.get("combined_impact_score", confidence))
            except Exception:  # noqa: BLE001
                pass

        if self.settings.include_sentiment and hasattr(analyzer, "sentiment_engine"):
            try:
                sentiment_score = await analyzer.sentiment_engine.score_matchup(game)
                confidence = max(confidence, min(max(sentiment_score, 0.0), 1.0))
            except Exception:  # noqa: BLE001
                pass

        odds = int(analysis.get("odds", odds))
        edge = float(edge)
        confidence = float(np.clip(confidence, 0.0, 1.0))

        # Revalidate after analyzer adjustments
        edge, confidence = self._validate_and_recalculate_edge(edge, confidence, odds)

        return edge, confidence, odds

    def _validate_and_recalculate_edge(
        self, edge: float, confidence: float, odds: int
    ) -> Tuple[float, float]:
        """
        🐛 FIX #4: Validate and recalculate edge from first principles.
        Don't trust edge values blindly - recalculate from probabilities.
        """
        market_prob = self._odds_to_probability(odds)
        
        # Treat confidence as predicted probability
        predicted_prob = confidence
        
        # Calculate true edge based on which side we're betting
        if predicted_prob > 0.5:
            # Betting on favorite/likely winner
            true_edge = predicted_prob - market_prob
        else:
            # Betting on underdog
            true_edge = (1 - predicted_prob) - (1 - market_prob)
        
        # Sanity check: edges > 30% are suspicious
        if abs(true_edge) > 0.30:
            logger.warning(
                f"Suspicious edge {true_edge:.1%} (market: {market_prob:.1%}, predicted: {predicted_prob:.1%})"
            )
            return 0.0, confidence
        
        return max(0.0, true_edge), confidence

    def _calculate_kelly_stake(
        self, edge: float, confidence: float, odds: int, bankroll: float
    ) -> float:
        """
        🐛 FIX #3: Proper Kelly Criterion implementation.
        
        Kelly formula: f = (p*b - (1-p)) / b
        where:
        - p = win probability 
        - b = net odds (decimal_odds - 1)
        - f = fraction of bankroll to bet
        """
        # Convert odds to decimal
        decimal_odds = self._american_odds_to_multiplier(odds)
        b = decimal_odds - 1.0  # Net odds
        
        # Win probability = market probability + edge
        market_prob = self._odds_to_probability(odds)
        p = min(0.99, max(0.01, market_prob + edge))  # Clamp to avoid division issues
        
        # Kelly fraction
        kelly_fraction = max(0.0, (p * b - (1 - p)) / b)
        
        # Use fractional Kelly for safety (default 25%)
        fractional_kelly = self.settings.fractional_kelly
        stake = bankroll * kelly_fraction * fractional_kelly
        
        return max(1.0, stake)  # Minimum $1 bet

    def _load_historical_games(self, season: str) -> List[Dict[str, Any]]:
        data_dir = self.settings.historical_data_dir
        if not data_dir.exists():
            raise FileNotFoundError(
                f"NFL historical data directory '{data_dir}' not found. "
                "Populate it with real betting history (CSV/JSON/Parquet)."
            )

        candidates: List[Path] = []
        for pattern in self.settings.historical_file_patterns:
            candidate = data_dir / pattern.format(season=season)
            if candidate.exists():
                candidates.append(candidate)

        if not candidates:
            candidates.extend(sorted(data_dir.glob(f"*{season}*.csv")))
            candidates.extend(sorted(data_dir.glob(f"*{season}*.json")))
            candidates.extend(sorted(data_dir.glob(f"*{season}*.parquet")))

        if not candidates:
            return []

        file_path = candidates[0]
        suffix = file_path.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(file_path)
        elif suffix == ".json":
            with file_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            df = pd.DataFrame(payload)
        elif suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported historical data format: {file_path}")

        required_columns = {
            "game_id",
            "home_team",
            "away_team",
            "edge_value",
            "confidence",
            "odds",
            "actual_result",
        }
        missing = required_columns.difference(df.columns)
        if missing:
            raise ValueError(
                f"Historical dataset {file_path} missing required columns: {sorted(missing)}. "
                "Ensure edge, confidence, odds, and actual result columns are present."
            )

        df = df.replace({np.nan: None})
        return df.to_dict(orient="records")

    def _summarize_bets(self, bets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        if not bets:
            return {
                "median_edge": 0.0,
                "median_confidence": 0.0,
                "max_drawdown": 0.0,
                "sharpe_ratio": 0.0,
                "volatility": 0.0,
                "total_profit": 0.0,
            }

        df = pd.DataFrame(bets)
        bankroll_curve = df["bankroll"].tolist() if "bankroll" in df else []
        max_drawdown = self._calculate_max_drawdown(bankroll_curve) if bankroll_curve else 0.0
        sharpe_ratio = self._calculate_sharpe_ratio(df["profit"])
        volatility = float(np.nan_to_num(df["profit"].std(ddof=1), nan=0.0))
        total_profit = float(df["profit"].sum())

        return {
            "median_edge": float(df["edge"].median()),
            "median_confidence": float(df["confidence"].median()),
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "total_profit": total_profit,
        }

    def _analyze_by_division(self, bets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance by NFL division (NFC East, AFC West, etc.)"""
        if not bets:
            return {}
        
        df = pd.DataFrame(bets)
        
        division_stats = {}
        for division in df["division"].unique():
            div_bets = df[df["division"] == division]
            if len(div_bets) == 0:
                continue
            
            wins = div_bets["won"].sum()
            total = len(div_bets)
            profit = div_bets["profit"].sum()
            
            division_stats[division] = {
                "bets": total,
                "wins": int(wins),
                "win_rate": wins / total if total > 0 else 0.0,
                "profit": float(profit),
                "avg_profit_per_bet": float(profit / total) if total > 0 else 0.0,
            }
        
        return division_stats

    def _test_statistical_significance(self, bets: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """
        🆕 Test if results are statistically significant (not just luck).
        Uses t-test to compare profits to zero.
        """
        if not bets or len(bets) < 10:
            return {
                "significant": False,
                "p_value": 1.0,
                "message": "Not enough bets for statistical testing"
            }
        
        df = pd.DataFrame(bets)
        profits = df["profit"].values
        
        # One-sample t-test: are profits significantly different from 0?
        t_statistic, p_value = stats.ttest_1samp(profits, 0)
        
        # p < 0.05 means statistically significant (95% confidence)
        significant = p_value < 0.05
        
        return {
            "significant": significant,
            "p_value": float(p_value),
            "t_statistic": float(t_statistic),
            "message": (
                "✅ Results are statistically significant (p < 0.05)" if significant
                else "⚠️ Results could be due to random variance (p >= 0.05)"
            ),
            "confidence_level": 95 if p_value < 0.05 else (90 if p_value < 0.10 else 0),
        }

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        avg_roi = results.get("average_roi", 0.0)
        win_rate = results.get("win_rate", 0.0)
        aggregate_metrics = results.get("aggregate_metrics", {})
        stat_sig = results.get("statistical_significance", {})

        if avg_roi > 15:
            grade = "EXCELLENT"
        elif avg_roi > 8:
            grade = "GOOD"
        elif avg_roi > 2:
            grade = "AVERAGE"
        else:
            grade = "POOR"

        return {
            "performance_grade": grade,
            "average_roi": avg_roi,
            "win_rate": win_rate,
            "total_games": results.get("total_games", 0),
            "total_bets": results.get("total_bets", 0),
            "statistically_significant": stat_sig.get("significant", False),
            "p_value": stat_sig.get("p_value", 1.0),
            "recommendations": self._generate_recommendations(results),
            "risk_assessment": self._assess_risk(aggregate_metrics),
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        recommendations: List[str] = []
        avg_roi = results.get("average_roi", 0.0)
        win_rate = results.get("win_rate", 0.0)
        total_bets = results.get("total_bets", 0)
        stat_sig = results.get("statistical_significance", {})

        if not stat_sig.get("significant", False):
            recommendations.append(
                "⚠️ Results not statistically significant. Need more data or better edge detection."
            )
        
        if win_rate < 0.524:  # Need 52.4% to beat -110 vig
            recommendations.append(
                "Win rate under 52.4%. Tighten confidence thresholds or improve model inputs."
            )
        if avg_roi < 5:
            recommendations.append("ROI modest. Revisit edge calibration and staking strategy.")
        if total_bets < results.get("total_games", 0) * 0.15:
            recommendations.append(
                "Bet volume low. Consider reducing min edge threshold to capture more opportunities."
            )

        # Division-specific recommendations
        div_analysis = results.get("division_analysis", {})
        if div_analysis:
            best_division = max(div_analysis.items(), key=lambda x: x[1]["profit"], default=(None, None))
            if best_division[0]:
                recommendations.append(
                    f"🎯 Focus on {best_division[0]} - most profitable division "
                    f"(${best_division[1]['profit']:.0f} profit)"
                )

        return recommendations

    def _assess_risk(self, aggregate_metrics: Dict[str, Any]) -> Dict[str, Any]:
        volatility = aggregate_metrics.get("volatility", 0.0)
        sharpe_ratio = aggregate_metrics.get("sharpe_ratio", 0.0)
        max_drawdown = aggregate_metrics.get("max_drawdown", 0.0)

        if max_drawdown < 0.08 and sharpe_ratio >= 1.8:
            risk_level = "LOW"
        elif max_drawdown < 0.18 and sharpe_ratio >= 1.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"

        return {
            "risk_level": risk_level,
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
        }

    def display_results(self, results: Dict[str, Any]) -> None:
        print("\n🏆 NFL BACKTEST RESULTS (IMPROVED - BUG FIXES APPLIED)")
        print("=" * 70)
        print(f"Seasons Tested: {results['seasons_tested']}")
        print(f"Total Games Simulated: {results['total_games']}")
        print(f"Total Bets Placed: {results['total_bets']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Average ROI: {results['average_roi']:.2f}%")
        print(f"Total Profit: ${results['total_profit']:.2f}")

        # Statistical significance
        stat_sig = results.get("statistical_significance", {})
        print(f"\n📊 Statistical Significance:")
        print(f"   {stat_sig.get('message', 'Unknown')}")
        print(f"   P-Value: {stat_sig.get('p_value', 1.0):.4f}")

        summary = results["summary"]
        risk = summary["risk_assessment"]
        print(f"\n📊 Performance Grade: {summary['performance_grade']}")
        print(
            f"Risk Level: {risk['risk_level']} | "
            f"Max Drawdown: {risk['max_drawdown']:.1%} | "
            f"Sharpe Ratio: {risk['sharpe_ratio']:.2f}"
        )

        # Division analysis
        div_analysis = results.get("division_analysis", {})
        if div_analysis:
            print("\n🏈 Performance by Division:")
            sorted_divs = sorted(
                div_analysis.items(), 
                key=lambda x: x[1]["profit"], 
                reverse=True
            )
            for div, stats in sorted_divs[:5]:  # Top 5
                print(
                    f"   {div}: {stats['bets']} bets | "
                    f"Win Rate: {stats['win_rate']:.1%} | "
                    f"Profit: ${stats['profit']:.0f}"
                )

        print("\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")

        print("\n📅 Season-by-Season Summary:")
        for season in results["season_results"]:
            print(
                f"   {season['season']}: bets={season['bets_placed']} | "
                f"ROI={season['season_roi']:.2f}% | "
                f"Win Rate={season['win_rate']:.1%} | "
                f"Profit=${season['total_profit']:.2f}"
            )

    @staticmethod
    def _extract_float(payload: Dict[str, Any], keys: Sequence[str], default: float) -> float:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
        return float(default)

    @staticmethod
    def _safe_mean(values: Sequence[float]) -> float:
        filtered = [v for v in values if v is not None]
        if not filtered:
            return 0.0
        return float(np.mean(filtered))

    @staticmethod
    def _american_odds_to_multiplier(odds: int) -> float:
        """
        🐛 FIX #1: Proper American odds to decimal conversion.
        
        Returns total payout multiplier (including original stake).
        
        Examples:
        - +200 → 3.0x (bet $100, win $200, get back $300)
        - -110 → 1.909x (bet $110, win $100, get back $210)
        """
        if odds >= 0:
            return (odds / 100.0) + 1.0
        else:
            return (100.0 / abs(odds)) + 1.0

    @staticmethod
    def _odds_to_probability(odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds >= 0:
            return 100.0 / (odds + 100.0)
        else:
            return abs(odds) / (abs(odds) + 100.0)

    @staticmethod
    def _calculate_max_drawdown(equity_curve: Sequence[float]) -> float:
        if not equity_curve:
            return 0.0

        peak = equity_curve[0]
        max_drawdown = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            if peak:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
        return float(max_drawdown)

    @staticmethod
    def _calculate_sharpe_ratio(profits: pd.Series) -> float:
        """
        🐛 FIX #5: Correct Sharpe ratio calculation.
        
        For per-bet Sharpe, don't annualize (no sqrt(n) term).
        This gives a fair risk-adjusted return metric.
        """
        if profits.empty or len(profits) < 2:
            return 0.0

        std_dev = float(np.nan_to_num(profits.std(ddof=1), nan=0.0))
        if std_dev == 0:
            return 0.0

        mean_profit = float(np.nan_to_num(profits.mean(), nan=0.0))
        
        # Per-bet Sharpe (no annualization)
        return float(mean_profit / std_dev)


__all__ = ["NFLBacktesterImproved", "BacktestSettings"]
