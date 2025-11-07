#!/usr/bin/env python3
"""
NCAA/College Football Backtesting Harness
==========================================

Provides an asynchronous backtester for the college football intelligence
pipeline. This implementation mirrors the NFL backtester but uses NCAA-specific
configuration, prioritizer, and analyzer components.
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

from college_football_system.game_prioritization import GamePrioritizer
from college_football_system.main_analyzer import CollegeFootballMainAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class BacktestSettings:
    seasons: List[str]
    bankroll_start: float = 10000.0
    unit_size: float = 100.0
    max_unit_multiplier: float = 3.0
    max_exposure: float = 0.12
    include_weather: bool = True
    include_sentiment: bool = True
    random_seed: int = 42
    historical_data_dir: Path = field(
        default_factory=lambda: Path("data/football/historical/ncaaf")
    )
    historical_file_patterns: Tuple[str, ...] = (
        "ncaaf_{season}.csv",
        "ncaaf_{season}.json",
        "college_{season}.csv",
        "college_{season}.json",
        "season_{season}.csv",
        "season_{season}.json",
    )
    # NCAA-specific thresholds
    confidence_threshold: float = 0.60
    min_edge_threshold: float = 0.03


class NCAABacktester:
    """Comprehensive NCAA/College Football backtesting engine."""

    def __init__(
        self,
        *,
        settings: Optional[BacktestSettings] = None,
    ) -> None:
        self.settings = settings or self._default_settings()

        # Initialize NCAA-specific components
        self.prioritizer = GamePrioritizer()
        self.backtest_results: List[Dict[str, Any]] = []
        self.performance_metrics: Dict[str, Any] = {}
        self.initialized_at = datetime.utcnow()

        logger.info(
            "🧪 NCAA Backtester initialized for seasons: %s",
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
        """Run comprehensive backtest across multiple seasons."""
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
        aggregate["summary"] = self._generate_summary(aggregate)

        self.backtest_results.append(aggregate)
        self.performance_metrics = aggregate["summary"]

        return aggregate

    async def _run_season_backtest(
        self,
        analyzer_factory: Optional[Callable[[], Any]],
        season: str,
    ) -> Dict[str, Any]:
        """Run backtest for a single season."""
        logger.info("🏈 NCAA backtesting season %s", season)
        historical_games = self._load_historical_games(season)
        if not historical_games:
            raise ValueError(
                f"No historical NCAA data found for season {season}. "
                "Populate data/football/historical/ncaaf/ before running the backtester."
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
                    "Ensure NCAA dataset includes actual outcomes (1 for win, 0 for loss)."
                )

            if confidence < self.settings.confidence_threshold:
                continue
            if edge <= self.settings.min_edge_threshold:
                continue
            if bankroll <= 0:
                break

            # Kelly criterion with NCAA-specific multipliers
            unit_multiplier = float(np.clip(1.0 + edge * 5.0, 0.5, self.settings.max_unit_multiplier))
            stake = min(bankroll * self.settings.max_exposure, self.settings.unit_size * unit_multiplier)
            stake = min(stake, bankroll)

            payout_multiplier = self._american_odds_to_multiplier(odds)
            won = bool(actual)
            profit = stake * payout_multiplier if won else -stake

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
        """Evaluate a game and return edge, confidence, and odds."""
        default_edge = float(game.get("edge_value", 0.0))
        default_confidence = float(game.get("confidence", 0.5))
        odds = int(game.get("odds", -110))

        if not analyzer:
            return default_edge, default_confidence, odds

        try:
            # Try different analyzer methods
            if hasattr(analyzer, "analyze_game"):
                analysis = analyzer.analyze_game(game)
                if asyncio.iscoroutine(analysis) or asyncio.isfuture(analysis):
                    analysis = await analysis
            elif hasattr(analyzer, "run_comprehensive_analysis"):
                analysis = await analyzer.run_comprehensive_analysis(game)
            elif hasattr(analyzer, "analyze"):
                analysis = await analyzer.analyze(game)
            else:
                analysis = None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Analyzer evaluation failed for %s: %s", game.get("game_id"), exc, exc_info=True)
            analysis = None

        if not analysis:
            return default_edge, default_confidence, odds

        edge = self._extract_float(analysis, ["edge", "edge_value", "total_edge"], default_edge)
        confidence = self._extract_float(
            analysis, ["confidence", "combined_confidence", "model_confidence"], default_confidence
        )

        # Apply weather and sentiment adjustments if available
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

        return edge, confidence, odds

    def _load_historical_games(self, season: str) -> List[Dict[str, Any]]:
        """Load historical game data for a given season."""
        data_dir = self.settings.historical_data_dir
        if not data_dir.exists():
            raise FileNotFoundError(
                f"NCAA historical data directory '{data_dir}' not found. "
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
        """Summarize betting performance metrics."""
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

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary and grading."""
        avg_roi = results.get("average_roi", 0.0)
        win_rate = results.get("win_rate", 0.0)
        aggregate_metrics = results.get("aggregate_metrics", {})

        # NCAA-specific grading thresholds (slightly more aggressive than NFL)
        if avg_roi > 18:
            grade = "EXCELLENT"
        elif avg_roi > 10:
            grade = "GOOD"
        elif avg_roi > 3:
            grade = "AVERAGE"
        else:
            grade = "POOR"

        return {
            "performance_grade": grade,
            "average_roi": avg_roi,
            "win_rate": win_rate,
            "total_games": results.get("total_games", 0),
            "total_bets": results.get("total_bets", 0),
            "recommendations": self._generate_recommendations(results),
            "risk_assessment": self._assess_risk(aggregate_metrics),
        }

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on backtest results."""
        recommendations: List[str] = []
        avg_roi = results.get("average_roi", 0.0)
        win_rate = results.get("win_rate", 0.0)
        total_bets = results.get("total_bets", 0)

        if win_rate < 0.54:
            recommendations.append("Win rate under 54%. Tighten confidence thresholds or improve model inputs.")
        if avg_roi < 6:
            recommendations.append("ROI modest. Revisit edge calibration and staking strategy.")
        if total_bets < results.get("total_games", 0) * 0.18:
            recommendations.append("Bet volume low. Consider reducing min edge threshold to capture more opportunities.")

        recommendations.append("Leverage prioritizer logs to tune conference weights based on profitable conferences.")
        recommendations.append("Focus on Group of Five conferences for potential hidden value.")
        recommendations.append("Consider rivalry game impacts and home field advantage in college football.")
        return recommendations

    def _assess_risk(self, aggregate_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess risk level based on volatility and drawdown metrics."""
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
        """Display comprehensive backtest results."""
        print("\n🏆 NCAA FOOTBALL BACKTEST RESULTS")
        print("=" * 70)
        print(f"Seasons Tested: {results['seasons_tested']}")
        print(f"Total Games Simulated: {results['total_games']}")
        print(f"Total Bets Placed: {results['total_bets']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Average ROI: {results['average_roi']:.2f}%")
        print(f"Total Profit: ${results['total_profit']:.2f}")

        summary = results["summary"]
        risk = summary["risk_assessment"]
        print(f"\n📊 Performance Grade: {summary['performance_grade']}")
        print(f"Risk Level: {risk['risk_level']} | Max Drawdown: {risk['max_drawdown']:.1%} | Sharpe Ratio: {risk['sharpe_ratio']:.2f}")

        print("\n💡 Recommendations:")
        for rec in summary["recommendations"]:
            print(f"   • {rec}")

        print("\n📅 Season-by-Season Summary:")
        for season in results["season_results"]:
            print(
                f"   {season['season']}: bets={season['bets_placed']} | ROI={season['season_roi']:.2f}% | "
                f"Win Rate={season['win_rate']:.1%} | Profit=${season['total_profit']:.2f}"
            )

    @staticmethod
    def _extract_float(payload: Dict[str, Any], keys: Sequence[str], default: float) -> float:
        """Extract float value from payload using multiple possible keys."""
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
        """Calculate mean safely, filtering out None values."""
        filtered = [v for v in values if v is not None]
        if not filtered:
            return 0.0
        return float(np.mean(filtered))

    @staticmethod
    def _american_odds_to_multiplier(odds: int) -> float:
        """Convert American odds to payout multiplier."""
        if odds >= 0:
            return odds / 100.0
        return 100.0 / abs(odds)

    @staticmethod
    def _calculate_max_drawdown(equity_curve: Sequence[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
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
        """Calculate Sharpe ratio from profit series."""
        if profits.empty:
            return 0.0

        std_dev = float(np.nan_to_num(profits.std(ddof=1), nan=0.0))
        if std_dev == 0:
            return 0.0

        mean_profit = float(np.nan_to_num(profits.mean(), nan=0.0))
        return float((mean_profit / std_dev) * np.sqrt(len(profits))) if std_dev else 0.0


__all__ = ["NCAABacktester", "BacktestSettings"]
