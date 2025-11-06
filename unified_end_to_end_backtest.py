#!/usr/bin/env python3
"""
Unified End-to-End NFL Backtesting Workflow
===========================================

Replicates the Colab pipeline locally:

1. Collects real NFL game data (default 2015-2024) via the ESPN API helpers.
2. Runs the NarrativeIntegratedAICouncil prediction engine across every game.
3. Grades the predictions to produce backtest metrics.
4. Analyzes referee crew correlations (home bias, penalties, win rates).
5. Writes consolidated reports to ``reports/backtesting``.

Run:
    python unified_end_to_end_backtest.py \
        --start-year 2015 \
        --end-year 2024 \
        --include-backtests
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from betting_types import GameData
from collect_historical_nfl import (
    fetch_nfl_historical_seasons,
    save_training_data,
)
from unified_betting_intelligence import NarrativeIntegratedAICouncil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATA_DIR = Path("data/football/historical")
REPORT_DIR = Path("reports/backtesting")


def ensure_historical_data(start_year: int, end_year: int, force: bool) -> Path:
    """Fetch historical data if it isn't already present."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DATA_DIR / f"nfl_{start_year}_{end_year}_games.json"

    if output_path.exists() and not force:
        logger.info("Using cached historical data at %s", output_path)
        return output_path

    logger.info("Collecting historical NFL data (%s-%s)", start_year, end_year)
    games = fetch_nfl_historical_seasons(start_year, end_year)
    if not games:
        raise RuntimeError("No historical games were collected.")

    save_training_data(games, filename=str(output_path))
    logger.info("Historical data saved to %s", output_path)
    return output_path


def load_games(path: Path) -> List[Dict[str, Any]]:
    """Load games from JSON file."""
    payload = json.loads(path.read_text())
    games = payload.get("games", [])
    logger.info("Loaded %s games across %s seasons", len(games), payload.get("seasons"))
    return games


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_game_payload(game: Dict[str, Any]) -> GameData:
    """Convert raw game dictionary into GameData for predictions."""
    spread = _coerce_float(game.get("spread"), 0.0)
    total = _coerce_float(game.get("total"), 44.5)

    timestamp: datetime
    if game.get("date"):
        try:
            timestamp = datetime.fromisoformat(game["date"].replace("Z", "+00:00"))
        except ValueError:
            timestamp = datetime(game["season"], 1, 1)
    else:
        timestamp = datetime(game["season"], 1, 1)

    injuries = []
    if game.get("home_injury_score", 0) > 0:
        injuries.append(f"{game['home_team']} injuries")
    if game.get("away_injury_score", 0) > 0:
        injuries.append(f"{game['away_team']} injuries")

    game_data: GameData = {
        "game_id": game.get("game_id", f"{game['season']}_{game['week']}_{game['home_team']}"),
        "home_team": game.get("home_team", "HOME"),
        "away_team": game.get("away_team", "AWAY"),
        "spread": spread,
        "total": total,
        "home_ml_odds": -110,
        "away_ml_odds": 110,
        "timestamp": timestamp,
        "referee": game.get("referee_name", "Unknown"),
        "is_trap_game": bool(game.get("is_division_game", False)),
        "injuries": injuries or None,
        "reddit_lean": 0.0,
        "expert_pct_home": 0.5,
        "sharp_public_ml": 0.0,
        "sharp_public_total": 0.0,
        "contrarian_opportunity": 0.0,
        "crowd_roar": float(game.get("attendance", 0) / 75000) if game.get("attendance") else 0.0,
        "spread_model_home_pct": 0.5,
        "total_model_over_pct": 0.5,
        "home_advantage_pct": 0.5 + (game.get("crew_home_bias", 0.0) - 0.5) / 5,
        "division": "division" if game.get("is_division_game") else "non_division",
        "conference": game.get("conference", ""),
        "kickoff_window": "primetime" if game.get("is_primetime") else "standard",
        "weather_tag": "dome" if game.get("is_dome") else "outdoor",
        "travel_profile": "long" if game.get("away_travel_distance", 0) > 750 else "standard",
    }
    return game_data


def run_predictions(games: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate unified council predictions for each game."""
    council = NarrativeIntegratedAICouncil()
    predictions: List[Dict[str, Any]] = []
    enriched_games: List[Dict[str, Any]] = []

    for game in games:
        game_payload = build_game_payload(game)
        prediction = council.make_unified_prediction(game_payload)

        pred_record = {
            "game_id": prediction.game_id,
            "home_team": prediction.home_team,
            "away_team": prediction.away_team,
            "season": game.get("season"),
            "week": game.get("week"),
            "spread_pick": prediction.spread_prediction.pick,
            "spread_line": prediction.spread_prediction.adjusted_line,
            "spread_confidence": prediction.spread_prediction.confidence,
            "spread_edge": prediction.spread_prediction.edge,
            "total_pick": prediction.total_prediction.pick,
            "total_line": prediction.total_prediction.adjusted_line,
            "total_confidence": prediction.total_prediction.confidence,
            "total_edge": prediction.total_prediction.edge,
            "ml_pick": prediction.moneyline_prediction.pick,
            "ml_confidence": prediction.moneyline_prediction.confidence,
            "ml_edge": prediction.moneyline_prediction.edge,
            "confidence": prediction.confidence,
            "risk_level": prediction.risk_level.value,
            "edge_signals": prediction.edge_signals,
            "referee_name": game.get("referee_name"),
            "crew_id": game.get("crew_id"),
            "crew_home_bias": game.get("crew_home_bias"),
            "crew_penalties_avg": game.get("crew_penalties_avg"),
            "crew_variance": game.get("crew_variance"),
        }

        enriched_game = dict(game)
        enriched_game["prediction"] = pred_record

        predictions.append(pred_record)
        enriched_games.append(enriched_game)

    logger.info("Generated predictions for %s games", len(predictions))
    return predictions, enriched_games


def _grade_spread(game: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[str, float]:
    spread = _coerce_float(game.get("spread"), None)
    if spread is None:
        return "NO_DATA", 0.0

    margin = _coerce_float(game.get("home_score"), 0) - _coerce_float(game.get("away_score"), 0)
    pick = prediction["spread_pick"]
    if margin == spread:
        return "PUSH", 0.0
    if pick == "home":
        return ("WIN", 0.91) if margin > spread else ("LOSS", -1.0)
    else:
        return ("WIN", 0.91) if margin < spread else ("LOSS", -1.0)


def _grade_total(game: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[str, float]:
    total = _coerce_float(game.get("total"), None)
    if total is None:
        return "NO_DATA", 0.0

    final_total = _coerce_float(game.get("home_score"), 0) + _coerce_float(game.get("away_score"), 0)
    pick = prediction["total_pick"]
    if final_total == total:
        return "PUSH", 0.0
    if pick == "over":
        return ("WIN", 0.91) if final_total > total else ("LOSS", -1.0)
    else:
        return ("WIN", 0.91) if final_total < total else ("LOSS", -1.0)


def _grade_moneyline(game: Dict[str, Any], prediction: Dict[str, Any]) -> Tuple[str, float]:
    home_score = _coerce_float(game.get("home_score"), None)
    away_score = _coerce_float(game.get("away_score"), None)
    if home_score is None or away_score is None:
        return "NO_DATA", 0.0
    winner = "home" if home_score > away_score else "away"
    return ("WIN", 1.0) if prediction["ml_pick"] == winner else ("LOSS", -1.0)


def grade_predictions(enriched_games: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    graded: List[Dict[str, Any]] = []
    for game in enriched_games:
        prediction = game["prediction"]
        spread_result, spread_profit = _grade_spread(game, prediction)
        total_result, total_profit = _grade_total(game, prediction)
        ml_result, ml_profit = _grade_moneyline(game, prediction)

        record = {
            **prediction,
            "spread_result": spread_result,
            "spread_profit": spread_profit,
            "total_result": total_result,
            "total_profit": total_profit,
            "ml_result": ml_result,
            "ml_profit": ml_profit,
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
            "home_margin": _coerce_float(game.get("home_score"), 0)
            - _coerce_float(game.get("away_score"), 0),
        }
        graded.append(record)
    return graded


def summarize_metrics(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(graded)
    if df.empty:
        return {}

    def win_rate(series: pd.Series) -> float:
        total = (series != "NO_DATA").sum()
        if total == 0:
            return 0.0
        return (series == "WIN").sum() / total

    summary = {
        "games": int(len(df)),
        "spread_win_rate": float(win_rate(df["spread_result"])),
        "spread_roi": float(df["spread_profit"].sum() / max(1, (df["spread_result"] != "NO_DATA").sum())),
        "total_win_rate": float(win_rate(df["total_result"])),
        "total_roi": float(df["total_profit"].sum() / max(1, (df["total_result"] != "NO_DATA").sum())),
        "moneyline_win_rate": float(win_rate(df["ml_result"])),
        "moneyline_roi": float(df["ml_profit"].sum() / max(1, (df["ml_result"] != "NO_DATA").sum())),
        "average_confidence": float(df["confidence"].mean()),
        "risk_distribution": df["risk_level"].value_counts(normalize=True).to_dict(),
    }
    return summary


def analyze_referee_bias(graded: List[Dict[str, Any]]) -> Dict[str, Any]:
    df = pd.DataFrame(graded)
    if df.empty:
        return {}

    df["spread_win"] = (df["spread_result"] == "WIN").astype(int)
    df["ml_win"] = (df["ml_result"] == "WIN").astype(int)

    grouped = (
        df.groupby("referee_name")
        .agg(
            games=("game_id", "count"),
            spread_win_rate=("spread_win", "mean"),
            spread_roi=("spread_profit", "mean"),
            moneyline_win_rate=("ml_win", "mean"),
            avg_margin=("home_margin", "mean"),
            avg_bias=("crew_home_bias", "mean"),
            avg_penalties=("crew_penalties_avg", "mean"),
        )
        .reset_index()
    )

    bias_corr = grouped["avg_bias"].corr(grouped["avg_margin"])
    penalty_corr = grouped["avg_penalties"].corr(grouped["spread_win_rate"])

    top_bias = grouped.sort_values("avg_bias", ascending=False).head(5).to_dict(orient="records")
    bottom_bias = grouped.sort_values("avg_bias", ascending=True).head(5).to_dict(orient="records")

    return {
        "records": grouped.to_dict(orient="records"),
        "bias_vs_margin_corr": float(bias_corr) if not np.isnan(bias_corr) else None,
        "penalties_vs_spread_win_corr": float(penalty_corr) if not np.isnan(penalty_corr) else None,
        "top_home_bias_crews": top_bias,
        "bottom_home_bias_crews": bottom_bias,
    }


def save_reports(
    predictions: List[Dict[str, Any]],
    graded: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    referee_report: Dict[str, Any],
    start_year: int,
    end_year: int,
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

    report_payload = {
        "generated_at": timestamp,
        "seasons": {"start": start_year, "end": end_year},
        "metrics": metrics,
        "referee_analysis": referee_report,
    }
    report_path = REPORT_DIR / f"unified_backtest_summary_{timestamp}.json"
    report_path.write_text(json.dumps(report_payload, indent=2))

    detail_path = REPORT_DIR / f"unified_backtest_predictions_{timestamp}.json"
    detail_payload = {
        "generated_at": timestamp,
        "predictions": predictions,
        "graded": graded,
    }
    detail_path.write_text(json.dumps(detail_payload, indent=2))

    logger.info("Summary saved to %s", report_path)
    logger.info("Detailed predictions saved to %s", detail_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified end-to-end NFL backtesting workflow")
    parser.add_argument("--start-year", type=int, default=2015, help="First season to include")
    parser.add_argument("--end-year", type=int, default=datetime.now().year - 1, help="Last season to include")
    parser.add_argument("--force-refresh", action="store_true", help="Re-download historical data even if cached")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip data scraping and use existing cached data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.skip_scrape:
        data_path = DATA_DIR / f"nfl_{args.start_year}_{args.end_year}_games.json"
        if not data_path.exists():
            raise FileNotFoundError(f"No cached data at {data_path}. Remove --skip-scrape or collect first.")
    else:
        data_path = ensure_historical_data(args.start_year, args.end_year, force=args.force_refresh)

    games = load_games(data_path)
    predictions, enriched_games = run_predictions(games)
    graded = grade_predictions(enriched_games)
    metrics = summarize_metrics(graded)
    referee_report = analyze_referee_bias(graded)
    save_reports(predictions, graded, metrics, referee_report, args.start_year, args.end_year)

    logger.info("Workflow complete. Spread win rate %.1f%%, ROI %.2f",
                metrics.get("spread_win_rate", 0.0) * 100,
                metrics.get("spread_roi", 0.0))


if __name__ == "__main__":
    main()
