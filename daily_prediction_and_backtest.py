# daily_prediction_and_backtest.py
"""
Daily prediction and backtesting module for the Ultimate Tri-Model MLB Betting System.
Fetches odds, parses markets, integrates AI analysis, and performs backtesting.
Uses asynchronous caching for AI responses.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import aiofiles
import polars as pl

import aiohttp
from ai_cache import AICache
from alerts.manager import AlertSeverity, AlertType, SmartAlertManager
from odds_fetcher import StructuredOdds
from player_performance import PlayerPerformancePredictor
from tri_model_api_config import get_trimodel_api_keys
from trimodel_game_analyzer import AnthropicClient, OpenAIClient, TriModelGameAnalyzer

logger = logging.getLogger(__name__)


async def run_prediction_workflow(
    target_date: str = None,
    alert_manager: SmartAlertManager = None,
    backtest: bool = False,
    historical_data_path: str = "data/historical_game_logs.parquet",
) -> dict:
    """
    Run the daily prediction or backtesting workflow with AI caching.

    Args:
        target_date (str, optional): Date in 'YYYY-MM-DD' format. If None, predicts live.
        alert_manager (SmartAlertManager, optional): Alert manager for notifications.
        backtest (bool): If True, run backtest mode using historical data.
        historical_data_path (str): Path to historical game logs for backtesting.

    Returns:
        Dict: Structured results with predictions and performance metrics.
    """
    api_keys = get_trimodel_api_keys()
    results = {
        "timestamp": datetime.now().isoformat(),
        "games_processed": 0,
        "h2h_bets": [],
        "prop_bets": [],
        "ai_insights": {},
        "prop_insights": {},
        "recommendations": [],
        "performance_metrics": {},
    }

    try:
        async with AICache() as cache:
            from odds_fetcher import OddsFetcher

            fetcher = OddsFetcher(api_key=api_keys["odds_api"])
            if backtest and target_date:
                logger.info(f"📊 Running backtest for {target_date}")
                historical_data = pl.read_parquet(historical_data_path)
                structured_odds = StructuredOdds(
                    games=[
                        GameInfo(**g)
                        for g in historical_data.filter(
                            pl.col("date") == target_date
                        ).to_dicts()
                    ],
                    h2h_bets=[
                        H2HBet(**b)
                        for b in historical_data.filter(
                            pl.col("market") == "h2h"
                        ).to_dicts()
                    ],
                    prop_bets=[
                        PlayerPropBet(**p)
                        for p in historical_data.filter(
                            pl.col("market").isin(["player_props", "batter_home_runs"])
                        ).to_dicts()
                    ],
                )
            else:
                logger.info("📊 Running live prediction")
                structured_odds = await fetcher.get_all_odds_with_props()

            results["games_processed"] = len(structured_odds.games)
            if not structured_odds.games:
                logger.warning("No game data returned. Halting workflow.")
                if alert_manager:
                    await alert_manager.send_alert(
                        alert_type=AlertType.SYSTEM_HEALTH_DEGRADED,
                        severity=AlertSeverity.WARNING,
                        message="No game data returned from odds fetcher.",
                        details={},
                    )
                return results

            results["h2h_bets"] = [asdict(b) for b in structured_odds.h2h_bets]
            results["prop_bets"] = [asdict(p) for p in structured_odds.prop_bets]

            clients = {}
            if key := api_keys.get("claude"):
                clients["claude"] = AnthropicClient(api_key=key)
            if key := api_keys.get("openai"):
                clients["openai"] = OpenAIClient(api_key=key)
            analyzer = TriModelGameAnalyzer(
                clients=clients, alert_manager=alert_manager
            )
            async with aiohttp.ClientSession() as session:
                results["ai_insights"] = await analyzer.analyze_games(
                    session, [g.__dict__ for g in structured_odds.games]
                )

            player_predictor = PlayerPerformancePredictor(model_dir="models")
            for prop in results["prop_bets"]:
                features_df = pl.DataFrame([prop])
                prediction = player_predictor.predict_performance(
                    features_df, metric="home_runs_actual"
                )
                results["prop_insights"][prop["player_name"]] = (
                    prediction.to_dicts()[0] if prediction.height > 0 else {}
                )

            from advanced_testing_system import BettingConfig, TestingBettingSystem

            betting_config = BettingConfig(
                bankroll=1000.0,
                max_total_exposure=120.0,
                min_edge_threshold=0.02,
                min_confidence=0.55,
            )
            betting_system = TestingBettingSystem(betting_config)
            potential_recs = betting_system.process_odds_data(
                results["h2h_bets"] + results["prop_bets"],
                {**results["ai_insights"], **results["prop_insights"]},
            )
            results["recommendations"] = betting_system.apply_portfolio_management(
                potential_recs
            )

            if backtest:
                actual_outcomes = (
                    historical_data.filter(pl.col("date") == target_date)
                    .select(["game_id", "team_name", "outcome"])
                    .to_dicts()
                )
                results["performance_metrics"] = evaluate_backtest(
                    results["recommendations"], actual_outcomes
                )

            from advanced_testing_system import format_recommendations_for_display, save_recommendations_to_file

            formatted_output = format_recommendations_for_display(
                results["recommendations"], betting_config
            )
            print(formatted_output)
            save_recommendations_to_file(
                results["recommendations"],
                betting_config,
                filename=f"ultimate_runs/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            )

    except Exception as e:
        logger.error(f"Workflow failure: {e}")
        if alert_manager:
            await alert_manager.send_alert(
                alert_type=AlertType.SYSTEM_HEALTH_DEGRADED,
                severity=AlertSeverity.CRITICAL,
                message=f"Prediction workflow failed: {e}",
                details={"error": str(e)},
            )
        raise

    return results


def evaluate_backtest(recommendations: list[dict], actual_outcomes: list[dict]) -> dict:
    """
    Evaluate backtest performance by comparing recommendations to actual outcomes.

    Args:
        recommendations (List[Dict]): List of betting recommendations.
        actual_outcomes (List[Dict]): List of actual game outcomes.

    Returns:
        Dict: Performance metrics (e.g., ROI, accuracy).
    """
    outcomes_map = {o["game_id"]: o["outcome"] for o in actual_outcomes}
    total_stake = 0
    total_return = 0
    correct_bets = 0
    total_bets = len(recommendations)

    for rec in recommendations:
        game_id = rec["game_id"]
        stake = rec["bet_amount"]
        odds = rec["odds"]
        team = rec["team_name"]
        total_stake += stake

        if game_id in outcomes_map and outcomes_map[game_id] == team:
            total_return += stake * (odds - 1)  # Net profit
            correct_bets += 1

    roi = ((total_return - total_stake) / total_stake * 100) if total_stake > 0 else 0
    accuracy = (correct_bets / total_bets * 100) if total_bets > 0 else 0

    return {
        "total_bets": total_bets,
        "correct_bets": correct_bets,
        "total_stake": total_stake,
        "total_return": total_return,
        "roi_percentage": roi,
        "accuracy_percentage": accuracy,
    }


if __name__ == "__main__":
    import asyncio

    from alerts.manager import create_production_alert_manager

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    alert_manager = create_production_alert_manager(
        slack_webhook=os.getenv("SLACK_WEBHOOK_URL"),
        email_config={
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "username": os.getenv("SMTP_USERNAME"),
            "password": os.getenv("SMTP_PASSWORD"),
            "from_email": os.getenv("SMTP_FROM_EMAIL"),
            "to_emails": os.getenv("ALERT_EMAILS", "").split(","),
        },
        sms_config={
            "account_sid": os.getenv("TWILIO_ACCOUNT_SID"),
            "auth_token": os.getenv("TWILIO_AUTH_TOKEN"),
            "from_phone": os.getenv("TWILIO_FROM_PHONE"),
            "to_phones": os.getenv("ALERT_PHONES", "").split(","),
        },
    )
    results = asyncio.run(run_prediction_workflow(alert_manager=alert_manager))
    asyncio.run(alert_manager.close())
