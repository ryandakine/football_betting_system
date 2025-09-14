#!/usr/bin/env python3
"""
Daily Prediction Scheduler
==========================
Automatically runs the daily prediction system every day to ensure continuous learning.
This scheduler ensures the system makes predictions on every MLB game daily,
regardless of betting value, for continuous learning and testing.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import schedule

from daily_prediction_system import DailyPredictionSystem
from self_learning_system import SelfLearningSystem

logger = logging.getLogger(__name__)


class DailyPredictionScheduler:
    """Scheduler for running daily predictions automatically."""

    def __init__(self, daily_system: DailyPredictionSystem):
        self.daily_system = daily_system
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Scheduler configuration
        self.prediction_time = "09:00"  # Run predictions at 9 AM
        self.outcome_time = "23:00"  # Record outcomes at 11 PM
        self.is_running = False

    def start_scheduler(self):
        """Start the automated scheduler."""
        logger.info("Starting Daily Prediction Scheduler")

        # Schedule daily predictions
        schedule.every().day.at(self.prediction_time).do(self._run_daily_predictions)

        # Schedule daily outcome recording
        schedule.every().day.at(self.outcome_time).do(self._run_daily_outcomes)

        # Schedule weekly model retraining
        schedule.every().sunday.at("02:00").do(self._run_weekly_retraining)

        # Schedule monthly performance review
        schedule.every().month.at("01:00").do(self._run_monthly_review)

        self.is_running = True

        logger.info(f"Scheduler started with the following schedule:")
        logger.info(f"  Daily predictions: {self.prediction_time}")
        logger.info(f"  Daily outcomes: {self.outcome_time}")
        logger.info(f"  Weekly retraining: Sunday 2:00 AM")
        logger.info(f"  Monthly review: 1st of month 1:00 AM")

        # Run the scheduler
        while self.is_running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def stop_scheduler(self):
        """Stop the scheduler."""
        logger.info("Stopping Daily Prediction Scheduler")
        self.is_running = False

    def _run_daily_predictions(self):
        """Run daily predictions for today."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"Running daily predictions for {today}")

            # Run predictions asynchronously
            asyncio.run(self.daily_system.make_daily_predictions(today))

            # Get and log summary
            summary = self.daily_system.get_prediction_summary(today)

            logger.info(f"Daily predictions completed for {today}:")
            logger.info(f"  Total predictions: {summary.get('total_predictions', 0)}")
            logger.info(f"  Value bets found: {summary.get('value_bets', 0)}")
            logger.info(
                f"  Average confidence: {summary.get('average_confidence', 0):.2f}"
            )
            logger.info(f"  Average edge: {summary.get('average_edge', 0):.3f}")

            # Log to file
            self._log_daily_summary(today, summary)

        except Exception as e:
            logger.error(f"Error running daily predictions: {e}")
            self._log_error("daily_predictions", str(e))

    def _run_daily_outcomes(self):
        """Record outcomes for today's games."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            logger.info(f"Recording daily outcomes for {today}")

            # Record outcomes asynchronously
            asyncio.run(self.daily_system.record_daily_outcomes(today))

            logger.info(f"Daily outcomes recorded for {today}")

        except Exception as e:
            logger.error(f"Error recording daily outcomes: {e}")
            self._log_error("daily_outcomes", str(e))

    def _run_weekly_retraining(self):
        """Run weekly model retraining."""
        try:
            logger.info("Running weekly model retraining")

            # Trigger model retraining
            self.daily_system.learning_system.retrain_with_new_data()

            # Get updated learning summary
            summary = self.daily_system.learning_system.get_learning_summary()

            logger.info("Weekly retraining completed:")
            logger.info(
                f"  Total predictions: {summary['overall_metrics']['total_predictions']}"
            )
            logger.info(f"  Current accuracy: {summary['overall_metrics']['accuracy']}")
            logger.info(f"  Current ROI: {summary['overall_metrics']['roi']}")

            # Log to file
            self._log_weekly_summary(summary)

        except Exception as e:
            logger.error(f"Error running weekly retraining: {e}")
            self._log_error("weekly_retraining", str(e))

    def _run_monthly_review(self):
        """Run monthly performance review."""
        try:
            logger.info("Running monthly performance review")

            # Get learning summary
            summary = self.daily_system.learning_system.get_learning_summary()

            # Calculate monthly metrics
            monthly_metrics = self._calculate_monthly_metrics()

            logger.info("Monthly review completed:")
            logger.info(f"  Monthly accuracy: {monthly_metrics.get('accuracy', 0):.1%}")
            logger.info(f"  Monthly ROI: {monthly_metrics.get('roi', 0):.1f}%")
            logger.info(
                f"  Predictions this month: {monthly_metrics.get('predictions', 0)}"
            )

            # Log to file
            self._log_monthly_summary(summary, monthly_metrics)

        except Exception as e:
            logger.error(f"Error running monthly review: {e}")
            self._log_error("monthly_review", str(e))

    def _calculate_monthly_metrics(self) -> dict:
        """Calculate metrics for the current month."""
        try:
            # This is a simplified calculation - you can enhance it
            current_month = datetime.now().strftime("%Y-%m")

            # Get predictions for current month
            prediction_files = list(
                Path("predictions").glob(f"predictions_{current_month}-*.json")
            )

            total_predictions = 0
            total_correct = 0
            total_profit = 0

            for file in prediction_files:
                with open(file) as f:
                    data = json.load(f)
                    predictions = data.get("predictions", [])
                    total_predictions += len(predictions)

            # Get learning system metrics
            summary = self.daily_system.learning_system.get_learning_summary()

            return {
                "month": current_month,
                "predictions": total_predictions,
                "accuracy": summary["overall_metrics"]["accuracy"],
                "roi": summary["overall_metrics"]["roi"],
                "total_profit": summary["overall_metrics"]["total_profit"],
            }

        except Exception as e:
            logger.error(f"Error calculating monthly metrics: {e}")
            return {}

    def _log_daily_summary(self, date: str, summary: dict):
        """Log daily summary to file."""
        log_file = self.log_dir / f"daily_summary_{date}.log"

        with open(log_file, "w") as f:
            f.write(f"Daily Prediction Summary - {date}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Total predictions: {summary.get('total_predictions', 0)}\n")
            f.write(f"Value bets found: {summary.get('value_bets', 0)}\n")
            f.write(f"Average confidence: {summary.get('average_confidence', 0):.2f}\n")
            f.write(f"Average edge: {summary.get('average_edge', 0):.3f}\n")

            if "top_value_bets" in summary:
                f.write("\nTop Value Bets:\n")
                for bet in summary["top_value_bets"][:5]:
                    f.write(
                        f"  {bet['home_team']} vs {bet['away_team']}: "
                        f"{bet['predicted_winner']} (confidence: {bet['confidence']:.2f}, "
                        f"edge: {bet['edge']:.3f})\n"
                    )

    def _log_weekly_summary(self, summary: dict):
        """Log weekly summary to file."""
        week_start = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
        week_end = datetime.now().strftime("%Y-%m-%d")

        log_file = self.log_dir / f"weekly_summary_{week_start}_to_{week_end}.log"

        with open(log_file, "w") as f:
            f.write(f"Weekly Learning Summary - {week_start} to {week_end}\n")
            f.write("=" * 50 + "\n")
            f.write(
                f"Total predictions: {summary['overall_metrics']['total_predictions']}\n"
            )
            f.write(f"Current accuracy: {summary['overall_metrics']['accuracy']}\n")
            f.write(f"Current ROI: {summary['overall_metrics']['roi']}\n")
            f.write(
                f"Total profit: ${summary['overall_metrics']['total_profit']:.2f}\n"
            )
            f.write(f"Recent trend: {summary['recent_trend']}\n")

            if "recommendations" in summary:
                f.write("\nRecommendations:\n")
                for rec in summary["recommendations"]:
                    f.write(f"  - {rec}\n")

    def _log_monthly_summary(self, summary: dict, monthly_metrics: dict):
        """Log monthly summary to file."""
        current_month = datetime.now().strftime("%Y-%m")

        log_file = self.log_dir / f"monthly_summary_{current_month}.log"

        with open(log_file, "w") as f:
            f.write(f"Monthly Performance Review - {current_month}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Monthly predictions: {monthly_metrics.get('predictions', 0)}\n")
            f.write(f"Monthly accuracy: {monthly_metrics.get('accuracy', 0):.1%}\n")
            f.write(f"Monthly ROI: {monthly_metrics.get('roi', 0):.1f}%\n")
            f.write(f"Monthly profit: ${monthly_metrics.get('total_profit', 0):.2f}\n")

            f.write(f"\nOverall System Performance:\n")
            f.write(
                f"Total predictions: {summary['overall_metrics']['total_predictions']}\n"
            )
            f.write(f"Overall accuracy: {summary['overall_metrics']['accuracy']}\n")
            f.write(f"Overall ROI: {summary['overall_metrics']['roi']}\n")
            f.write(f"System trend: {summary['recent_trend']}\n")

    def _log_error(self, operation: str, error: str):
        """Log errors to file."""
        log_file = self.log_dir / f"errors.log"

        with open(log_file, "a") as f:
            timestamp = datetime.now().isoformat()
            f.write(f"[{timestamp}] {operation}: {error}\n")

    def run_manual_prediction(self, date: str = None):
        """Run predictions manually for a specific date."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running manual predictions for {date}")

        try:
            # Run predictions
            asyncio.run(self.daily_system.make_daily_predictions(date))

            # Get summary
            summary = self.daily_system.get_prediction_summary(date)

            print(f"\nManual Prediction Summary for {date}:")
            print(f"Total predictions: {summary.get('total_predictions', 0)}")
            print(f"Value bets found: {summary.get('value_bets', 0)}")
            print(f"Average confidence: {summary.get('average_confidence', 0):.2f}")
            print(f"Average edge: {summary.get('average_edge', 0):.3f}")

            if summary.get("top_value_bets"):
                print("\nTop Value Bets:")
                for bet in summary["top_value_bets"][:3]:
                    print(
                        f"  {bet['home_team']} vs {bet['away_team']}: "
                        f"{bet['predicted_winner']} (confidence: {bet['confidence']:.2f}, "
                        f"edge: {bet['edge']:.3f})"
                    )

        except Exception as e:
            logger.error(f"Error running manual prediction: {e}")
            print(f"Error: {e}")

    def run_manual_outcome_recording(self, date: str = None):
        """Record outcomes manually for a specific date."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Running manual outcome recording for {date}")

        try:
            # Record outcomes
            asyncio.run(self.daily_system.record_daily_outcomes(date))

            print(f"Outcomes recorded for {date}")

        except Exception as e:
            logger.error(f"Error running manual outcome recording: {e}")
            print(f"Error: {e}")


# Example usage
if __name__ == "__main__":
    import json
    import os

    # Initialize the system
    learning_system = SelfLearningSystem()
    odds_api_key = os.getenv("ODDS_API_KEY", "your_odds_api_key_here")

    daily_system = DailyPredictionSystem(learning_system, odds_api_key)
    scheduler = DailyPredictionScheduler(daily_system)

    # Run manual prediction for today
    scheduler.run_manual_prediction()

    # Uncomment to start the automated scheduler
    # scheduler.start_scheduler()
