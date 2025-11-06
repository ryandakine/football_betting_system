#!/usr/bin/env python3
"""
Daily Reporter
==============
Consolidates system pulse into a daily report:
- MLB daily predictions summary (from DailyPredictionSystem files)
- Learning system summary (SelfLearningSystem or SelfLearningFeedbackSystem)
- NFL live tracking snapshot (from data/nfl_live_tracking.db if present)

Outputs:
- reports/daily/daily_report_YYYY-MM-DD.json
- reports/daily/daily_report_YYYY-MM-DD.txt
"""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


logger = logging.getLogger(__name__)


class DailyReporter:
    def __init__(
        self,
        reports_dir: str = "reports/daily",
        predictions_dir: str = "predictions",
        nfl_db_path: str = "data/nfl_live_tracking.db",
    ):
        self.reports_dir = Path(reports_dir)
        self.predictions_dir = Path(predictions_dir)
        self.nfl_db_path = Path(nfl_db_path)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_mlb_prediction_summary(self, date_str: str) -> Dict[str, Any]:
        """Load MLB daily prediction summary if present."""
        pred_file = self.predictions_dir / f"predictions_{date_str}.json"
        if not pred_file.exists():
            return {
                "status": "missing",
                "message": f"No predictions file for {date_str}",
            }
        try:
            with open(pred_file) as f:
                data = json.load(f)
            predictions = data.get("predictions", [])
            total = len(predictions)
            value_bets = [
                p
                for p in predictions
                if p.get("edge", 0) > 0.02 and p.get("confidence", 0) > 0.6
            ]
            avg_conf = (
                sum(p.get("confidence", 0) for p in predictions) / total
                if total
                else 0.0
            )
            avg_edge = (
                sum(p.get("edge", 0) for p in predictions) / total
                if total
                else 0.0
            )
            return {
                "status": "ok",
                "total_predictions": total,
                "value_bets": len(value_bets),
                "average_confidence": avg_conf,
                "average_edge": avg_edge,
                "top_value_bets": sorted(
                    value_bets, key=lambda x: x.get("edge", 0), reverse=True
                )[:5],
            }
        except Exception as e:
            logger.error(f"Failed to load prediction summary: {e}")
            return {"status": "error", "message": str(e)}

    def _load_learning_summary(self) -> Dict[str, Any]:
        """Try to import learning systems and extract summary/insights."""
        # Prefer SelfLearningSystem (has get_learning_summary), else fallback
        # to SelfLearningFeedbackSystem insights
        try:
            from self_learning_system import SelfLearningSystem  # type: ignore
            learning = SelfLearningSystem()
            summary = learning.get_learning_summary()
            return {
                "status": "ok",
                "summary": summary,
                "source": "SelfLearningSystem",
            }
        except Exception as primary_err:
            try:
                from self_learning_feedback_system import (
                    SelfLearningFeedbackSystem,
                )  # type: ignore
                learning_fb = SelfLearningFeedbackSystem()
                insights = learning_fb.get_enhanced_learning_insights()
                return {
                    "status": "ok",
                    "insights": insights,
                    "source": "SelfLearningFeedbackSystem",
                }
            except Exception as secondary_err:
                logger.warning(
                    (
                        "Learning summary unavailable: "
                        f"{primary_err}; {secondary_err}"
                    )
                )
                return {
                    "status": "missing",
                    "message": "No learning system summary available",
                }

    def _load_nfl_tracking_snapshot(self) -> Dict[str, Any]:
        """Read high-level stats from NFL live tracking DB if exists."""
        if not self.nfl_db_path.exists():
            return {
                "status": "missing",
                "message": "NFL tracking DB not found",
            }
        try:
            with sqlite3.connect(str(self.nfl_db_path)) as conn:
                cursor = conn.cursor()
                # Active live games
                cursor.execute(
                    (
                        "SELECT COUNT(*) FROM live_games "
                        "WHERE status = 'in_progress'"
                    )
                )
                active_games = cursor.fetchone()[0]

                # Predictions in last 24h
                cursor.execute(
                    """
                    SELECT COUNT(*), AVG(prediction_confidence)
                    FROM game_outcomes
                    WHERE recorded_at > datetime('now', '-24 hours')
                    """
                )
                row = cursor.fetchone() or (0, None)
                recent_preds = int(row[0] or 0)
                avg_conf = float(row[1]) if row[1] is not None else 0.0

                # Recent accuracy
                cursor.execute(
                    """
                    SELECT prediction_correct
                    FROM game_outcomes
                    WHERE recorded_at > datetime('now', '-24 hours')
                    """
                )
                flags = [r[0] for r in cursor.fetchall()]
                recent_acc = (
                    sum(1 for f in flags if f == 1) / len(flags)
                    if flags
                    else 0.0
                )

                return {
                    "status": "ok",
                    "active_games": active_games,
                    "recent_predictions": recent_preds,
                    "recent_accuracy": recent_acc,
                    "avg_confidence": avg_conf,
                }
        except Exception as e:
            logger.error(f"Failed to read NFL tracking DB: {e}")
            return {"status": "error", "message": str(e)}

    def generate(self, date_str: str | None = None) -> Dict[str, Any]:
        """Generate and persist daily report for given date (default today)."""
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")

        mlb_section = self._load_mlb_prediction_summary(date_str)
        learning_section = self._load_learning_summary()
        nfl_section = self._load_nfl_tracking_snapshot()

        report: Dict[str, Any] = {
            "date": date_str,
            "generated_at": datetime.now().isoformat(),
            "mlb_predictions": mlb_section,
            "learning": learning_section,
            "nfl_tracking": nfl_section,
        }

        # Save JSON
        json_path = self.reports_dir / f"daily_report_{date_str}.json"
        with open(json_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        # Save text summary
        lines = []
        lines.append(f"📅 Daily System Pulse — {date_str}")
        lines.append("=" * 60)

        # MLB
        lines.append("\n⚾ MLB Predictions:")
        if mlb_section.get("status") == "ok":
            lines.append(f"  Total: {mlb_section.get('total_predictions', 0)}")
            lines.append(f"  Value Bets: {mlb_section.get('value_bets', 0)}")
            lines.append(
                (
                    "  Avg Confidence: "
                    f"{mlb_section.get('average_confidence', 0.0):.2f}"
                )
            )
            lines.append(
                f"  Avg Edge: {mlb_section.get('average_edge', 0.0):.3f}"
            )
        else:
            lines.append(
                (
                    f"  {mlb_section.get('status')}: "
                    f"{mlb_section.get('message')}"
                )
            )

        # Learning
        lines.append("\n🧠 Learning:")
        if learning_section.get("status") == "ok":
            if learning_section.get("summary"):
                s = learning_section["summary"].get("overall_metrics", {})
                lines.append(
                    f"  Total Predictions: {s.get('total_predictions', 0)}"
                )
                lines.append(f"  Accuracy: {s.get('accuracy', 'n/a')}")
                lines.append(f"  ROI: {s.get('roi', 'n/a')}")
                trend_text = learning_section["summary"].get(
                    "recent_trend", "n/a"
                )
                lines.append(f"  Trend: {trend_text}")
            else:
                lines.append(
                    "  Enhanced insights available "
                    "(feedback system)."
                )
        else:
            msg_status = learning_section.get('status')
            msg_text = learning_section.get('message')
            msg_line = f"  {msg_status}: {msg_text}"
            lines.append(msg_line)

        # NFL
        lines.append("\n🏈 NFL Tracking:")
        if nfl_section.get("status") == "ok":
            lines.append(
                f"  Active Games: {nfl_section.get('active_games', 0)}"
            )
            lines.append(
                (
                    "  Recent Predictions: "
                    f"{nfl_section.get('recent_predictions', 0)}"
                )
            )
            lines.append(
                (
                    "  Recent Accuracy: "
                    f"{nfl_section.get('recent_accuracy', 0.0):.2%}"
                )
            )
            lines.append(
                (
                    "  Avg Confidence: "
                    f"{nfl_section.get('avg_confidence', 0.0):.2f}"
                )
            )
        else:
            lines.append(
                (
                    f"  {nfl_section.get('status')}: "
                    f"{nfl_section.get('message')}"
                )
            )

        txt_path = self.reports_dir / f"daily_report_{date_str}.txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

        logger.info(f"Daily report saved to {json_path} and {txt_path}")
        return report


def main():
    date_override = os.getenv("REPORT_DATE")
    reporter = DailyReporter()
    reporter.generate(date_override)


if __name__ == "__main__":
    main()
