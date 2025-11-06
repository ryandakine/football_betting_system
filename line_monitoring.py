# line_monitoring.py
"""
ENHANCED Line Movement Tracking & Betting Timing System
Full-Featured Professional Version with Email Alerts, Web Dashboard, and Tests
FIXED VERSION - All syntax errors corrected
"""

import asyncio
import json
import logging
import os
import smtplib

# from .line_monitoring import router as line_monitoring_router
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from fastapi import APIRouter, BackgroundTasks, HTTPException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
LINE_HISTORY_DIR = Path("data/line_history")
ALERTS_DIR = Path("data/alerts")
LOGS_DIR = Path("data/logs")
CONFIG_DIR = Path("data/config")

# Create directories
for dir_path in [LINE_HISTORY_DIR, ALERTS_DIR, LOGS_DIR, CONFIG_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

BETTING_WINDOWS = {
    "TOO_EARLY": 120,  # >2 hours before game
    "OPTIMAL_WINDOW_MIN": 60,  # 1-4 hours before game
    "OPTIMAL_WINDOW_MAX": 240,
    "URGENT": 30,  # 30-60 mins before game
    "TOO_LATE": 0,  # <30 mins
}

THRESHOLDS = {
    "EDGE_ALERT": 4,  # cents change to trigger alert
    "SIGNIFICANT_MOVE": 8,  # cents for major line moves
    "MAJOR_MOVE": 15,  # cents for critical moves
    "MIN_EDGE_TO_TRACK": 5,  # minimum edge to start tracking
}

# Email configuration (load from environment or config)
EMAIL_CONFIG = {
    "enabled": os.getenv("EMAIL_ALERTS_ENABLED", "false").lower() == "true",
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "username": os.getenv("EMAIL_USERNAME", ""),
    "password": os.getenv("EMAIL_PASSWORD", ""),
    "recipient": os.getenv("ALERT_EMAIL", ""),
    "sender_name": "MLB Betting Intelligence System",
}


def get_line_history_path(game_id: str) -> Path:
    return LINE_HISTORY_DIR / f"{game_id}_history.json"


def get_alert_log_path(game_id: str) -> Path:
    return ALERTS_DIR / f"{game_id}_alerts.json"


def get_daily_log_path() -> Path:
    return LOGS_DIR / f"monitoring_{datetime.now().strftime('%Y%m%d')}.log"


def now_iso() -> str:
    return datetime.utcnow().isoformat()


def minutes_until_game(commence_time: str) -> float:
    """Calculate minutes until game starts with enhanced error handling"""
    try:
        # Handle multiple time formats
        if commence_time.endswith("Z"):
            game_time = datetime.fromisoformat(commence_time.replace("Z", "+00:00"))
        else:
            game_time = datetime.fromisoformat(commence_time)

        now = datetime.utcnow().replace(tzinfo=game_time.tzinfo)
        return (game_time - now).total_seconds() / 60
    except Exception as e:
        logger.warning(f"Error parsing commence_time '{commence_time}': {e}")
        return 0


def format_currency(cents: float) -> str:
    """Format cents as currency"""
    return f"${cents/100:.2f}"


def format_time_left(minutes: float) -> str:
    """Format minutes as human-readable time"""
    if minutes <= 0:
        return "Game started"
    elif minutes < 60:
        return f"{minutes:.0f} minutes"
    elif minutes < 1440:  # 24 hours
        hours = minutes / 60
        return f"{hours:.1f} hours"
    else:
        days = minutes / 1440
        return f"{days:.1f} days"


# --- ENHANCED LINE MOVEMENT TRACKER ---
class LineMovementTracker:
    def __init__(self):
        self.history_cache: dict[str, list[dict]] = {}
        self.active_games: set[str] = set()

    def record_snapshot(self, game_id: str, snapshot: dict) -> bool:
        """Record line snapshot with enhanced validation"""
        try:
            # Validate minimum required fields
            required_fields = ["edge_cents", "commence_time"]
            for field in required_fields:
                if field not in snapshot:
                    logger.warning(
                        f"Missing required field '{field}' in snapshot for {game_id}"
                    )
                    return False

            # Skip if edge is too small to track
            edge_cents = snapshot.get("edge_cents", 0)
            if edge_cents < THRESHOLDS["MIN_EDGE_TO_TRACK"]:
                return False

            path = get_line_history_path(game_id)
            history = self.load_history(game_id)

            # Add metadata
            snapshot.update(
                {
                    "timestamp": now_iso(),
                    "session_id": os.getenv("SESSION_ID", "default"),
                    "data_source": snapshot.get("data_source", "system"),
                }
            )

            history.append(snapshot)
            self.active_games.add(game_id)

            # Intelligent history management
            if len(history) > 50:
                # Keep last 30 + any major moves
                major_moves = [
                    h
                    for h in history
                    if abs(h.get("edge_cents", 0)) >= THRESHOLDS["MAJOR_MOVE"]
                ]
                recent_history = history[-30:]
                # Combine and deduplicate
                history = list(
                    {h["timestamp"]: h for h in major_moves + recent_history}.values()
                )
                history.sort(key=lambda x: x["timestamp"])

            with path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)
            self.history_cache[game_id] = history

            logger.info(f"Recorded snapshot for {game_id}: {edge_cents}¢ edge")
            return True

        except Exception as e:
            logger.error(f"Error recording snapshot for {game_id}: {e}")
            return False

    def load_history(self, game_id: str) -> list[dict]:
        """Load history with caching and error recovery"""
        if game_id in self.history_cache:
            return self.history_cache[game_id]

        path = get_line_history_path(game_id)
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    history = json.load(f)
                    # Validate history structure
                    if isinstance(history, list):
                        self.history_cache[game_id] = history
                        return history
        except Exception as e:
            logger.error(f"Error loading history for {game_id}: {e}")

        # Return empty history if load fails
        history = []
        self.history_cache[game_id] = history
        return history

    def get_latest_snapshot(self, game_id: str) -> dict | None:
        """Get most recent snapshot with validation"""
        history = self.load_history(game_id)
        if not history:
            return None

        latest = history[-1]
        # Validate latest snapshot has required fields
        if "edge_cents" in latest and "commence_time" in latest:
            return latest
        return None

    def detect_movement(self, game_id: str) -> dict:
        """Enhanced movement detection with trend analysis"""
        history = self.load_history(game_id)
        if len(history) < 2:
            return {
                "direction": "stable",
                "magnitude": 0,
                "change": 0,
                "significant": False,
                "trend": "insufficient_data",
                "velocity": 0,  # cents per hour
            }

        prev, curr = history[-2], history[-1]
        prev_edge = prev.get("edge_cents", 0)
        curr_edge = curr.get("edge_cents", 0)
        change = curr_edge - prev_edge

        # Calculate velocity (change per hour)
        try:
            time_diff = (
                datetime.fromisoformat(curr["timestamp"])
                - datetime.fromisoformat(prev["timestamp"])
            ).total_seconds() / 3600
            velocity = change / time_diff if time_diff > 0 else 0
        except:
            velocity = 0

        # Determine direction
        if abs(change) < THRESHOLDS["EDGE_ALERT"]:
            direction = "stable"
        elif change > 0:
            direction = "favorable"
        else:
            direction = "unfavorable"

        # Analyze trend over last 5 snapshots
        trend = self._analyze_trend(history[-5:])

        return {
            "direction": direction,
            "magnitude": abs(change),
            "change": change,
            "significant": abs(change) >= THRESHOLDS["SIGNIFICANT_MOVE"],
            "critical": abs(change) >= THRESHOLDS["MAJOR_MOVE"],
            "trend": trend,
            "velocity": velocity,
            "analysis": self._generate_movement_analysis(direction, abs(change), trend),
        }

    def _analyze_trend(self, recent_history: list[dict]) -> str:
        """Analyze trend over recent snapshots"""
        if len(recent_history) < 3:
            return "insufficient_data"

        edges = [h.get("edge_cents", 0) for h in recent_history]

        # Calculate trend
        increases = sum(1 for i in range(1, len(edges)) if edges[i] > edges[i - 1])
        decreases = sum(1 for i in range(1, len(edges)) if edges[i] < edges[i - 1])

        if increases > decreases:
            return "improving"
        elif decreases > increases:
            return "declining"
        else:
            return "volatile"

    def _generate_movement_analysis(
        self, direction: str, magnitude: float, trend: str
    ) -> str:
        """Generate human-readable movement analysis"""
        if direction == "stable":
            return f"Line stable with {trend} trend"
        elif direction == "favorable":
            return f"Line moved {magnitude:.1f}¢ in your favor ({trend} trend)"
        else:
            return f"Line moved {magnitude:.1f}¢ against you ({trend} trend)"

    def get_edge_trend(self, game_id: str) -> list[dict]:
        """Get comprehensive edge trend data"""
        history = self.load_history(game_id)
        trend_data = []

        for i, snap in enumerate(history):
            data_point = {
                "timestamp": snap["timestamp"],
                "edge_cents": snap.get("edge_cents", 0),
                "fanduel_odds": snap.get("fanduel_odds"),
                "value_rating": snap.get("value_rating", "Unknown"),
            }

            # Add movement info if not first snapshot
            if i > 0:
                prev_edge = history[i - 1].get("edge_cents", 0)
                curr_edge = snap.get("edge_cents", 0)
                data_point["movement"] = curr_edge - prev_edge

            trend_data.append(data_point)

        return trend_data

    def get_active_games(self) -> list[str]:
        """Get list of currently active games"""
        return list(self.active_games)

    def cleanup_old_games(self, hours_old: int = 24):
        """Clean up games that have finished"""
        cutoff = datetime.utcnow() - timedelta(hours=hours_old)
        games_to_remove = []

        for game_id in self.active_games:
            latest = self.get_latest_snapshot(game_id)
            if latest:
                try:
                    commence_time = datetime.fromisoformat(
                        latest["commence_time"].replace("Z", "+00:00")
                    )
                    if commence_time < cutoff:
                        games_to_remove.append(game_id)
                except:
                    continue

        for game_id in games_to_remove:
            self.active_games.discard(game_id)
            logger.info(f"Cleaned up old game: {game_id}")


# --- ENHANCED BETTING TIMING ADVISOR ---
class BettingTimingAdvisor:
    def get_status(self, commence_time: str) -> dict:
        """Enhanced status with detailed timing info"""
        minutes_left = minutes_until_game(commence_time)

        # Determine status
        if minutes_left <= 0:
            status = "GAME_STARTED"
            priority = "NONE"
            action_window = "CLOSED"
        elif minutes_left <= BETTING_WINDOWS["TOO_LATE"]:
            status = "TOO_LATE"
            priority = "LOW"
            action_window = "RISKY"
        elif minutes_left <= BETTING_WINDOWS["URGENT"]:
            status = "URGENT"
            priority = "HIGH"
            action_window = "CLOSING"
        elif (
            BETTING_WINDOWS["OPTIMAL_WINDOW_MIN"]
            <= minutes_left
            <= BETTING_WINDOWS["OPTIMAL_WINDOW_MAX"]
        ):
            status = "OPTIMAL_WINDOW"
            priority = "MEDIUM"
            action_window = "OPTIMAL"
        else:
            status = "TOO_EARLY"
            priority = "LOW"
            action_window = "EARLY"

        return {
            "status": status,
            "minutes_left": round(minutes_left, 1),
            "priority": priority,
            "action_window": action_window,
            "time_description": format_time_left(minutes_left),
            "optimal_window_opens_in": max(
                0, minutes_left - BETTING_WINDOWS["OPTIMAL_WINDOW_MAX"]
            ),
            "window_closes_in": max(0, minutes_left - BETTING_WINDOWS["URGENT"]),
        }

    def get_recommendation(
        self, status: str, movement: dict, edge_cents: float
    ) -> dict:
        """Enhanced recommendation with confidence scoring"""
        action = "MONITOR"
        reason = "Standard monitoring"
        confidence = 0.5
        urgency_level = "LOW"

        if status == "TOO_EARLY":
            if movement["trend"] == "improving":
                action = "WAIT"
                reason = "Edge improving, wait for better value"
                confidence = 0.7
            else:
                action = "MONITOR"
                reason = "Too early, monitor for line movement"
                confidence = 0.6

        elif status == "OPTIMAL_WINDOW":
            if (
                movement["direction"] == "favorable"
                and movement["trend"] == "improving"
            ):
                action = "WAIT_AND_WATCH"
                reason = "Edge improving, but ready to bet if reverses"
                confidence = 0.8
                urgency_level = "MEDIUM"
            elif movement["direction"] == "unfavorable" or movement["critical"]:
                action = "BET_NOW"
                reason = "Edge declining or critical move detected"
                confidence = 0.9
                urgency_level = "HIGH"
            else:
                action = "BET_NOW"
                reason = "Optimal window with good edge"
                confidence = 0.8
                urgency_level = "MEDIUM"

        elif status == "URGENT":
            action = "BET_IMMEDIATELY"
            reason = "Time running out, bet now or lose opportunity"
            confidence = 0.9
            urgency_level = "HIGH"

        elif status == "TOO_LATE":
            action = "SKIP"
            reason = "Too risky, game starting soon"
            confidence = 0.9
            urgency_level = "LOW"

        else:
            action = "SKIP"
            reason = "Game started or finished"
            confidence = 1.0
            urgency_level = "NONE"

        # Adjust confidence based on edge size
        if edge_cents > 20:
            confidence = min(1.0, confidence + 0.1)
        elif edge_cents < 8:
            confidence = max(0.3, confidence - 0.1)

        return {
            "action": action,
            "reason": reason,
            "confidence": confidence,
            "urgency_level": urgency_level,
            "edge_assessment": self._assess_edge_strength(edge_cents),
        }

    def _assess_edge_strength(self, edge_cents: float) -> str:
        """Assess the strength of the betting edge"""
        if edge_cents >= 25:
            return "EXCEPTIONAL"
        elif edge_cents >= 15:
            return "STRONG"
        elif edge_cents >= 10:
            return "GOOD"
        elif edge_cents >= 5:
            return "MODERATE"
        else:
            return "WEAK"


# --- ENHANCED ALERT MANAGER WITH EMAIL ---
class AlertManager:
    def __init__(self):
        self.alert_cache: dict[str, list[dict]] = {}
        self.email_enabled = EMAIL_CONFIG["enabled"]
        self.daily_alert_count = 0
        self.last_email_sent = {}

    def log_alert(
        self,
        game_id: str,
        alert_type: str,
        message: str,
        priority: str = "MEDIUM",
        email_alert: bool = False,
    ) -> bool:
        """Enhanced alert logging with email capability"""
        try:
            path = get_alert_log_path(game_id)
            alert = {
                "timestamp": now_iso(),
                "type": alert_type,
                "message": message,
                "priority": priority,
                "email_sent": False,
            }

            alerts = self.load_alerts(game_id)
            alerts.append(alert)

            # Keep last 20 alerts
            if len(alerts) > 20:
                alerts = alerts[-20:]

            with path.open("w", encoding="utf-8") as f:
                json.dump(alerts, f, indent=2)
            self.alert_cache[game_id] = alerts

            # Console output with enhanced formatting
            priority_icon = {"HIGH": "🚨", "MEDIUM": "⚠️", "LOW": "ℹ️"}.get(
                priority, "📢"
            )
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"{priority_icon} [{timestamp}][{game_id}] {alert_type}: {message}")

            # Send email if enabled and conditions met
            if email_alert and self.email_enabled and priority in ["HIGH", "CRITICAL"]:
                if self._should_send_email(game_id, alert_type):
                    email_sent = self._send_email_alert(
                        game_id, alert_type, message, priority
                    )
                    if email_sent:
                        alert["email_sent"] = True
                        # Update the file with email status
                        with path.open("w", encoding="utf-8") as f:
                            json.dump(alerts, f, indent=2)

            self.daily_alert_count += 1
            logger.info(f"Alert logged: {game_id} - {alert_type} - {priority}")
            return True

        except Exception as e:
            logger.error(f"Error logging alert: {e}")
            return False

    def _should_send_email(self, game_id: str, alert_type: str) -> bool:
        """Check if email should be sent (avoid spam)"""
        now = datetime.now()
        key = f"{game_id}_{alert_type}"

        # Don't send same alert type for same game more than once per hour
        if key in self.last_email_sent:
            if (now - self.last_email_sent[key]).total_seconds() < 3600:
                return False

        # Don't send more than 10 emails per day
        if self.daily_alert_count > 10:
            return False

        self.last_email_sent[key] = now
        return True

    def _send_email_alert(
        self, game_id: str, alert_type: str, message: str, priority: str
    ) -> bool:
        """Send email alert"""
        try:
            if not EMAIL_CONFIG["username"] or not EMAIL_CONFIG["recipient"]:
                logger.warning("Email credentials not configured")
                return False

            msg = MIMEMultipart()
            msg["From"] = f"{EMAIL_CONFIG['sender_name']} <{EMAIL_CONFIG['username']}>"
            msg["To"] = EMAIL_CONFIG["recipient"]
            msg["Subject"] = f"🚨 MLB Betting Alert: {alert_type} - {game_id}"

            # Create HTML email body
            html_body = f"""
            <html>
            <body>
                <h2>🚨 MLB Betting Intelligence Alert</h2>
                <p><strong>Game ID:</strong> {game_id}</p>
                <p><strong>Alert Type:</strong> {alert_type}</p>
                <p><strong>Priority:</strong> {priority}</p>
                <p><strong>Message:</strong> {message}</p>
                <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <hr>
                <p><em>This alert was generated by your MLB Betting Intelligence System</em></p>
            </body>
            </html>
            """

            msg.attach(MIMEText(html_body, "html"))

            # Send email
            with smtplib.SMTP(
                EMAIL_CONFIG["smtp_server"], EMAIL_CONFIG["smtp_port"]
            ) as server:
                server.starttls()
                server.login(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
                server.send_message(msg)

            logger.info(f"Email alert sent for {game_id}: {alert_type}")
            return True

        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
            return False

    def load_alerts(self, game_id: str) -> list[dict]:
        """Load alerts with error handling"""
        if game_id in self.alert_cache:
            return self.alert_cache[game_id]

        path = get_alert_log_path(game_id)
        try:
            if path.exists():
                with path.open("r", encoding="utf-8") as f:
                    alerts = json.load(f)
                    if isinstance(alerts, list):
                        self.alert_cache[game_id] = alerts
                        return alerts
        except Exception as e:
            logger.error(f"Error loading alerts for {game_id}: {e}")

        alerts = []
        self.alert_cache[game_id] = alerts
        return alerts

    def get_recent_alerts(self, hours: int = 4) -> list[dict]:
        """Get all recent alerts across all games"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        recent_alerts = []

        for file in ALERTS_DIR.glob("*_alerts.json"):
            game_id = file.stem.replace("_alerts", "")
            alerts = self.load_alerts(game_id)

            for alert in alerts:
                try:
                    alert_time = datetime.fromisoformat(alert["timestamp"])
                    if alert_time > cutoff:
                        alert["game_id"] = game_id
                        recent_alerts.append(alert)
                except:
                    continue

        # Sort by timestamp, newest first
        recent_alerts.sort(key=lambda x: x["timestamp"], reverse=True)
        return recent_alerts

    def get_alert_summary(self) -> dict:
        """Get summary of alert activity"""
        return {
            "daily_count": self.daily_alert_count,
            "email_enabled": self.email_enabled,
            "recent_alerts": len(self.get_recent_alerts(2)),
            "last_reset": datetime.now().strftime("%Y-%m-%d"),
        }


# --- INTEGRATION OBJECTS ---
line_tracker = LineMovementTracker()
timing_advisor = BettingTimingAdvisor()
alert_manager = AlertManager()


# --- ENHANCED BACKGROUND MONITORING ---
async def monitor_lines_periodically(system, interval_minutes: int = 15):
    """Enhanced background monitoring with comprehensive logging"""
    logger.info(
        f"🔄 Starting enhanced line monitoring every {interval_minutes} minutes..."
    )

    while True:
        monitoring_start = datetime.now()

        try:
            logger.info(
                f"⏰ {monitoring_start.strftime('%H:%M:%S')} - Starting monitoring cycle..."
            )

            # Get current opportunities
            picks = await system.analyze_opportunities_concurrently([])

            if not picks:
                logger.info("No opportunities found in this cycle")
                await asyncio.sleep(interval_minutes * 60)
                continue

            processed_count = 0
            alerts_triggered = 0

            for pick in picks:
                try:
                    game_id = pick["game_id"]

                    # Create comprehensive snapshot
                    snapshot = {
                        "timestamp": now_iso(),
                        "edge_cents": pick.get("edge_cents", 0),
                        "fanduel_odds": pick.get("fanduel_odds"),
                        "commence_time": pick.get("commence_time"),
                        "value_rating": pick.get("value_rating"),
                        "home_team": pick.get("home_team"),
                        "away_team": pick.get("away_team"),
                        "sport": pick.get("sport", "baseball"),
                        "data_source": "system_analysis",
                    }

                    # Record snapshot
                    if not line_tracker.record_snapshot(game_id, snapshot):
                        continue

                    processed_count += 1

                    # Analyze movement and timing
                    movement = line_tracker.detect_movement(game_id)
                    status_info = timing_advisor.get_status(snapshot["commence_time"])
                    recommendation = timing_advisor.get_recommendation(
                        status_info["status"], movement, snapshot["edge_cents"]
                    )

                    # Generate intelligent alerts
                    alert_triggered = False

                    if (
                        status_info["status"] == "OPTIMAL_WINDOW"
                        and recommendation["action"] == "BET_NOW"
                    ):
                        alert_manager.log_alert(
                            game_id,
                            "OPTIMAL_BETTING_WINDOW",
                            f"Optimal window: {recommendation['reason']}",
                            "HIGH",
                            email_alert=True,
                        )
                        alert_triggered = True

                    elif movement["critical"]:
                        alert_manager.log_alert(
                            game_id,
                            "CRITICAL_LINE_MOVEMENT",
                            f"Critical move: {movement['analysis']}",
                            "CRITICAL",
                            email_alert=True,
                        )
                        alert_triggered = True

                    elif (
                        movement["significant"]
                        and movement["direction"] == "unfavorable"
                    ):
                        alert_manager.log_alert(
                            game_id,
                            "UNFAVORABLE_MOVEMENT",
                            f"Significant unfavorable move: {movement['analysis']}",
                            "HIGH",
                            email_alert=True,
                        )
                        alert_triggered = True

                    elif (
                        status_info["status"] == "URGENT"
                        and recommendation["action"] == "BET_IMMEDIATELY"
                    ):
                        alert_manager.log_alert(
                            game_id,
                            "URGENT_BETTING_WINDOW",
                            f"Last chance: {status_info['time_description']} remaining",
                            "HIGH",
                            email_alert=True,
                        )
                        alert_triggered = True

                    if alert_triggered:
                        alerts_triggered += 1

                except Exception as e:
                    logger.error(
                        f"Error processing pick {pick.get('game_id', 'unknown')}: {e}"
                    )
                    continue

            # Cleanup old games
            line_tracker.cleanup_old_games()

            # Log monitoring summary
            monitoring_duration = (datetime.now() - monitoring_start).total_seconds()
            logger.info(
                f"✅ Monitoring cycle complete: {processed_count} games processed, "
                f"{alerts_triggered} alerts triggered, {monitoring_duration:.1f}s duration"
            )

        except Exception as e:
            logger.error(f"❌ Monitoring cycle failed: {e}")

        await asyncio.sleep(interval_minutes * 60)


# --- ENHANCED FASTAPI ROUTER ---
router = APIRouter(prefix="/line-monitoring", tags=["Line Monitoring"])


@router.get("/movement/{game_id}")
async def get_line_movement(game_id: str):
    """Get comprehensive line movement data for a game"""
    try:
        history = line_tracker.load_history(game_id)
        if not history:
            raise HTTPException(status_code=404, detail="Game not found")

        trend = line_tracker.get_edge_trend(game_id)
        movement = line_tracker.detect_movement(game_id)
        latest = line_tracker.get_latest_snapshot(game_id)

        return {
            "game_id": game_id,
            "latest_snapshot": latest,
            "movement_analysis": movement,
            "trend_data": trend,
            "history_summary": {
                "total_snapshots": len(history),
                "first_recorded": history[0]["timestamp"] if history else None,
                "last_updated": history[-1]["timestamp"] if history else None,
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting line movement for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/recommendations")
async def get_betting_recommendations():
    """Get comprehensive betting recommendations"""
    try:
        recommendations = []

        for file in LINE_HISTORY_DIR.glob("*_history.json"):
            game_id = file.stem.replace("_history", "")
            latest = line_tracker.get_latest_snapshot(game_id)

            if not latest:
                continue

            movement = line_tracker.detect_movement(game_id)
            status_info = timing_advisor.get_status(latest["commence_time"])
            recommendation = timing_advisor.get_recommendation(
                status_info["status"], movement, latest["edge_cents"]
            )

            recommendations.append(
                {
                    "game_id": game_id,
                    "teams": f"{latest.get('away_team', 'TBD')} @ {latest.get('home_team', 'TBD')}",
                    "edge_cents": latest["edge_cents"],
                    "edge_formatted": format_currency(latest["edge_cents"]),
                    "status": status_info,
                    "movement": movement,
                    "recommendation": recommendation,
                    "last_updated": latest["timestamp"],
                }
            )

        # Sort by priority and confidence
        priority_order = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        recommendations.sort(
            key=lambda x: (
                priority_order.get(x["status"]["priority"], 0),
                x["recommendation"]["confidence"],
                x["edge_cents"],
            ),
            reverse=True,
        )

        return {
            "count": len(recommendations),
            "recommendations": recommendations,
            "generated_at": now_iso(),
            "summary": {
                "high_priority": len(
                    [r for r in recommendations if r["status"]["priority"] == "HIGH"]
                ),
                "medium_priority": len(
                    [r for r in recommendations if r["status"]["priority"] == "MEDIUM"]
                ),
                "low_priority": len(
                    [r for r in recommendations if r["status"]["priority"] == "LOW"]
                ),
            },
        }

    except Exception as e:
        logger.error(f"Error getting recommendations: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts/{game_id}")
async def get_alerts(game_id: str):
    """Get alerts for a specific game"""
    try:
        alerts = alert_manager.load_alerts(game_id)
        return {
            "game_id": game_id,
            "alerts": alerts,
            "count": len(alerts),
            "recent_count": len(
                [
                    a
                    for a in alerts
                    if (
                        datetime.utcnow() - datetime.fromisoformat(a["timestamp"])
                    ).total_seconds()
                    < 14400
                ]
            ),
        }
    except Exception as e:
        logger.error(f"Error getting alerts for {game_id}: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/dashboard")
async def get_dashboard():
    """Comprehensive dashboard with real-time metrics"""
    try:
        # Get active games
        active_games = []
        urgent_games = []

        for file in LINE_HISTORY_DIR.glob("*_history.json"):
            game_id = file.stem.replace("_history", "")
            latest = line_tracker.get_latest_snapshot(game_id)

            if not latest:
                continue

            status_info = timing_advisor.get_status(latest["commence_time"])
            movement = line_tracker.detect_movement(game_id)

            game_info = {
                "game_id": game_id,
                "teams": f"{latest.get('away_team', 'TBD')} @ {latest.get('home_team', 'TBD')}",
                "edge_cents": latest["edge_cents"],
                "edge_formatted": format_currency(latest["edge_cents"]),
                "status": status_info["status"],
                "time_left": status_info["time_description"],
                "movement": movement["direction"],
                "last_updated": latest["timestamp"],
            }

            active_games.append(game_info)

            if status_info["priority"] == "HIGH":
                urgent_games.append(game_info)

        # Get recent alerts
        recent_alerts = alert_manager.get_recent_alerts(2)

        return {
            "summary": {
                "total_games_tracked": len(active_games),
                "urgent_games": len(urgent_games),
                "recent_alerts": len(recent_alerts),
                "system_status": "ACTIVE",
                "last_updated": now_iso(),
            },
            "urgent_games": urgent_games,
            "recent_alerts": recent_alerts[:10],  # Last 10 alerts
            "alert_summary": alert_manager.get_alert_summary(),
            "system_health": {
                "monitoring_active": True,
                "email_alerts": EMAIL_CONFIG["enabled"],
                "data_directories": {
                    "line_history": len(list(LINE_HISTORY_DIR.glob("*.json"))),
                    "alerts": len(list(ALERTS_DIR.glob("*.json"))),
                    "logs": len(list(LOGS_DIR.glob("*.log"))),
                },
            },
        }

    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/alerts/recent")
async def get_recent_alerts(hours: int = 4):
    """Get recent alerts across all games"""
    try:
        alerts = alert_manager.get_recent_alerts(hours)
        return {
            "alerts": alerts,
            "count": len(alerts),
            "hours": hours,
            "generated_at": now_iso(),
        }
    except Exception as e:
        logger.error(f"Error getting recent alerts: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/test-email")
async def test_email_alert():
    """Test email alert functionality"""
    try:
        if not EMAIL_CONFIG["enabled"]:
            return {"success": False, "message": "Email alerts are not enabled"}

        success = alert_manager._send_email_alert(
            "TEST_GAME",
            "TEST_ALERT",
            "This is a test email from your MLB Betting Intelligence System",
            "MEDIUM",
        )

        return {
            "success": success,
            "message": (
                "Test email sent successfully"
                if success
                else "Failed to send test email"
            ),
        }

    except Exception as e:
        logger.error(f"Error testing email: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/health")
async def health_check():
    """System health check endpoint"""
    try:
        return {
            "status": "healthy",
            "timestamp": now_iso(),
            "components": {
                "line_tracker": "active",
                "timing_advisor": "active",
                "alert_manager": "active",
                "email_alerts": "enabled" if EMAIL_CONFIG["enabled"] else "disabled",
            },
            "data_status": {
                "games_tracked": len(list(LINE_HISTORY_DIR.glob("*.json"))),
                "alerts_logged": len(list(ALERTS_DIR.glob("*.json"))),
                "directories_exist": all(
                    [LINE_HISTORY_DIR.exists(), ALERTS_DIR.exists(), LOGS_DIR.exists()]
                ),
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# --- UTILITY FUNCTIONS ---
def cleanup_old_data(days_to_keep: int = 7):
    """Enhanced cleanup with logging"""
    cutoff = datetime.utcnow() - timedelta(days=days_to_keep)
    cleaned_files = 0

    for directory in [LINE_HISTORY_DIR, ALERTS_DIR, LOGS_DIR]:
        for file in directory.glob("*.json"):
            try:
                if file.stat().st_mtime < cutoff.timestamp():
                    file.unlink()
                    cleaned_files += 1
                    logger.info(f"Cleaned up old file: {file}")
            except Exception as e:
                logger.error(f"Error cleaning up {file}: {e}")

    logger.info(f"Cleanup complete: {cleaned_files} files removed")
    return cleaned_files


def export_game_data(game_id: str) -> dict:
    """Export all data for a specific game"""
    try:
        history = line_tracker.load_history(game_id)
        alerts = alert_manager.load_alerts(game_id)

        return {
            "game_id": game_id,
            "export_timestamp": now_iso(),
            "line_history": history,
            "alerts": alerts,
            "summary": {
                "total_snapshots": len(history),
                "total_alerts": len(alerts),
                "first_tracked": history[0]["timestamp"] if history else None,
                "last_updated": history[-1]["timestamp"] if history else None,
            },
        }
    except Exception as e:
        logger.error(f"Error exporting data for {game_id}: {e}")
        return {"error": str(e)}


# --- TESTING UTILITIES ---
class TestingUtilities:
    """Utilities for testing the line monitoring system"""

    @staticmethod
    def create_test_snapshot(
        game_id: str, edge_cents: float, minutes_until_game: float
    ) -> dict:
        """Create a test snapshot for testing"""
        commence_time = (
            datetime.utcnow() + timedelta(minutes=minutes_until_game)
        ).isoformat()
        return {
            "game_id": game_id,
            "edge_cents": edge_cents,
            "commence_time": commence_time,
            "fanduel_odds": -110,
            "home_team": "Test Home",
            "away_team": "Test Away",
            "value_rating": "GOOD",
        }

    @staticmethod
    def simulate_line_movement(
        game_id: str, initial_edge: float, movements: list[float]
    ):
        """Simulate line movement for testing"""
        for i, movement in enumerate(movements):
            snapshot = TestingUtilities.create_test_snapshot(
                game_id, initial_edge + movement, 120 - (i * 30)  # Game gets closer
            )
            line_tracker.record_snapshot(game_id, snapshot)

    @staticmethod
    def run_basic_tests():
        """Run basic system tests"""
        test_results = []

        # Test 1: Basic snapshot recording
        try:
            test_snapshot = TestingUtilities.create_test_snapshot("TEST_001", 15.5, 120)
            success = line_tracker.record_snapshot("TEST_001", test_snapshot)
            test_results.append({"test": "snapshot_recording", "passed": success})
        except Exception as e:
            test_results.append(
                {"test": "snapshot_recording", "passed": False, "error": str(e)}
            )

        # Test 2: Movement detection
        try:
            TestingUtilities.simulate_line_movement("TEST_002", 10.0, [0, 5, -3, 8])
            movement = line_tracker.detect_movement("TEST_002")
            test_results.append(
                {"test": "movement_detection", "passed": movement is not None}
            )
        except Exception as e:
            test_results.append(
                {"test": "movement_detection", "passed": False, "error": str(e)}
            )

        # Test 3: Timing advisor
        try:
            status = timing_advisor.get_status(
                (datetime.utcnow() + timedelta(hours=2)).isoformat()
            )
            test_results.append(
                {
                    "test": "timing_advisor",
                    "passed": status["status"] == "OPTIMAL_WINDOW",
                }
            )
        except Exception as e:
            test_results.append(
                {"test": "timing_advisor", "passed": False, "error": str(e)}
            )

        # Test 4: Alert logging
        try:
            success = alert_manager.log_alert(
                "TEST_003", "TEST_ALERT", "Test message", "LOW"
            )
            test_results.append({"test": "alert_logging", "passed": success})
        except Exception as e:
            test_results.append(
                {"test": "alert_logging", "passed": False, "error": str(e)}
            )

        return test_results


__all__ = [
    "LineMovementTracker",
    "BettingTimingAdvisor",
    "AlertManager",
    "line_tracker",
    "timing_advisor",
    "alert_manager",
    "monitor_lines_periodically",
    "router",
    "cleanup_old_data",
    "export_game_data",
    "TestingUtilities",
]
