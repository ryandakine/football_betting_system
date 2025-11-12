#!/usr/bin/env python3
"""
NFL Betting System Monitor - Continuous State Monitoring with Auto-Remediation

WHY THIS EXISTS:
Instead of human-triggered analysis, the system continuously monitors ideal state
and auto-remediates when drift is detected.

DESIGN PHILOSOPHY: Self-Healing Systems
- Define ideal state for each metric
- Detect drift automatically
- Auto-remediate when possible
- Alert human when manual intervention needed

IDEAL STATE (NFL System):
- ROI: 42-47% (with contrarian intelligence)
- CLV: +1.5 points average
- Contrarian alignment: Fade public when >70% on one side
- Home favorite bias: 40-50% (not 60-70%)
- Trap detection: Catch handle divergence accurately
- Win rate: 60%+ on contrarian picks

MONITORS:
1. Line Movement Monitor - Detect steam moves, reverse line movement
2. CLV Monitor - Track closing line value vs bet placement time
3. Contrarian Bias Monitor - Detect if picking with public too often
4. Home Favorite Bias Monitor - Catch regression to home favorite picks
5. ROI Performance Monitor - Detect ROI drift from target
6. Trap Detection Accuracy Monitor - Validate trap scores

AUTO-REMEDIATION:
- Alert on steam moves → Suggest immediate bet
- CLV drift → Adjust bet timing recommendations
- Contrarian drift → Increase contrarian weight in DeepSeek
- Home favorite bias → Flag for reanalysis
- ROI drift → Trigger model retraining or threshold adjustment

USAGE:
    # Start continuous monitoring
    python nfl_system_monitor.py --start

    # Check current status
    python nfl_system_monitor.py --status

    # Run single check
    python nfl_system_monitor.py --check-all
"""

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from bankroll_tracker import BankrollTracker


@dataclass
class Drift:
    """Represents detected drift from ideal state"""
    metric_name: str
    current_value: float
    target_value: float
    drift_amount: float
    severity: str  # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    detected_at: str
    can_auto_fix: bool
    remediation_action: Optional[str] = None
    root_cause: Optional[str] = None


class BaseMonitor(ABC):
    """Base class for all monitors"""

    def __init__(self, name: str, check_interval_seconds: int = 60):
        self.name = name
        self.check_interval = check_interval_seconds
        self.last_check = None
        self.drift_history = []
        self.data_dir = Path(__file__).parent / "data" / "monitoring"
        self.data_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def check_for_drift(self) -> Optional[Drift]:
        """Check if metric has drifted from ideal state"""
        pass

    @abstractmethod
    def get_ideal_state(self) -> Dict:
        """Return ideal state definition"""
        pass

    def log_drift(self, drift: Drift):
        """Log detected drift"""
        self.drift_history.append(drift)

        # Save to file
        log_file = self.data_dir / f"{self.name}_drift_log.json"

        if log_file.exists():
            with open(log_file, 'r') as f:
                logs = json.load(f)
        else:
            logs = []

        logs.append({
            'metric_name': drift.metric_name,
            'current_value': drift.current_value,
            'target_value': drift.target_value,
            'drift_amount': drift.drift_amount,
            'severity': drift.severity,
            'detected_at': drift.detected_at,
            'can_auto_fix': drift.can_auto_fix,
            'remediation_action': drift.remediation_action,
            'root_cause': drift.root_cause
        })

        with open(log_file, 'w') as f:
            json.dump(logs, f, indent=2)

    def get_drift_severity(self, drift_pct: float) -> str:
        """Classify drift severity"""
        if abs(drift_pct) < 5:
            return "LOW"
        elif abs(drift_pct) < 10:
            return "MEDIUM"
        elif abs(drift_pct) < 20:
            return "HIGH"
        else:
            return "CRITICAL"


class LineMovementMonitor(BaseMonitor):
    """
    Monitors line movements for steam moves and reverse line movement.

    IDEAL STATE:
    - Lines move gradually (<0.5 pts per hour)
    - No steam moves (>1.5 pts in <5 minutes)
    - Reverse line movement detected (sharp money indicator)

    DRIFT:
    - Steam move detected (>1.5 pts in <5 min)
    - Missing reverse line movement opportunities

    AUTO-REMEDIATION:
    - Alert on steam move → Suggest immediate bet
    - Alert on reverse line movement → Validate contrarian pick
    """

    def __init__(self):
        super().__init__(name="line_movement", check_interval_seconds=60)
        self.line_history_file = self.data_dir / "line_history.json"

    def get_ideal_state(self) -> Dict:
        return {
            'max_line_move_per_hour': 0.5,
            'steam_move_threshold': 1.5,  # points
            'steam_move_time_threshold': 300,  # 5 minutes
            'reverse_line_movement_detection': True
        }

    def check_for_drift(self) -> Optional[Drift]:
        """Check for steam moves or reverse line movement"""
        # Load line history
        if not self.line_history_file.exists():
            return None

        with open(self.line_history_file, 'r') as f:
            line_history = json.load(f)

        # Check for recent steam moves
        for game_id, history in line_history.items():
            if len(history) < 2:
                continue

            # Get last two line readings
            prev = history[-2]
            curr = history[-1]

            time_delta = (
                datetime.fromisoformat(curr['timestamp']) -
                datetime.fromisoformat(prev['timestamp'])
            ).total_seconds()

            line_move = abs(curr['line'] - prev['line'])

            # Detect steam move
            if line_move > 1.5 and time_delta < 300:  # 5 minutes
                drift = Drift(
                    metric_name="steam_move",
                    current_value=line_move,
                    target_value=0.5,
                    drift_amount=line_move - 0.5,
                    severity="HIGH",
                    detected_at=datetime.now().isoformat(),
                    can_auto_fix=False,
                    remediation_action=f"Alert: Steam move detected on {game_id} - Sharp money moving line",
                    root_cause=f"Line moved {line_move:.1f} points in {time_delta:.0f} seconds"
                )
                return drift

            # Detect reverse line movement
            if self._is_reverse_line_movement(history):
                drift = Drift(
                    metric_name="reverse_line_movement",
                    current_value=1.0,  # Binary: detected
                    target_value=1.0,  # We want to detect this
                    drift_amount=0.0,  # This is good drift
                    severity="LOW",
                    detected_at=datetime.now().isoformat(),
                    can_auto_fix=False,
                    remediation_action=f"Alert: Reverse line movement on {game_id} - Validate contrarian pick",
                    root_cause="Line moving against public betting percentage"
                )
                return drift

        return None

    def _is_reverse_line_movement(self, history: List[Dict]) -> bool:
        """Detect if line is moving against public betting"""
        # This would require public betting % data
        # Simplified: Check if line moved toward underdog despite favorite being popular
        if len(history) < 2:
            return False

        prev = history[-2]
        curr = history[-1]

        # If line moved toward underdog (less negative or more positive)
        # AND public is on favorite, it's reverse line movement
        line_moved_toward_underdog = curr['line'] > prev['line']

        # Would check public % here if available
        # For now, flag any significant move toward underdog
        return line_moved_toward_underdog and abs(curr['line'] - prev['line']) > 0.5


class CLVMonitor(BaseMonitor):
    """
    Monitors closing line value to optimize bet timing.

    IDEAL STATE:
    - Average CLV: +1.5 points
    - Bet timing: Early week (Tuesday-Wednesday)
    - Positive CLV on 70%+ of bets

    DRIFT:
    - CLV drops below +0.5
    - Betting too late (Thursday+)
    - Missing line value

    AUTO-REMEDIATION:
    - Adjust bet timing recommendations
    - Alert to bet earlier
    - Flag games losing value
    """

    def __init__(self):
        super().__init__(name="clv", check_interval_seconds=3600)  # Check hourly
        self.tracker = BankrollTracker()

    def get_ideal_state(self) -> Dict:
        return {
            'target_avg_clv': 1.5,
            'target_positive_clv_pct': 0.70,
            'optimal_bet_day': 'Tuesday',
            'optimal_bet_hour': 10
        }

    def check_for_drift(self) -> Optional[Drift]:
        """Check if CLV is drifting from target"""
        bet_log_file = Path(__file__).parent / "data" / "bet_log.json"

        if not bet_log_file.exists():
            return None

        with open(bet_log_file, 'r') as f:
            bets = json.load(f)

        # Filter to recent bets (last 10)
        recent_bets = bets[-10:] if len(bets) > 10 else bets

        # Calculate average CLV
        clvs = []
        for bet in recent_bets:
            if 'clv' in bet:
                clvs.append(bet['clv'])

        if not clvs:
            return None

        avg_clv = sum(clvs) / len(clvs)
        target_clv = 1.5

        # Check for drift
        if avg_clv < target_clv - 0.5:  # Drifted 0.5+ points
            # Analyze root cause
            bet_times = [datetime.fromisoformat(bet['placed_at']) for bet in recent_bets if 'placed_at' in bet]
            avg_hour = sum(t.hour for t in bet_times) / len(bet_times) if bet_times else 12

            drift_pct = ((target_clv - avg_clv) / target_clv) * 100

            drift = Drift(
                metric_name="clv_drift",
                current_value=avg_clv,
                target_value=target_clv,
                drift_amount=target_clv - avg_clv,
                severity=self.get_drift_severity(drift_pct),
                detected_at=datetime.now().isoformat(),
                can_auto_fix=True,
                remediation_action=f"Adjust bet timing: Currently betting at {avg_hour:.0f}:00, move to 10:00 for +{target_clv - avg_clv:.1f} CLV",
                root_cause=f"Betting too late (avg hour: {avg_hour:.0f})"
            )
            return drift

        return None


class ContrarianBiasMonitor(BaseMonitor):
    """
    Monitors if system is picking with public too often (losing contrarian edge).

    IDEAL STATE:
    - Fade public when >70% on one side
    - Contrarian picks: 40-50% of total picks
    - Public alignment: <30%

    DRIFT:
    - Picking with public >50% of time
    - Not fading when public >70%
    - Losing contrarian edge

    AUTO-REMEDIATION:
    - Increase contrarian weight in DeepSeek prompt
    - Flag picks that align with public
    - Suggest contrarian reanalysis
    """

    def __init__(self):
        super().__init__(name="contrarian_bias", check_interval_seconds=3600)
        self.tracker = BankrollTracker()

    def get_ideal_state(self) -> Dict:
        return {
            'target_contrarian_pick_pct': 0.45,  # 40-50%
            'max_public_alignment_pct': 0.30,
            'fade_threshold': 0.70  # Fade when public >70%
        }

    def check_for_drift(self) -> Optional[Drift]:
        """Check if picking with public too often"""
        bet_log_file = Path(__file__).parent / "data" / "bet_log.json"

        if not bet_log_file.exists():
            return None

        with open(bet_log_file, 'r') as f:
            bets = json.load(f)

        # Filter to recent bets (last 20)
        recent_bets = bets[-20:] if len(bets) > 20 else bets

        # Count public alignment
        public_aligned = 0
        contrarian_picks = 0
        total = 0

        for bet in recent_bets:
            if 'public_alignment' in bet:
                total += 1
                if bet['public_alignment'] == 'with_public':
                    public_aligned += 1
                elif bet['public_alignment'] == 'contrarian':
                    contrarian_picks += 1

        if total == 0:
            return None

        public_alignment_pct = public_aligned / total
        contrarian_pct = contrarian_picks / total

        # Check for drift
        target_max_public = 0.30

        if public_alignment_pct > target_max_public + 0.20:  # 20% over target
            drift_pct = ((public_alignment_pct - target_max_public) / target_max_public) * 100

            drift = Drift(
                metric_name="contrarian_bias_drift",
                current_value=public_alignment_pct,
                target_value=target_max_public,
                drift_amount=public_alignment_pct - target_max_public,
                severity=self.get_drift_severity(drift_pct),
                detected_at=datetime.now().isoformat(),
                can_auto_fix=True,
                remediation_action=f"Increase contrarian weight in DeepSeek prompt from current to +2",
                root_cause=f"Picking with public {public_alignment_pct:.0%} of time (target: <{target_max_public:.0%})"
            )
            return drift

        return None


class HomeFavoriteBiasMonitor(BaseMonitor):
    """
    Monitors home favorite bias (the original problem we fixed).

    IDEAL STATE:
    - Home favorite picks: 40-50%
    - Home underdog picks: 20-30%
    - Away picks: 30-40%

    DRIFT:
    - Home favorite picks >60%
    - Regression to old bias

    AUTO-REMEDIATION:
    - Alert on bias regression
    - Flag for DeepSeek reanalysis with contrarian data
    - Increase contrarian weight
    """

    def __init__(self):
        super().__init__(name="home_favorite_bias", check_interval_seconds=3600)
        self.tracker = BankrollTracker()

    def get_ideal_state(self) -> Dict:
        return {
            'target_home_favorite_pct': 0.45,  # 40-50%
            'max_home_favorite_pct': 0.60,  # Red flag at 60%
            'target_balance': 'balanced'
        }

    def check_for_drift(self) -> Optional[Drift]:
        """Check if regressing to home favorite bias"""
        bet_log_file = Path(__file__).parent / "data" / "bet_log.json"

        if not bet_log_file.exists():
            return None

        with open(bet_log_file, 'r') as f:
            bets = json.load(f)

        # Filter to recent bets (last 20)
        recent_bets = bets[-20:] if len(bets) > 20 else bets

        # Count home favorites
        home_favorites = 0
        total = 0

        for bet in recent_bets:
            pick = bet.get('pick', '')
            game = bet.get('game', '')

            if not pick or not game:
                continue

            total += 1

            # Determine if home team
            parts = game.split('@')
            if len(parts) == 2:
                home_team = parts[1].strip()

                # Check if pick is on home team AND they're favorite (negative spread)
                if home_team in pick and ('-' in pick or 'favorite' in pick.lower()):
                    home_favorites += 1

        if total == 0:
            return None

        home_favorite_pct = home_favorites / total
        target_max = 0.60

        # Check for drift
        if home_favorite_pct > target_max:
            drift_pct = ((home_favorite_pct - target_max) / target_max) * 100

            drift = Drift(
                metric_name="home_favorite_bias",
                current_value=home_favorite_pct,
                target_value=0.45,  # Ideal target
                drift_amount=home_favorite_pct - 0.45,
                severity=self.get_drift_severity(drift_pct),
                detected_at=datetime.now().isoformat(),
                can_auto_fix=True,
                remediation_action="CRITICAL: Home favorite bias detected - Increase contrarian analysis weight",
                root_cause=f"Picking home favorites {home_favorite_pct:.0%} (target: 40-50%, max: 60%)"
            )
            return drift

        return None


class ROIPerformanceMonitor(BaseMonitor):
    """
    Monitors ROI to detect performance drift.

    IDEAL STATE:
    - ROI: 42-47% (with contrarian)
    - Win rate: 60%+
    - Consistent performance

    DRIFT:
    - ROI drops below 35%
    - Win rate drops below 55%
    - Variance spike

    AUTO-REMEDIATION:
    - Trigger model retraining
    - Tighten confidence thresholds
    - Activate circuit breaker if critical
    """

    def __init__(self):
        super().__init__(name="roi_performance", check_interval_seconds=3600)
        self.tracker = BankrollTracker()

    def get_ideal_state(self) -> Dict:
        return {
            'target_roi': 0.42,  # 42% with contrarian
            'min_roi': 0.35,  # Red flag below 35%
            'target_win_rate': 0.60,
            'min_win_rate': 0.55
        }

    def check_for_drift(self) -> Optional[Drift]:
        """Check if ROI is drifting from target"""
        stats = self.tracker.get_stats()

        current_roi = stats['roi'] / 100  # Convert to decimal
        current_wr = stats['win_rate']
        num_bets = stats['total_bets']

        # Need at least 20 bets for meaningful stats
        if num_bets < 20:
            return None

        target_roi = 0.42
        min_roi = 0.35

        # Check for drift
        if current_roi < min_roi:
            drift_pct = ((target_roi - current_roi) / target_roi) * 100

            # Diagnose root cause
            if current_wr < 0.55:
                root_cause = "Win rate degradation - Models may need retraining"
                remediation = "Trigger model retraining on recent data OR tighten confidence thresholds"
            else:
                root_cause = "Odds/line value issues - Getting bad closing line value"
                remediation = "Improve line shopping OR adjust bet timing to earlier in week"

            drift = Drift(
                metric_name="roi_performance_drift",
                current_value=current_roi,
                target_value=target_roi,
                drift_amount=target_roi - current_roi,
                severity=self.get_drift_severity(drift_pct),
                detected_at=datetime.now().isoformat(),
                can_auto_fix=False,  # Requires human decision
                remediation_action=remediation,
                root_cause=root_cause
            )
            return drift

        return None


class NFLSystemMonitor:
    """
    Main orchestrator for continuous system monitoring.

    Runs all monitors and handles drift detection/remediation.
    """

    def __init__(self):
        self.monitors = [
            LineMovementMonitor(),
            CLVMonitor(),
            ContrarianBiasMonitor(),
            HomeFavoriteBiasMonitor(),
            ROIPerformanceMonitor()
        ]

        self.alert_file = Path(__file__).parent / "data" / "monitoring" / "alerts.json"
        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

    def run_single_check(self) -> List[Drift]:
        """Run all monitors once and return detected drifts"""
        drifts = []

        print("=" * 70)
        print("🔍 NFL SYSTEM MONITOR - Running Checks")
        print("=" * 70)
        print()

        for monitor in self.monitors:
            print(f"Checking {monitor.name}...", end=" ")

            drift = monitor.check_for_drift()

            if drift:
                print(f"❌ DRIFT DETECTED")
                monitor.log_drift(drift)
                drifts.append(drift)
                self._alert_drift(drift)
            else:
                print(f"✅ OK")

        print()
        return drifts

    def run_continuous_monitoring(self, check_interval_seconds: int = 300):
        """Run continuous monitoring loop"""
        print("=" * 70)
        print("🚀 NFL SYSTEM MONITOR - Starting Continuous Monitoring")
        print("=" * 70)
        print()
        print(f"Check interval: {check_interval_seconds} seconds")
        print(f"Monitoring: {len(self.monitors)} metrics")
        print()
        print("Press Ctrl+C to stop")
        print()

        try:
            while True:
                drifts = self.run_single_check()

                if drifts:
                    print(f"⚠️  Detected {len(drifts)} drift(s)")
                    for drift in drifts:
                        print(f"   - {drift.metric_name}: {drift.severity}")
                else:
                    print("✅ All systems normal")

                print(f"\nNext check in {check_interval_seconds} seconds...")
                print()

                time.sleep(check_interval_seconds)

        except KeyboardInterrupt:
            print("\n\n🛑 Monitoring stopped by user")

    def _alert_drift(self, drift: Drift):
        """Alert on detected drift"""
        # Load existing alerts
        if self.alert_file.exists():
            with open(self.alert_file, 'r') as f:
                alerts = json.load(f)
        else:
            alerts = []

        # Add new alert
        alert = {
            'metric': drift.metric_name,
            'severity': drift.severity,
            'current_value': drift.current_value,
            'target_value': drift.target_value,
            'drift_amount': drift.drift_amount,
            'detected_at': drift.detected_at,
            'remediation': drift.remediation_action,
            'root_cause': drift.root_cause,
            'can_auto_fix': drift.can_auto_fix
        }

        alerts.append(alert)

        # Save alerts
        with open(self.alert_file, 'w') as f:
            json.dump(alerts, f, indent=2)

        # Print alert
        print()
        print("=" * 70)
        print(f"🚨 DRIFT ALERT: {drift.metric_name}")
        print("=" * 70)
        print(f"Severity: {drift.severity}")
        print(f"Current: {drift.current_value:.2f}")
        print(f"Target: {drift.target_value:.2f}")
        print(f"Drift: {drift.drift_amount:+.2f}")
        print()
        print(f"Root Cause: {drift.root_cause}")
        print(f"Remediation: {drift.remediation_action}")
        print()
        if drift.can_auto_fix:
            print("✅ Can auto-fix")
        else:
            print("⚠️  Requires manual intervention")
        print("=" * 70)
        print()

    def get_status(self):
        """Get current monitoring status"""
        print("=" * 70)
        print("📊 NFL SYSTEM MONITOR - Current Status")
        print("=" * 70)
        print()

        # Show ideal state for each monitor
        for monitor in self.monitors:
            print(f"📍 {monitor.name.upper()}")
            ideal_state = monitor.get_ideal_state()
            for key, value in ideal_state.items():
                print(f"   {key}: {value}")
            print()

        # Show recent drifts
        if self.alert_file.exists():
            with open(self.alert_file, 'r') as f:
                alerts = json.load(f)

            recent_alerts = alerts[-10:] if len(alerts) > 10 else alerts

            if recent_alerts:
                print("🚨 RECENT ALERTS (Last 10):")
                print()
                for alert in recent_alerts:
                    print(f"   • {alert['metric']} ({alert['severity']})")
                    print(f"     {alert['detected_at']}")
                    print(f"     {alert['remediation']}")
                    print()
            else:
                print("✅ No recent alerts")
        else:
            print("✅ No alerts yet")


def main():
    """CLI interface"""
    import argparse

    parser = argparse.ArgumentParser(description="NFL System Monitor")
    parser.add_argument('--start', action='store_true',
                       help='Start continuous monitoring')
    parser.add_argument('--check-all', action='store_true',
                       help='Run single check across all monitors')
    parser.add_argument('--status', action='store_true',
                       help='Show current monitoring status')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300)')

    args = parser.parse_args()

    monitor = NFLSystemMonitor()

    if args.start:
        monitor.run_continuous_monitoring(check_interval_seconds=args.interval)
    elif args.check_all:
        monitor.run_single_check()
    elif args.status:
        monitor.get_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
