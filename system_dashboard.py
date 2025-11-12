#!/usr/bin/env python3
"""
System Dashboard - Real-time betting system health monitoring

Shows at a glance:
- Current system status
- Performance metrics
- Active risks
- Optimization opportunities
- Quick actions

USAGE:
    python system_dashboard.py
    python system_dashboard.py --watch  # Auto-refresh every 5 seconds
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    from bankroll_tracker import BankrollTracker
    from circuit_breaker import CircuitBreaker
    from kelly_criterion import KellyCriterion
    from system_optimizer import SystemOptimizer
except ImportError as e:
    print(f"❌ Missing module: {e}")
    sys.exit(1)


class SystemDashboard:
    """Real-time system health dashboard"""

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.bankroll_tracker = BankrollTracker()
        self.circuit_breaker = CircuitBreaker()
        self.optimizer = SystemOptimizer()
        self.kelly = KellyCriterion()

    def get_status_indicator(self, score: float) -> str:
        """Get status indicator based on score"""
        if score >= 80:
            return "🟢 EXCELLENT"
        elif score >= 60:
            return "🟡 GOOD"
        elif score >= 40:
            return "🟠 FAIR"
        else:
            return "🔴 POOR"

    def get_roi_indicator(self, roi: float) -> str:
        """Get ROI status indicator"""
        if roi >= 15:
            return "🚀"
        elif roi >= 5:
            return "✅"
        elif roi >= 0:
            return "➡️"
        elif roi >= -10:
            return "⚠️"
        else:
            return "🚨"

    def get_circuit_breaker_status(self) -> Dict:
        """Get circuit breaker status"""
        try:
            status = self.circuit_breaker.get_status()
            return {
                "enabled": status.get("enabled", True),
                "triggered": status.get("triggered", False),
                "bets_today": status.get("bets_today", 0),
                "max_bets": status.get("max_daily_bets", 3),
            }
        except Exception:
            return {
                "enabled": True,
                "triggered": False,
                "bets_today": 0,
                "max_bets": 3,
            }

    def render_dashboard(self) -> str:
        """Render the dashboard"""
        metrics = self.optimizer.calculate_metrics()
        workflow = self.optimizer.analyze_workflow_efficiency()
        risk = self.optimizer.check_risk_management()
        score = self.optimizer.calculate_optimization_score()
        cb_status = self.get_circuit_breaker_status()

        # Calculate optimal bet size
        optimal_bet = None
        if metrics.current_bankroll > 0:
            kelly_result = self.kelly.calculate_kelly(
                win_probability=0.55,  # Assume 55% win rate
                odds=-110,  # Standard odds
                bankroll=metrics.current_bankroll,
                fraction=0.25  # Quarter Kelly
            )
            if kelly_result:
                optimal_bet = kelly_result.fractional_kelly_amount

        output = []

        # Header
        output.append("╔" + "═" * 78 + "╗")
        output.append("║" + " " * 22 + "🎯 BETTING SYSTEM DASHBOARD" + " " * 28 + "║")
        output.append("║" + " " * 25 + f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}" + " " * 29 + "║")
        output.append("╚" + "═" * 78 + "╝")
        output.append("")

        # System Health (Top Row)
        output.append("┌" + "─" * 78 + "┐")
        status = self.get_status_indicator(score)
        output.append(f"│ 📊 SYSTEM HEALTH: {status:<25} Score: {score:.1f}/100{' ' * 18}│")
        output.append("└" + "─" * 78 + "┘")
        output.append("")

        # Quick Stats (3 columns)
        output.append("┌" + "─" * 24 + "┬" + "─" * 26 + "┬" + "─" * 26 + "┐")
        output.append(f"│ 💰 BANKROLL           │ 📈 PERFORMANCE         │ 🎲 BETS TODAY          │")
        output.append("├" + "─" * 24 + "┼" + "─" * 26 + "┼" + "─" * 26 + "┤")

        # Row 1
        bankroll_str = f"${metrics.current_bankroll:.2f}"
        profit_str = f"${metrics.total_profit:+.2f}"
        roi_indicator = self.get_roi_indicator(metrics.roi)
        bets_today_str = f"{cb_status['bets_today']}/{cb_status['max_bets']}"

        output.append(f"│ Current: {bankroll_str:<14}│ P/L: {profit_str:<16}│ Placed: {bets_today_str:<14}│")

        # Row 2
        if optimal_bet:
            optimal_str = f"${optimal_bet:.2f}"
        else:
            optimal_str = "N/A"

        roi_str = f"{metrics.roi:+.1f}%"
        winrate_str = f"{metrics.win_rate:.1f}%"
        cb_emoji = "✅" if not cb_status['triggered'] else "🚫"

        output.append(f"│ Next Bet: {optimal_str:<13}│ ROI: {roi_indicator} {roi_str:<13}│ Breaker: {cb_emoji:<13}│")

        # Row 3
        streak_emoji = "🔥" if metrics.current_streak > 0 else "❄️"
        streak_str = f"{metrics.current_streak:+d}"

        output.append(f"│ Total Bets: {metrics.total_bets:<11}│ Win Rate: {winrate_str:<14}│ Streak: {streak_emoji} {streak_str:<11}│")

        output.append("└" + "─" * 24 + "┴" + "─" * 26 + "┴" + "─" * 26 + "┘")
        output.append("")

        # Risk Management
        output.append("┌" + "─" * 78 + "┐")
        risk_color = "🟢" if risk['risk_score'] < 30 else "🟡" if risk['risk_score'] < 60 else "🔴"
        output.append(f"│ 🛡️  RISK MANAGEMENT {risk_color:<58}│")
        output.append("├" + "─" * 78 + "┤")

        risk_status = "HEALTHY" if risk['risk_score'] < 30 else "MODERATE" if risk['risk_score'] < 60 else "HIGH"
        output.append(f"│ Status: {risk_status:<35} Risk Score: {risk['risk_score']}/100{' ' * 16}│")

        if risk['warnings']:
            output.append("│ " + " " * 76 + "│")
            output.append(f"│ ⚠️  Warnings:{' ' * 64}│")
            for warning in risk['warnings'][:2]:  # Show top 2
                warning_short = warning[:72] if len(warning) <= 72 else warning[:69] + "..."
                output.append(f"│   • {warning_short:<73}│")

        output.append("└" + "─" * 78 + "┘")
        output.append("")

        # Workflow Status
        output.append("┌" + "─" * 78 + "┐")
        output.append(f"│ ⚙️  WORKFLOW AUTOMATION{' ' * 54}│")
        output.append("├" + "─" * 78 + "┤")

        auto_pct = workflow['automation_level']
        auto_bar = self._render_progress_bar(auto_pct, 40)
        output.append(f"│ Automation: {auto_bar} {auto_pct:.0f}%{' ' * 10}│")

        active_components = len(workflow['automated_components'])
        output.append(f"│ Active Components: {active_components}/8{' ' * 54}│")

        output.append("└" + "─" * 78 + "┘")
        output.append("")

        # Recent Activity
        output.append("┌" + "─" * 78 + "┐")
        output.append(f"│ 📝 RECENT ACTIVITY{' ' * 59}│")
        output.append("├" + "─" * 78 + "┤")

        bets = self.optimizer.load_bet_history()
        if bets:
            for bet in bets[-3:]:  # Last 3 bets
                game = bet.get('game', 'Unknown')[:20]
                pick = bet.get('pick', 'Unknown')[:15]
                result = bet.get('result', 'PENDING')
                amount = bet.get('amount', 0)

                result_emoji = "✅" if result == "WIN" else "❌" if result == "LOSS" else "⏳"

                line = f"│ {result_emoji} {game:<20} {pick:<15} ${amount:>5.2f}{' ' * 12}│"
                output.append(line)
        else:
            output.append(f"│ No bets recorded yet{' ' * 57}│")

        output.append("└" + "─" * 78 + "┘")
        output.append("")

        # Top Priority Actions
        actions = self.optimizer.generate_action_plan()[:3]
        if actions:
            output.append("┌" + "─" * 78 + "┐")
            output.append(f"│ 🎯 PRIORITY ACTIONS{' ' * 58}│")
            output.append("├" + "─" * 78 + "┤")

            for i, action in enumerate(actions, 1):
                priority_emoji = {1: "🔴", 2: "🟡", 3: "🟢"}[action['priority']]
                title = action['title'][:50]
                output.append(f"│ {i}. {priority_emoji} {title:<69}│")

            output.append("└" + "─" * 78 + "┘")
            output.append("")

        # Footer with commands
        output.append("┌" + "─" * 78 + "┐")
        output.append(f"│ 💡 QUICK COMMANDS{' ' * 60}│")
        output.append("├" + "─" * 78 + "┤")
        output.append(f"│ Place Bet:        python auto_execute_bets.py --auto{' ' * 22}│")
        output.append(f"│ Full Report:      python system_optimizer.py --report{' ' * 19}│")
        output.append(f"│ Kelly Sizing:     python kelly_criterion.py --odds -110 --win-prob 0.55{' '}│")
        output.append(f"│ Line Shopping:    python strategic_line_shopping.py --funding-plan 100{' ' * 3}│")
        output.append("└" + "─" * 78 + "┘")

        return "\n".join(output)

    def _render_progress_bar(self, percentage: float, width: int = 20) -> str:
        """Render a progress bar"""
        filled = int(width * percentage / 100)
        empty = width - filled
        bar = "█" * filled + "░" * empty
        return f"[{bar}]"


def main():
    import argparse

    parser = argparse.ArgumentParser(description="System dashboard")
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - refresh every 5 seconds"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Refresh interval in seconds (default: 5)"
    )

    args = parser.parse_args()

    dashboard = SystemDashboard()

    if args.watch:
        try:
            while True:
                # Clear screen
                os.system('clear' if os.name != 'nt' else 'cls')

                # Render dashboard
                print(dashboard.render_dashboard())

                # Wait
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\n\n👋 Dashboard stopped")
            sys.exit(0)
    else:
        print(dashboard.render_dashboard())


if __name__ == "__main__":
    main()
