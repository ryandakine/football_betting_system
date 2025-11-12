#!/usr/bin/env python3
"""
System Optimizer - Full betting system health check and optimization

Analyzes:
1. Current performance metrics
2. Workflow bottlenecks
3. ROI optimization opportunities
4. Risk management effectiveness
5. Automation gaps
6. Resource allocation

Outputs:
- System health score
- Optimization recommendations
- Priority action items
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    from bankroll_tracker import BankrollTracker
    from circuit_breaker import CircuitBreaker
except ImportError:
    print("⚠️  Some modules not found - running in limited mode")


@dataclass
class SystemMetrics:
    """System performance metrics"""
    total_bets: int = 0
    win_rate: float = 0.0
    roi: float = 0.0
    current_bankroll: float = 0.0
    starting_bankroll: float = 100.0
    total_wagered: float = 0.0
    total_profit: float = 0.0
    avg_bet_size: float = 0.0
    largest_bet: float = 0.0
    smallest_bet: float = 0.0

    # Risk metrics
    max_drawdown: float = 0.0
    current_streak: int = 0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0

    # Efficiency metrics
    bets_per_week: float = 0.0
    clv_capture_rate: float = 0.0

    # System health
    circuit_breaker_triggers: int = 0
    contrarian_filter_blocks: int = 0
    trap_detector_warnings: int = 0


class SystemOptimizer:
    """Analyzes and optimizes the betting system"""

    def __init__(self):
        self.base_path = Path(__file__).parent
        self.data_path = self.base_path / "data"
        self.bet_log_path = self.data_path / "bet_log.json"
        self.bankroll_path = self.base_path / ".bankroll"

    def load_bet_history(self) -> List[Dict]:
        """Load bet history from log file"""
        if not self.bet_log_path.exists():
            return []

        with open(self.bet_log_path, 'r') as f:
            return json.load(f)

    def calculate_metrics(self) -> SystemMetrics:
        """Calculate comprehensive system metrics"""
        bets = self.load_bet_history()
        metrics = SystemMetrics()

        if not bets:
            return metrics

        # Basic metrics
        metrics.total_bets = len(bets)
        wins = sum(1 for b in bets if b.get('result') == 'WIN')
        losses = sum(1 for b in bets if b.get('result') == 'LOSS')

        if metrics.total_bets > 0:
            metrics.win_rate = wins / metrics.total_bets * 100

        # Financial metrics
        metrics.total_wagered = sum(b.get('amount', 0) for b in bets)

        if metrics.total_wagered > 0:
            metrics.avg_bet_size = metrics.total_wagered / metrics.total_bets

        bet_amounts = [b.get('amount', 0) for b in bets]
        if bet_amounts:
            metrics.largest_bet = max(bet_amounts)
            metrics.smallest_bet = min(bet_amounts)

        # Get current bankroll
        if self.bankroll_path.exists():
            with open(self.bankroll_path, 'r') as f:
                bankroll_data = json.load(f)
                metrics.current_bankroll = bankroll_data.get('current_bankroll', 100.0)

        # Calculate profit and ROI
        metrics.total_profit = metrics.current_bankroll - metrics.starting_bankroll

        if metrics.total_wagered > 0:
            metrics.roi = (metrics.total_profit / metrics.total_wagered) * 100

        # Streak analysis
        current_streak = 0
        longest_win = 0
        longest_loss = 0
        temp_win = 0
        temp_loss = 0

        for bet in reversed(bets):  # Most recent first
            result = bet.get('result')
            if result == 'WIN':
                temp_win += 1
                temp_loss = 0
                if current_streak == 0:
                    current_streak = temp_win
            elif result == 'LOSS':
                temp_loss += 1
                temp_win = 0
                if current_streak == 0:
                    current_streak = -temp_loss

            longest_win = max(longest_win, temp_win)
            longest_loss = max(longest_loss, temp_loss)

        metrics.current_streak = current_streak
        metrics.longest_win_streak = longest_win
        metrics.longest_loss_streak = longest_loss

        return metrics

    def analyze_workflow_efficiency(self) -> Dict[str, any]:
        """Analyze workflow for bottlenecks and inefficiencies"""
        analysis = {
            "automation_level": 0.0,
            "manual_steps": [],
            "bottlenecks": [],
            "optimization_opportunities": []
        }

        # Check what's automated
        automated_components = []
        manual_steps = []

        # Check if key files exist
        components = {
            "auto_execute_bets.py": "Main workflow orchestrator",
            "bankroll_tracker.py": "Bankroll tracking",
            "circuit_breaker.py": "Risk management",
            "contrarian_intelligence.py": "Contrarian filter",
            "trap_detector.py": "Trap detection",
            "bet_validator.py": "Mock data validation",
            "line_shopper.py": "Line shopping",
            "referee_fetcher.py": "Referee analysis",
        }

        for file, description in components.items():
            if (self.base_path / file).exists():
                automated_components.append(description)
            else:
                manual_steps.append(f"Missing: {description}")

        analysis["automation_level"] = len(automated_components) / len(components) * 100
        analysis["automated_components"] = automated_components
        analysis["manual_steps"] = manual_steps

        # Identify bottlenecks
        bets = self.load_bet_history()
        if len(bets) < 10:
            analysis["bottlenecks"].append("Low bet volume - may indicate workflow friction")

        # Optimization opportunities
        if len(bets) > 0:
            avg_bet = sum(b.get('amount', 0) for b in bets) / len(bets)
            if avg_bet < 5:
                analysis["optimization_opportunities"].append(
                    f"Average bet size is ${avg_bet:.2f} - consider Kelly Criterion for optimal sizing"
                )

        return analysis

    def check_risk_management(self) -> Dict[str, any]:
        """Analyze risk management effectiveness"""
        metrics = self.calculate_metrics()

        risk_analysis = {
            "risk_score": 0,  # 0-100, lower is better
            "warnings": [],
            "recommendations": []
        }

        # Check bet sizing
        if metrics.total_bets > 0:
            if metrics.largest_bet > metrics.starting_bankroll * 0.1:
                risk_analysis["warnings"].append(
                    f"Largest bet (${metrics.largest_bet:.2f}) exceeds 10% of starting bankroll"
                )
                risk_analysis["risk_score"] += 20

            # Check average bet size relative to bankroll
            if metrics.current_bankroll > 0:
                bet_pct = (metrics.avg_bet_size / metrics.current_bankroll) * 100
                if bet_pct > 5:
                    risk_analysis["warnings"].append(
                        f"Average bet is {bet_pct:.1f}% of current bankroll (should be 1-3%)"
                    )
                    risk_analysis["risk_score"] += 15

        # Check for loss streaks
        if metrics.longest_loss_streak > 5:
            risk_analysis["warnings"].append(
                f"Longest loss streak: {metrics.longest_loss_streak} bets"
            )
            risk_analysis["risk_score"] += 10

        # Check drawdown
        if metrics.total_profit < -20:
            risk_analysis["warnings"].append(
                f"Significant drawdown: ${abs(metrics.total_profit):.2f}"
            )
            risk_analysis["risk_score"] += 25

        # Recommendations
        if risk_analysis["risk_score"] < 30:
            risk_analysis["recommendations"].append("✅ Risk management is solid")
        elif risk_analysis["risk_score"] < 60:
            risk_analysis["recommendations"].append("⚠️ Consider reducing bet sizes")
        else:
            risk_analysis["recommendations"].append("🚨 Reduce position sizes immediately")

        return risk_analysis

    def calculate_optimization_score(self) -> float:
        """Calculate overall system optimization score (0-100)"""
        metrics = self.calculate_metrics()
        workflow = self.analyze_workflow_efficiency()
        risk = self.check_risk_management()

        score = 0.0

        # ROI component (30 points max)
        if metrics.roi > 30:
            score += 30
        elif metrics.roi > 15:
            score += 25
        elif metrics.roi > 5:
            score += 15
        elif metrics.roi > 0:
            score += 5

        # Automation component (25 points max)
        score += workflow["automation_level"] * 0.25

        # Risk management component (25 points max)
        risk_score = max(0, 100 - risk["risk_score"])
        score += risk_score * 0.25

        # Win rate component (20 points max)
        if metrics.win_rate > 60:
            score += 20
        elif metrics.win_rate > 55:
            score += 15
        elif metrics.win_rate > 52:
            score += 10
        elif metrics.win_rate > 50:
            score += 5

        return min(100, score)

    def generate_report(self) -> str:
        """Generate comprehensive optimization report"""
        metrics = self.calculate_metrics()
        workflow = self.analyze_workflow_efficiency()
        risk = self.check_risk_management()
        score = self.calculate_optimization_score()

        report = []
        report.append("=" * 80)
        report.append("🎯 BETTING SYSTEM OPTIMIZATION REPORT")
        report.append("=" * 80)
        report.append("")

        # Overall Score
        report.append(f"📊 SYSTEM HEALTH SCORE: {score:.1f}/100")
        if score >= 80:
            report.append("   Status: ✅ EXCELLENT - System is highly optimized")
        elif score >= 60:
            report.append("   Status: ⚠️ GOOD - Room for improvement")
        elif score >= 40:
            report.append("   Status: 🔶 FAIR - Needs optimization")
        else:
            report.append("   Status: 🚨 POOR - Immediate action required")
        report.append("")

        # Performance Metrics
        report.append("📈 PERFORMANCE METRICS")
        report.append("─" * 80)
        report.append(f"Current Bankroll:    ${metrics.current_bankroll:.2f}")
        report.append(f"Starting Bankroll:   ${metrics.starting_bankroll:.2f}")
        report.append(f"Total Profit/Loss:   ${metrics.total_profit:+.2f}")
        report.append(f"Total Bets:          {metrics.total_bets}")
        report.append(f"Win Rate:            {metrics.win_rate:.1f}%")
        report.append(f"ROI:                 {metrics.roi:+.1f}%")
        report.append(f"Total Wagered:       ${metrics.total_wagered:.2f}")
        report.append(f"Average Bet Size:    ${metrics.avg_bet_size:.2f}")
        report.append("")

        # Streaks
        if metrics.total_bets > 0:
            report.append("📊 STREAK ANALYSIS")
            report.append("─" * 80)
            streak_emoji = "🔥" if metrics.current_streak > 0 else "❄️"
            report.append(f"Current Streak:      {streak_emoji} {metrics.current_streak:+d}")
            report.append(f"Longest Win Streak:  {metrics.longest_win_streak}")
            report.append(f"Longest Loss Streak: {metrics.longest_loss_streak}")
            report.append("")

        # Workflow Efficiency
        report.append("⚙️ WORKFLOW EFFICIENCY")
        report.append("─" * 80)
        report.append(f"Automation Level:    {workflow['automation_level']:.0f}%")
        report.append(f"Automated Components: {len(workflow['automated_components'])}/{len(workflow['automated_components']) + len(workflow['manual_steps'])}")
        if workflow['manual_steps']:
            report.append("\nManual Steps:")
            for step in workflow['manual_steps']:
                report.append(f"  • {step}")
        report.append("")

        # Risk Management
        report.append("🛡️ RISK MANAGEMENT")
        report.append("─" * 80)
        report.append(f"Risk Score:          {risk['risk_score']}/100 (lower is better)")
        if risk['warnings']:
            report.append("\n⚠️ Warnings:")
            for warning in risk['warnings']:
                report.append(f"  • {warning}")
        if risk['recommendations']:
            report.append("\n💡 Recommendations:")
            for rec in risk['recommendations']:
                report.append(f"  • {rec}")
        report.append("")

        # Optimization Opportunities
        report.append("🚀 OPTIMIZATION OPPORTUNITIES")
        report.append("─" * 80)

        opportunities = []

        # ROI optimization
        if metrics.roi < 10 and metrics.total_bets > 5:
            opportunities.append({
                "priority": "HIGH",
                "area": "ROI Improvement",
                "issue": f"Current ROI is {metrics.roi:.1f}% (target: 10-20%)",
                "action": "Review pick quality, line shopping, and contrarian filtering"
            })

        # Bet sizing
        if metrics.current_bankroll > 0:
            optimal_bet = metrics.current_bankroll * 0.02  # 2% Kelly
            if abs(metrics.avg_bet_size - optimal_bet) > optimal_bet * 0.5:
                opportunities.append({
                    "priority": "MEDIUM",
                    "area": "Bet Sizing",
                    "issue": f"Average bet ${metrics.avg_bet_size:.2f} vs optimal ${optimal_bet:.2f}",
                    "action": "Implement Kelly Criterion for bet sizing"
                })

        # Volume
        if metrics.total_bets < 10:
            opportunities.append({
                "priority": "LOW",
                "area": "Betting Volume",
                "issue": f"Only {metrics.total_bets} bets placed",
                "action": "Increase volume to get statistical significance"
            })

        # Line shopping
        opportunities.append({
            "priority": "MEDIUM",
            "area": "Line Shopping",
            "issue": "CLV capture rate not tracked",
            "action": "Fund 3-5 sportsbooks for optimal line shopping (see strategic_line_shopping.py)"
        })

        # Automation
        if workflow['manual_steps']:
            opportunities.append({
                "priority": "LOW",
                "area": "Automation",
                "issue": f"{len(workflow['manual_steps'])} manual steps remaining",
                "action": "Complete automation of all workflow components"
            })

        # Sort by priority
        priority_order = {"HIGH": 1, "MEDIUM": 2, "LOW": 3}
        opportunities.sort(key=lambda x: priority_order[x['priority']])

        for i, opp in enumerate(opportunities, 1):
            priority_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[opp['priority']]
            report.append(f"\n{i}. {priority_emoji} {opp['priority']} - {opp['area']}")
            report.append(f"   Issue:  {opp['issue']}")
            report.append(f"   Action: {opp['action']}")

        report.append("")
        report.append("=" * 80)

        return "\n".join(report)

    def generate_action_plan(self) -> List[Dict]:
        """Generate prioritized action items"""
        metrics = self.calculate_metrics()
        workflow = self.analyze_workflow_efficiency()
        risk = self.check_risk_management()

        actions = []

        # High priority actions
        if risk['risk_score'] > 60:
            actions.append({
                "priority": 1,
                "title": "🚨 Reduce Risk Immediately",
                "description": "Risk score is too high - reduce bet sizes to 1-2% of bankroll",
                "command": "Review circuit_breaker.py settings"
            })

        if metrics.roi < 0 and metrics.total_bets > 10:
            actions.append({
                "priority": 1,
                "title": "🔴 Fix Negative ROI",
                "description": "System is losing money - review pick quality and filters",
                "command": "python auto_execute_bets.py --auto (verify contrarian filter is working)"
            })

        # Medium priority
        if metrics.current_bankroll > 0 and metrics.avg_bet_size > metrics.current_bankroll * 0.05:
            actions.append({
                "priority": 2,
                "title": "⚠️ Implement Kelly Criterion",
                "description": "Bet sizing not optimal - implement fractional Kelly",
                "command": "Add Kelly sizing to auto_execute_bets.py"
            })

        actions.append({
            "priority": 2,
            "title": "💰 Optimize Line Shopping",
            "description": "Fund 3-5 strategic sportsbooks to capture best lines",
            "command": "python strategic_line_shopping.py --funding-plan YOUR_BUDGET"
        })

        # Low priority
        if metrics.total_bets < 20:
            actions.append({
                "priority": 3,
                "title": "📊 Increase Sample Size",
                "description": "Need more bets for statistical significance",
                "command": "Continue betting with current strategy"
            })

        if workflow['automation_level'] < 100:
            actions.append({
                "priority": 3,
                "title": "⚙️ Complete Automation",
                "description": "Some workflow steps still manual",
                "command": "Review workflow and automate remaining steps"
            })

        return sorted(actions, key=lambda x: x['priority'])


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Optimize betting system")
    parser.add_argument("--report", action="store_true", help="Generate full optimization report")
    parser.add_argument("--metrics", action="store_true", help="Show performance metrics only")
    parser.add_argument("--actions", action="store_true", help="Show prioritized action items")
    parser.add_argument("--score", action="store_true", help="Show optimization score only")

    args = parser.parse_args()

    optimizer = SystemOptimizer()

    if args.score:
        score = optimizer.calculate_optimization_score()
        print(f"\n🎯 System Optimization Score: {score:.1f}/100\n")

    elif args.metrics:
        metrics = optimizer.calculate_metrics()
        print("\n📈 PERFORMANCE METRICS")
        print("=" * 50)
        print(f"Bankroll:     ${metrics.current_bankroll:.2f}")
        print(f"Profit/Loss:  ${metrics.total_profit:+.2f}")
        print(f"ROI:          {metrics.roi:+.1f}%")
        print(f"Win Rate:     {metrics.win_rate:.1f}%")
        print(f"Total Bets:   {metrics.total_bets}")
        print()

    elif args.actions:
        actions = optimizer.generate_action_plan()
        print("\n🎯 PRIORITIZED ACTION ITEMS")
        print("=" * 80)
        for action in actions:
            priority_emoji = {1: "🔴 HIGH", 2: "🟡 MEDIUM", 3: "🟢 LOW"}[action['priority']]
            print(f"\n{priority_emoji} - {action['title']}")
            print(f"  {action['description']}")
            print(f"  → {action['command']}")
        print()

    else:
        # Full report
        report = optimizer.generate_report()
        print("\n" + report)

        print("\n💡 Quick Actions:")
        actions = optimizer.generate_action_plan()[:3]  # Top 3
        for i, action in enumerate(actions, 1):
            print(f"\n{i}. {action['title']}")
            print(f"   → {action['command']}")
        print()


if __name__ == "__main__":
    main()
