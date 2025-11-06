import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any
import psutil
import sys

class SystemPerformanceAnalyzer:
    def __init__(self):
        self.start_time = datetime.now()
        self.performance_data = self._load_performance_data()
        self.test_results = self._load_test_results()

    def _load_performance_data(self) -> Dict:
        """Load latest performance data"""
        performance_dir = Path("performance_data")
        if not performance_dir.exists():
            return {}

        # Get the most recent session file
        session_files = list(performance_dir.glob("session_*.json"))
        if not session_files:
            return {}

        latest_file = max(session_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'r') as f:
            return json.load(f)

    def _load_test_results(self) -> Dict:
        """Load latest test results"""
        test_results_dir = Path("test_results")
        if not test_results_dir.exists():
            return {}

        # Get the most recent test result
        test_files = list(test_results_dir.glob("fake_money_test_*.json"))
        if not test_files:
            return {}

        latest_file = max(test_files, key=lambda x: x.stat().st_mtime)

        with open(latest_file, 'r') as f:
            return json.load(f)

    def get_system_resources(self) -> Dict[str, Any]:
        """Get current system resource usage"""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids())
        }

    def analyze_betting_performance(self) -> Dict[str, Any]:
        """Analyze betting performance metrics"""
        metrics = self.performance_data.get("metrics", {})

        return {
            "overall_roi": metrics.get("roi", 0),
            "win_rate": metrics.get("win_rate", 0),
            "total_profit": metrics.get("total_profit", 0),
            "total_bets": metrics.get("total_bets", 0),
            "profit_factor": metrics.get("profit_factor", 0),
            "sharpe_ratio": metrics.get("sharpe_ratio", 0),
            "max_drawdown": metrics.get("max_drawdown", 0),
            "expected_value_accuracy": metrics.get("expected_value_accuracy", 0)
        }

    def analyze_test_performance(self) -> Dict[str, Any]:
        """Analyze test system performance"""
        perf_metrics = self.test_results.get("performance_metrics", {})

        return {
            "predictions_made": perf_metrics.get("predictions_made", 0),
            "predictions_correct": perf_metrics.get("predictions_correct", 0),
            "avg_edge_found": perf_metrics.get("avg_edge_found", 0),
            "games_analyzed": perf_metrics.get("games_analyzed", 0),
            "recommendations_generated": perf_metrics.get("recommendations_generated", 0),
            "total_potential_ev": self.test_results.get("summary", {}).get("total_potential_ev", 0)
        }

    def identify_bottlenecks(self) -> List[str]:
        """Identify potential system bottlenecks"""
        bottlenecks = []

        # Check if there are performance issues based on data
        betting_perf = self.analyze_betting_performance()
        test_perf = self.analyze_test_performance()
        system_resources = self.get_system_resources()

        # ROI analysis
        if betting_perf.get("overall_roi", 0) < 5:
            bottlenecks.append("Low ROI performance - system may need better edge detection")

        # Win rate analysis
        if betting_perf.get("win_rate", 0) < 0.5:
            bottlenecks.append("Sub-optimal win rate - consider improving prediction accuracy")

        # Edge analysis
        if test_perf.get("avg_edge_found", 0) < 10:
            bottlenecks.append("Low average edge found - system may need better market analysis")

        # System resource analysis
        if system_resources.get("cpu_percent", 0) > 80:
            bottlenecks.append("High CPU usage - consider optimizing compute-intensive operations")

        if system_resources.get("memory_percent", 0) > 80:
            bottlenecks.append("High memory usage - consider memory optimization techniques")

        # Prediction efficiency
        if test_perf.get("recommendations_generated", 0) == 0:
            bottlenecks.append("No recommendations generated - check data pipeline and model integration")

        return bottlenecks

    def generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generate specific optimization recommendations"""
        recommendations = []

        betting_perf = self.analyze_betting_performance()
        bottlenecks = self.identify_bottlenecks()

        # ROI optimization
        if betting_perf.get("overall_roi", 0) < 5:
            recommendations.append({
                "category": "Betting Strategy",
                "priority": "High",
                "recommendation": "Improve ROI through better market selection and edge detection",
                "action": "Review and enhance the market intelligence algorithms"
            })

        # Win rate improvement
        if betting_perf.get("win_rate", 0) < 0.5:
            recommendations.append({
                "category": "Prediction Accuracy",
                "priority": "High",
                "recommendation": "Increase win rate through better model training and feature engineering",
                "action": "Implement advanced feature engineering and model validation"
            })

        # Edge enhancement
        if "Low average edge found" in bottlenecks:
            recommendations.append({
                "category": "Market Analysis",
                "priority": "Medium",
                "recommendation": "Enhance edge detection algorithms for better value identification",
                "action": "Integrate additional data sources and improve market timing"
            })

        # System performance
        if "High CPU usage" in bottlenecks:
            recommendations.append({
                "category": "System Optimization",
                "priority": "Medium",
                "recommendation": "Optimize CPU-intensive operations",
                "action": "Implement caching, parallel processing, and code profiling"
            })

        if "High memory usage" in bottlenecks:
            recommendations.append({
                "category": "Memory Management",
                "priority": "Medium",
                "recommendation": "Optimize memory usage patterns",
                "action": "Implement memory pooling, data structure optimization, and cleanup routines"
            })

        # Data pipeline
        if "No recommendations generated" in bottlenecks:
            recommendations.append({
                "category": "Data Pipeline",
                "priority": "High",
                "recommendation": "Fix data pipeline and model integration issues",
                "action": "Debug data collection, model inference, and recommendation generation"
            })

        return recommendations

    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        betting_perf = self.analyze_betting_performance()
        test_perf = self.analyze_test_performance()
        system_resources = self.get_system_resources()
        bottlenecks = self.identify_bottlenecks()
        recommendations = self.generate_optimization_recommendations()

        report = []
        report.append("=" * 80)
        report.append("📊 FOOTBALL BETTING SYSTEM PERFORMANCE ANALYSIS")
        report.append("=" * 80)

        # System Resources
        report.append("\n🔧 SYSTEM RESOURCES:")
        report.append(f"  CPU Usage: {system_resources['cpu_percent']:.1f}%")
        report.append(f"  Memory Usage: {system_resources['memory_percent']:.1f}%")
        report.append(f"  Disk Usage: {system_resources['disk_usage']:.1f}%")
        report.append(f"  Active Processes: {system_resources['process_count']}")

        # Betting Performance
        report.append("\n🎯 BETTING PERFORMANCE:")
        report.append(f"  Overall ROI: {betting_perf['overall_roi']:.2f}%")
        report.append(f"  Win Rate: {betting_perf['win_rate']:.1%}")
        report.append(f"  Total Profit: ${betting_perf['total_profit']:.2f}")
        report.append(f"  Total Bets: {betting_perf['total_bets']}")
        report.append(f"  Profit Factor: {betting_perf['profit_factor']:.2f}")
        report.append(f"  Sharpe Ratio: {betting_perf['sharpe_ratio']:.2f}")
        report.append(f"  Max Drawdown: {betting_perf['max_drawdown']:.1%}")
        report.append(f"  EV Accuracy: {betting_perf['expected_value_accuracy']:.1f}%")

        # Test System Performance
        report.append("\n🧪 TEST SYSTEM PERFORMANCE:")
        report.append(f"  Predictions Made: {test_perf['predictions_made']}")
        report.append(f"  Predictions Correct: {test_perf['predictions_correct']}")
        report.append(f"  Average Edge Found: {test_perf['avg_edge_found']:.2f}%")
        report.append(f"  Games Analyzed: {test_perf['games_analyzed']}")
        report.append(f"  Recommendations Generated: {test_perf['recommendations_generated']}")
        report.append(f"  Total Potential EV: ${test_perf['total_potential_ev']:.2f}")

        # Bottlenecks
        if bottlenecks:
            report.append("\n⚠️ IDENTIFIED BOTTLENECKS:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                report.append(f"  {i}. {bottleneck}")

        # Recommendations
        if recommendations:
            report.append("\n💡 OPTIMIZATION RECOMMENDATIONS:")

            # Group by priority
            high_priority = [r for r in recommendations if r["priority"] == "High"]
            medium_priority = [r for r in recommendations if r["priority"] == "Medium"]

            if high_priority:
                report.append("\n  HIGH PRIORITY:")
                for rec in high_priority:
                    report.append(f"    • {rec['category']}: {rec['recommendation']}")
                    report.append(f"      Action: {rec['action']}")

            if medium_priority:
                report.append("\n  MEDIUM PRIORITY:")
                for rec in medium_priority:
                    report.append(f"    • {rec['category']}: {rec['recommendation']}")
                    report.append(f"      Action: {rec['action']}")

        # Performance Score
        performance_score = self._calculate_performance_score(betting_perf, test_perf)
        report.append(f"\n📈 OVERALL PERFORMANCE SCORE: {performance_score:.1f}/100")

        # Assessment
        if performance_score >= 80:
            report.append("🎉 EXCELLENT - System is performing optimally")
        elif performance_score >= 60:
            report.append("👍 GOOD - System performing well with room for improvement")
        elif performance_score >= 40:
            report.append("⚠️ MODERATE - System needs significant improvements")
        else:
            report.append("🚨 CRITICAL - System requires immediate attention")

        report.append("\n" + "=" * 80)

        return "\n".join(report)

    def _calculate_performance_score(self, betting_perf: Dict, test_perf: Dict) -> float:
        """Calculate overall performance score out of 100"""
        score = 0

        # ROI scoring (30 points max)
        roi = betting_perf.get("overall_roi", 0)
        if roi >= 10:
            score += 30
        elif roi >= 5:
            score += 20
        elif roi >= 0:
            score += 10
        else:
            score += max(0, roi * 2)  # Negative ROI reduces score

        # Win rate scoring (25 points max)
        win_rate = betting_perf.get("win_rate", 0)
        score += min(25, win_rate * 25)

        # Edge scoring (20 points max)
        avg_edge = test_perf.get("avg_edge_found", 0)
        if avg_edge >= 20:
            score += 20
        elif avg_edge >= 10:
            score += 15
        elif avg_edge >= 5:
            score += 10
        else:
            score += max(0, avg_edge * 2)

        # System efficiency scoring (15 points max)
        recommendations = test_perf.get("recommendations_generated", 0)
        if recommendations > 0:
            score += 15
        elif test_perf.get("games_analyzed", 0) > 0:
            score += 10  # Partial credit for analysis

        # Risk management scoring (10 points max)
        sharpe_ratio = betting_perf.get("sharpe_ratio", 0)
        max_drawdown = betting_perf.get("max_drawdown", 0)

        if sharpe_ratio > 1:
            score += 10
        elif sharpe_ratio > 0.5:
            score += 7
        elif sharpe_ratio > 0:
            score += 5

        if max_drawdown < 0.1:
            score += 5
        elif max_drawdown < 0.2:
            score += 3

        return min(100, score)


def main():
    analyzer = SystemPerformanceAnalyzer()

    print("🔍 Analyzing Football Betting System Performance...")
    print("=" * 80)

    # Generate comprehensive report
    report = analyzer.generate_performance_report()
    print(report)

    # Save analysis to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = Path(f"performance_analysis_{timestamp}.txt")

    with open(report_file, 'w') as f:
        f.write(report)

    print(f"\n📄 Detailed analysis saved to: {report_file}")


if __name__ == "__main__":
    main()