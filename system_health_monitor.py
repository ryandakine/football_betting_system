#!/usr/bin/env python3
"""
MLB Betting System Health Monitor
=================================
Comprehensive health monitoring for all system components.
"""

import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("system_health.log"),
        logging.StreamHandler(sys.stdout),
    ],
)


@dataclass
class HealthCheck:
    """Health check result data class."""

    component: str
    status: str  # 'healthy', 'warning', 'critical', 'unknown'
    message: str
    response_time: float | None = None
    details: dict | None = None
    timestamp: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class SystemHealthMonitor:
    """Comprehensive system health monitoring."""

    def __init__(self):
        self.checks: list[HealthCheck] = []
        self.api_keys = {
            "youtube": os.getenv("YOUTUBE_API_KEY"),
            "gemini": os.getenv("GEMINI_API_KEY"),
            "openai": os.getenv("OPENAI_API_KEY"),
            "claude": os.getenv("CLAUDE_API_KEY"),
            "perplexity": os.getenv("PERPLEXITY_API_KEY"),
            "odds": os.getenv("THE_ODDS_API_KEY"),
            "slack": os.getenv("SLACK_WEBHOOK_URL"),
            "supabase_url": os.getenv("SUPABASE_URL"),
            "supabase_key": os.getenv("SUPABASE_ANON_KEY"),
        }

    def check_api_key(self, key_name: str, key_value: str) -> HealthCheck:
        """Check if API key is configured."""
        if not key_value:
            return HealthCheck(
                component=f"{key_name}_api_key",
                status="critical",
                message=f"{key_name.upper()} API key not configured",
            )

        # Check if key looks valid (basic format check)
        if key_name == "youtube" and not key_value.startswith("AIza"):
            return HealthCheck(
                component=f"{key_name}_api_key",
                status="warning",
                message=f"{key_name.upper()} API key format appears invalid",
            )
        elif key_name == "openai" and not key_value.startswith("sk-"):
            return HealthCheck(
                component=f"{key_name}_api_key",
                status="warning",
                message=f"{key_name.upper()} API key format appears invalid",
            )

        return HealthCheck(
            component=f"{key_name}_api_key",
            status="healthy",
            message=f"{key_name.upper()} API key configured",
        )

    def check_youtube_api(self) -> HealthCheck:
        """Test YouTube API connectivity."""
        start_time = time.time()

        try:
            url = "https://www.googleapis.com/youtube/v3/search"
            params = {
                "part": "snippet",
                "q": "MLB",
                "type": "video",
                "maxResults": 1,
                "key": self.api_keys["youtube"],
            }

            response = requests.get(url, params=params, timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                data = response.json()
                if "items" in data:
                    return HealthCheck(
                        component="youtube_api",
                        status="healthy",
                        message="YouTube API responding correctly",
                        response_time=response_time,
                        details={"videos_found": len(data.get("items", []))},
                    )
                else:
                    return HealthCheck(
                        component="youtube_api",
                        status="warning",
                        message="YouTube API responded but no items found",
                        response_time=response_time,
                    )
            elif response.status_code == 403:
                return HealthCheck(
                    component="youtube_api",
                    status="critical",
                    message="YouTube API access denied - check API key and permissions",
                    response_time=response_time,
                )
            elif response.status_code == 429:
                return HealthCheck(
                    component="youtube_api",
                    status="warning",
                    message="YouTube API rate limit exceeded",
                    response_time=response_time,
                )
            else:
                return HealthCheck(
                    component="youtube_api",
                    status="critical",
                    message=f"YouTube API error: {response.status_code}",
                    response_time=response_time,
                )

        except requests.exceptions.Timeout:
            return HealthCheck(
                component="youtube_api",
                status="critical",
                message="YouTube API timeout",
                response_time=time.time() - start_time,
            )
        except Exception as e:
            return HealthCheck(
                component="youtube_api",
                status="critical",
                message=f"YouTube API error: {str(e)}",
                response_time=time.time() - start_time,
            )

    def check_gemini_api(self) -> HealthCheck:
        """Test Gemini API connectivity."""
        start_time = time.time()

        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
            params = {"key": self.api_keys["gemini"]}
            headers = {"Content-Type": "application/json"}
            data = {"contents": [{"parts": [{"text": "Hello, this is a test."}]}]}

            response = requests.post(
                url, params=params, headers=headers, json=data, timeout=10
            )
            response_time = time.time() - start_time

            if response.status_code == 200:
                return HealthCheck(
                    component="gemini_api",
                    status="healthy",
                    message="Gemini API responding correctly",
                    response_time=response_time,
                )
            elif response.status_code == 403:
                return HealthCheck(
                    component="gemini_api",
                    status="critical",
                    message="Gemini API access denied - check API key",
                    response_time=response_time,
                )
            else:
                return HealthCheck(
                    component="gemini_api",
                    status="critical",
                    message=f"Gemini API error: {response.status_code}",
                    response_time=response_time,
                )

        except Exception as e:
            return HealthCheck(
                component="gemini_api",
                status="critical",
                message=f"Gemini API error: {str(e)}",
                response_time=time.time() - start_time,
            )

    def check_supabase_connection(self) -> HealthCheck:
        """Test Supabase connectivity."""
        start_time = time.time()

        try:
            url = f"{self.api_keys['supabase_url']}/rest/v1/sentiment_data"
            headers = {
                "apikey": self.api_keys["supabase_key"],
                "Authorization": f"Bearer {self.api_keys['supabase_key']}",
            }

            response = requests.get(url, headers=headers, timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                return HealthCheck(
                    component="supabase_connection",
                    status="healthy",
                    message="Supabase connection successful",
                    response_time=response_time,
                )
            elif response.status_code == 401:
                return HealthCheck(
                    component="supabase_connection",
                    status="critical",
                    message="Supabase authentication failed - check API key",
                    response_time=response_time,
                )
            else:
                return HealthCheck(
                    component="supabase_connection",
                    status="critical",
                    message=f"Supabase error: {response.status_code}",
                    response_time=response_time,
                )

        except Exception as e:
            return HealthCheck(
                component="supabase_connection",
                status="critical",
                message=f"Supabase connection error: {str(e)}",
                response_time=time.time() - start_time,
            )

    def check_slack_webhook(self) -> HealthCheck:
        """Test Slack webhook."""
        start_time = time.time()

        try:
            data = {"text": "🔍 System health check - this is a test message"}

            response = requests.post(self.api_keys["slack"], json=data, timeout=10)
            response_time = time.time() - start_time

            if response.status_code == 200:
                return HealthCheck(
                    component="slack_webhook",
                    status="healthy",
                    message="Slack webhook working correctly",
                    response_time=response_time,
                )
            elif response.status_code == 404:
                return HealthCheck(
                    component="slack_webhook",
                    status="critical",
                    message="Slack webhook URL invalid",
                    response_time=response_time,
                )
            else:
                return HealthCheck(
                    component="slack_webhook",
                    status="warning",
                    message=f"Slack webhook error: {response.status_code}",
                    response_time=response_time,
                )

        except Exception as e:
            return HealthCheck(
                component="slack_webhook",
                status="critical",
                message=f"Slack webhook error: {str(e)}",
                response_time=time.time() - start_time,
            )

    def check_workflow_files(self) -> HealthCheck:
        """Check if workflow files exist and are valid JSON."""
        workflow_files = [
            "mlb_youtube_workflow.json",
            "n8n-workflows/enhanced-mlb-opportunity-detector.json",
            "n8n-workflows/real-time-odds-monitor.json",
            "n8n-workflows/sentiment-analysis-agent.json",
        ]

        missing_files = []
        invalid_files = []

        for file_path in workflow_files:
            if not os.path.exists(file_path):
                missing_files.append(file_path)
            else:
                try:
                    with open(file_path) as f:
                        json.load(f)
                except json.JSONDecodeError:
                    invalid_files.append(file_path)

        if missing_files and invalid_files:
            return HealthCheck(
                component="workflow_files",
                status="critical",
                message=f"Missing files: {missing_files}, Invalid JSON: {invalid_files}",
            )
        elif missing_files:
            return HealthCheck(
                component="workflow_files",
                status="critical",
                message=f"Missing workflow files: {missing_files}",
            )
        elif invalid_files:
            return HealthCheck(
                component="workflow_files",
                status="critical",
                message=f"Invalid JSON in files: {invalid_files}",
            )
        else:
            return HealthCheck(
                component="workflow_files",
                status="healthy",
                message=f"All {len(workflow_files)} workflow files present and valid",
            )

    def check_database_schema(self) -> HealthCheck:
        """Check if database schema file exists."""
        schema_file = "mlb_tables.sql"

        if not os.path.exists(schema_file):
            return HealthCheck(
                component="database_schema",
                status="critical",
                message="Database schema file missing",
            )

        try:
            with open(schema_file) as f:
                content = f.read()
                if "CREATE TABLE" in content and "sentiment_data" in content:
                    return HealthCheck(
                        component="database_schema",
                        status="healthy",
                        message="Database schema file present and contains required tables",
                    )
                else:
                    return HealthCheck(
                        component="database_schema",
                        status="warning",
                        message="Database schema file present but may be incomplete",
                    )
        except Exception as e:
            return HealthCheck(
                component="database_schema",
                status="critical",
                message=f"Error reading database schema: {str(e)}",
            )

    def check_system_resources(self) -> HealthCheck:
        """Check system resources (CPU, memory, disk)."""
        try:
            import psutil

            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            details = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
            }

            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90 or disk.percent > 90:
                status = "critical"
                message = "System resources critically high"
            elif cpu_percent > 70 or memory.percent > 70 or disk.percent > 70:
                status = "warning"
                message = "System resources elevated"
            else:
                status = "healthy"
                message = "System resources normal"

            return HealthCheck(
                component="system_resources",
                status=status,
                message=message,
                details=details,
            )

        except ImportError:
            return HealthCheck(
                component="system_resources",
                status="unknown",
                message="psutil not available - cannot check system resources",
            )
        except Exception as e:
            return HealthCheck(
                component="system_resources",
                status="critical",
                message=f"Error checking system resources: {str(e)}",
            )

    def run_all_checks(self) -> list[HealthCheck]:
        """Run all health checks."""
        logging.info("Starting comprehensive system health check...")

        # API key checks
        for key_name, key_value in self.api_keys.items():
            if key_name not in ["supabase_url", "supabase_key"]:
                self.checks.append(self.check_api_key(key_name, key_value))

        # API connectivity checks
        self.checks.append(self.check_youtube_api())
        self.checks.append(self.check_gemini_api())
        self.checks.append(self.check_supabase_connection())
        self.checks.append(self.check_slack_webhook())

        # File system checks
        self.checks.append(self.check_workflow_files())
        self.checks.append(self.check_database_schema())

        # System resource checks
        self.checks.append(self.check_system_resources())

        return self.checks

    def generate_report(self) -> dict[str, Any]:
        """Generate comprehensive health report."""
        if not self.checks:
            self.run_all_checks()

        total_checks = len(self.checks)
        healthy_checks = len([c for c in self.checks if c.status == "healthy"])
        warning_checks = len([c for c in self.checks if c.status == "warning"])
        critical_checks = len([c for c in self.checks if c.status == "critical"])

        overall_status = "healthy"
        if critical_checks > 0:
            overall_status = "critical"
        elif warning_checks > 0:
            overall_status = "warning"

        report = {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status,
            "summary": {
                "total_checks": total_checks,
                "healthy": healthy_checks,
                "warnings": warning_checks,
                "critical": critical_checks,
            },
            "checks": [vars(check) for check in self.checks],
            "recommendations": self.generate_recommendations(),
        }

        return report

    def generate_recommendations(self) -> list[str]:
        """Generate recommendations based on health check results."""
        recommendations = []

        critical_checks = [c for c in self.checks if c.status == "critical"]
        warning_checks = [c for c in self.checks if c.status == "warning"]

        if any("api_key" in c.component for c in critical_checks):
            recommendations.append(
                "Configure missing API keys in environment variables"
            )

        if any("youtube_api" in c.component for c in critical_checks):
            recommendations.append(
                "Check YouTube Data API v3 is enabled in Google Cloud Console"
            )

        if any("supabase" in c.component for c in critical_checks):
            recommendations.append(
                "Verify Supabase credentials and database connection"
            )

        if any("workflow_files" in c.component for c in critical_checks):
            recommendations.append("Restore missing workflow files from backup")

        if any("system_resources" in c.component for c in critical_checks):
            recommendations.append(
                "Consider scaling up system resources or optimizing workflows"
            )

        if not recommendations:
            recommendations.append("System is healthy - continue monitoring")

        return recommendations

    def save_report(self, filename: str = None) -> str:
        """Save health report to file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"health_report_{timestamp}.json"

        report = self.generate_report()

        with open(filename, "w") as f:
            json.dump(report, f, indent=2)

        logging.info(f"Health report saved to {filename}")
        return filename

    def print_summary(self):
        """Print health check summary to console."""
        report = self.generate_report()

        print("\n" + "=" * 60)
        print("🏥 MLB BETTING SYSTEM HEALTH CHECK")
        print("=" * 60)
        print(f"📅 Timestamp: {report['timestamp']}")
        print(f"📊 Overall Status: {report['overall_status'].upper()}")
        print(
            f"📈 Summary: {report['summary']['healthy']} healthy, "
            f"{report['summary']['warnings']} warnings, "
            f"{report['summary']['critical']} critical"
        )
        print("-" * 60)

        for check in self.checks:
            status_icon = {
                "healthy": "✅",
                "warning": "⚠️",
                "critical": "❌",
                "unknown": "❓",
            }.get(check.status, "❓")

            print(f"{status_icon} {check.component}: {check.message}")
            if check.response_time:
                print(f"   ⏱️  Response time: {check.response_time:.2f}s")
            if check.details:
                print(f"   📋 Details: {check.details}")

        print("-" * 60)
        print("💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"   • {rec}")
        print("=" * 60)


def main():
    """Main function to run health check."""
    monitor = SystemHealthMonitor()
    monitor.run_all_checks()
    monitor.print_summary()

    # Save report
    report_file = monitor.save_report()
    print(f"\n📄 Detailed report saved to: {report_file}")

    # Exit with appropriate code
    critical_checks = len([c for c in monitor.checks if c.status == "critical"])
    if critical_checks > 0:
        sys.exit(1)  # Exit with error code if critical issues found
    else:
        sys.exit(0)  # Exit successfully


if __name__ == "__main__":
    main()
