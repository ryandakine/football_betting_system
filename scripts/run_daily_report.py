#!/usr/bin/env python3
"""
Cron-friendly runner to generate the consolidated daily report.

Usage (cron):
  15 23 * * * /usr/bin/env python3 \
    /home/ryan/code/football_betting_system/scripts/run_daily_report.py >> \
    /home/ryan/code/football_betting_system/logs/daily_report.cron.log 2>&1
"""

import os
from datetime import datetime

from reporting.daily_reporter import DailyReporter


def main():
    date_override = os.getenv("REPORT_DATE")
    date_str = date_override or datetime.now().strftime("%Y-%m-%d")
    reporter = DailyReporter()
    reporter.generate(date_str)


if __name__ == "__main__":
    main()

