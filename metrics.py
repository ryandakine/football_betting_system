# mlb_betting_system/metrics.py
"""
Metrics collection and logging for MLB betting system.
"""

import logging

logger == logging.getLogger(__name__)


class BettingMetrics:
    def __init__(self):
        self.metrics = {}

    def record_metric(self, name: str, value: float):
        self.metrics[name] = value
        logger.info(f"[Metric] {name}: {value}")

    def record_error(self, name: str, count: int):
        self.metrics[name] = count
        logger.warning(f"[Error] {name}: {count}")

    def record_execution_time(self, name: str, seconds: float):
        self.metrics[name] = seconds
        logger.info(f"[ExecutionTime] {name}: {seconds:.2f}s")


# Create a shared instance
betting_metrics = BettingMetrics
