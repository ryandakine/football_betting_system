# advanced_components.py - Complete minimal version
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


class SystemMonitor:
    """Simple system monitoring"""

    def __init__(self):
        self.start_time = time.time()
        self.api_calls = 0
        self.errors = 0
        self.cache_hits = 0

    def record_api_call(self, provider, start_time, end_time, success, error_msg=None):
        self.api_calls += 1
        if not success:
            self.errors += 1

    def record_cache_hit(self):
        self.cache_hits += 1

    def record_cache_miss(self):
        pass

    def get_performance_report(self):
        return {
            "api_calls": self.api_calls,
            "errors": self.errors,
            "cache_hits": self.cache_hits,
            "uptime": time.time() - self.start_time,
        }


class ConfigValidator:
    """Simple configuration validation"""

    @staticmethod
    def validate_system_config(config: dict) -> tuple[bool, list[str]]:
        errors = []
        if config.get("bankroll", 0) <= 0:
            errors.append("Bankroll must be positive")
        return len(errors) == 0, errors


class IntelligentCache:
    """Simple file-based cache"""

    def __init__(self, max_size=1000, default_ttl=3600):
        self.cache_dir = Path("cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = max_size
        self.default_ttl = default_ttl

    def get(self, key: str):
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            if time.time() - cache_file.stat().st_mtime < self.default_ttl:
                try:
                    with open(cache_file) as f:
                        return json.load(f)
                except:
                    pass
        return None

    def set(self, key: str, value: Any):
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(value, f, default=str)
        except:
            pass

    def get_stats(self):
        return {"size": len(list(self.cache_dir.glob("*.json")))}


class RiskManager:
    """Simple risk management for Kelly Criterion"""

    def __init__(self, max_kelly_fraction=0.25):
        self.max_kelly_fraction = max_kelly_fraction
        self.min_edge = 0.02

    def get_kelly_stake(self, win_prob: float, odds: float, bankroll: float) -> float:
        """Calculate Kelly Criterion stake"""
        if win_prob <= 0 or odds <= 1:
            return 0.0

        edge = (win_prob * odds) - 1
        if edge <= self.min_edge:
            return 0.0

        # Kelly formula
        b = odds - 1
        p = win_prob
        q = 1 - p

        kelly_fraction = (b * p - q) / b
        kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        kelly_fraction = max(kelly_fraction, 0)

        return bankroll * kelly_fraction

    def assess_portfolio_risk(self, bets: list[dict]) -> dict:
        """Simple portfolio risk assessment"""
        if not bets:
            return {"total_risk": 0}

        total_stakes = [bet.get("stake", 0) for bet in bets]
        return {
            "total_exposure": sum(total_stakes),
            "position_count": len(bets),
            "max_stake": max(total_stakes) if total_stakes else 0,
        }
