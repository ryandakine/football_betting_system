from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class BettingConfig:
    version: str
    thresholds: Dict[str, Any]
    model_weights: Dict[str, Any]
    risk_limits: Dict[str, Any]
    feature_flags: Dict[str, Any]
    model_checksums: Dict[str, str]

    @classmethod
    def from_yaml(cls, path: Path = Path("config.yaml")) -> "BettingConfig":
        with path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        return cls(**data)

    def is_feature_enabled(self, feature: str) -> bool:
        key = f"use_{feature}"
        return bool(self.feature_flags.get(key, False))
