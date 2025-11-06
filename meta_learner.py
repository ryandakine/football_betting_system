#!/usr/bin/env python3
"""
Meta-learner for ensemble calibration.

Stores per-game features in SQLite, retrains a lightweight logistic regression
to blend base-model probabilities into sharper betting signals.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


@dataclass
class MetaLearnerConfig:
    db_path: Path = Path("data/meta_learner.sqlite")
    min_samples: int = 50
    retrain_interval: int = 25  # retrain after this many new labelled samples
    max_samples: int = 5000
    regularization: float = 1.0


@dataclass
class MetaLearner:
    config: MetaLearnerConfig = field(default_factory=MetaLearnerConfig)
    model: Optional[Pipeline] = None
    _fitted_sample_count: int = 0

    def __post_init__(self) -> None:
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()
        self._load_model_from_disk()

    # ------------------------------------------------------------------ #
    # Schema management
    # ------------------------------------------------------------------ #
    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    game_id TEXT,
                    prediction_time TEXT,
                    features TEXT,
                    base_probability REAL,
                    ensemble_probability REAL,
                    meta_probability REAL,
                    outcome REAL,
                    source TEXT
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_predictions_game
                ON predictions(game_id)
                """
            )

    # ------------------------------------------------------------------ #
    # Persistence helpers
    # ------------------------------------------------------------------ #
    def log_prediction(
        self,
        game_id: str,
        prediction_time: str,
        features: Dict[str, Any],
        base_probability: float,
        ensemble_probability: float,
        meta_probability: Optional[float] = None,
        source: str = "cloud_ensemble",
    ) -> None:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """
                INSERT INTO predictions
                (game_id, prediction_time, features, base_probability,
                 ensemble_probability, meta_probability, outcome, source)
                VALUES (?, ?, ?, ?, ?, ?, NULL, ?)
                """,
                (
                    game_id,
                    prediction_time,
                    json.dumps(features),
                    float(base_probability),
                    float(ensemble_probability),
                    float(meta_probability) if meta_probability is not None else None,
                    source,
                ),
            )

    def update_outcome(self, game_id: str, outcome: float) -> None:
        with sqlite3.connect(self.config.db_path) as conn:
            conn.execute(
                """
                UPDATE predictions
                SET outcome = ?
                WHERE game_id = ?
                """,
                (float(outcome), game_id),
            )

    # ------------------------------------------------------------------ #
    # Training
    # ------------------------------------------------------------------ #
    def train_if_needed(self) -> None:
        labelled = self._load_labelled_samples(limit=self.config.max_samples)
        if len(labelled) < self.config.min_samples:
            return
        if self._fitted_sample_count >= len(labelled):
            return
        if (len(labelled) - self._fitted_sample_count) < self.config.retrain_interval:
            return
        self._train(labelled)
        self._save_model_to_disk()

    def _train(self, labelled: List[Dict[str, Any]]) -> None:
        feature_names = sorted({key for sample in labelled for key in sample["features"]})
        X = np.array([[sample["features"].get(name, 0.0) for name in feature_names] for sample in labelled])
        y = np.array([sample["outcome"] for sample in labelled], dtype=float)

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=self.config.regularization,
                        max_iter=200,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)
        self.model = pipeline
        self.feature_names = feature_names
        self._fitted_sample_count = len(labelled)

    def predict(self, features: Dict[str, Any]) -> Optional[float]:
        if not self.model:
            return None
        feature_vector = np.array([[features.get(name, 0.0) for name in self.feature_names]])
        try:
            proba = self.model.predict_proba(feature_vector)[0, 1]
        except Exception:
            return None
        return float(proba)

    # ------------------------------------------------------------------ #
    # Loading helpers
    # ------------------------------------------------------------------ #
    def _load_labelled_samples(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        query = """
            SELECT features, outcome
            FROM predictions
            WHERE outcome IS NOT NULL
            ORDER BY prediction_time DESC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        with sqlite3.connect(self.config.db_path) as conn:
            rows = conn.execute(query).fetchall()
        labelled = []
        for row in rows:
            try:
                feats = json.loads(row[0])
            except Exception:
                continue
            labelled.append({"features": feats, "outcome": float(row[1])})
        return labelled

    def _save_model_to_disk(self) -> None:
        if not self.model:
            return
        payload = {
            "model": self.model,
            "feature_names": getattr(self, "feature_names", []),
            "sample_count": self._fitted_sample_count,
        }
        path = self.config.db_path.with_suffix(".pkl")
        import joblib

        joblib.dump(payload, path)

    def _load_model_from_disk(self) -> None:
        path = self.config.db_path.with_suffix(".pkl")
        if not path.exists():
            return
        try:
            import joblib

            payload = joblib.load(path)
            self.model = payload.get("model")
            self.feature_names = payload.get("feature_names", [])
            self._fitted_sample_count = payload.get("sample_count", 0)
        except Exception:
            self.model = None
            self._fitted_sample_count = 0
            self.feature_names = []

