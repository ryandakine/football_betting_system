"""
MLflow Integration for MLB Betting System
Provides model tracking, experiment management, and model registry
"""

import os
from datetime import datetime
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import mlflow.sklearn
import numpy as np
import pandas as pd


class MLflowMLBIntegration:
    """MLflow integration for MLB betting system model management"""

    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.experiment_name = "mlb-betting-predictions"
        mlflow.set_experiment(self.experiment_name)

    def log_training_run(
        self,
        model,
        model_name: str,
        metrics: dict[str, float],
        parameters: dict[str, Any],
        features: pd.DataFrame,
        target: pd.Series,
        model_type: str = "sklearn",
    ) -> str:
        """Log a complete training run"""

        with mlflow.start_run() as run:
            # Log parameters
            mlflow.log_params(parameters)

            # Log metrics
            mlflow.log_metrics(metrics)

            # Log feature importance if available
            if hasattr(model, "feature_importances_"):
                feature_importance = pd.DataFrame(
                    {
                        "feature": features.columns,
                        "importance": model.feature_importances_,
                    }
                )
                mlflow.log_artifact(
                    feature_importance.to_csv(), "feature_importance.csv"
                )

            # Log model
            if model_type == "sklearn":
                mlflow.sklearn.log_model(model, model_name)
            elif model_type == "pytorch":
                mlflow.pytorch.log_model(model, model_name)

            # Log sample predictions
            sample_predictions = pd.DataFrame(
                {
                    "actual": target.head(100),
                    "predicted": model.predict(features.head(100)),
                }
            )
            mlflow.log_artifact(sample_predictions.to_csv(), "sample_predictions.csv")

            # Log model performance visualization
            self._log_performance_plots(target, model.predict(features))

            return run.info.run_id

    def log_prediction_batch(
        self,
        predictions: np.ndarray,
        game_ids: list,
        confidence_scores: np.ndarray,
        metadata: dict[str, Any],
    ) -> str:
        """Log a batch of predictions for tracking"""

        with mlflow.start_run(nested=True) as run:
            # Log prediction metadata
            mlflow.log_params(metadata)

            # Log prediction statistics
            mlflow.log_metrics(
                {
                    "mean_confidence": float(np.mean(confidence_scores)),
                    "std_confidence": float(np.std(confidence_scores)),
                    "num_predictions": len(predictions),
                    "high_confidence_predictions": int(np.sum(confidence_scores > 0.8)),
                }
            )

            # Log predictions as artifact
            predictions_df = pd.DataFrame(
                {
                    "game_id": game_ids,
                    "prediction": predictions,
                    "confidence": confidence_scores,
                    "timestamp": datetime.now().isoformat(),
                }
            )
            mlflow.log_artifact(predictions_df.to_csv(), "predictions.csv")

            return run.info.run_id

    def log_betting_opportunity(
        self, opportunity: dict[str, Any], model_version: str
    ) -> str:
        """Log a betting opportunity analysis"""

        with mlflow.start_run(nested=True) as run:
            # Log opportunity details
            mlflow.log_params(
                {
                    "model_version": model_version,
                    "game_id": opportunity.get("game_id"),
                    "confidence_score": opportunity.get("confidence_score"),
                    "recommendation": opportunity.get("recommendation"),
                }
            )

            # Log opportunity as artifact
            opportunity_df = pd.DataFrame([opportunity])
            mlflow.log_artifact(opportunity_df.to_csv(), "betting_opportunity.csv")

            return run.info.run_id

    def _log_performance_plots(self, y_true: pd.Series, y_pred: np.ndarray):
        """Log performance visualization plots"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # Create performance plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            # Actual vs Predicted
            axes[0, 0].scatter(y_true, y_pred, alpha=0.5)
            axes[0, 0].plot(
                [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
            )
            axes[0, 0].set_xlabel("Actual")
            axes[0, 0].set_ylabel("Predicted")
            axes[0, 0].set_title("Actual vs Predicted")

            # Residuals
            residuals = y_true - y_pred
            axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
            axes[0, 1].axhline(y=0, color="r", linestyle="--")
            axes[0, 1].set_xlabel("Predicted")
            axes[0, 1].set_ylabel("Residuals")
            axes[0, 1].set_title("Residuals Plot")

            # Residuals Distribution
            axes[1, 0].hist(residuals, bins=30, alpha=0.7)
            axes[1, 0].set_xlabel("Residuals")
            axes[1, 0].set_ylabel("Frequency")
            axes[1, 0].set_title("Residuals Distribution")

            # Prediction Distribution
            axes[1, 1].hist(y_pred, bins=30, alpha=0.7)
            axes[1, 1].set_xlabel("Predictions")
            axes[1, 1].set_ylabel("Frequency")
            axes[1, 1].set_title("Prediction Distribution")

            plt.tight_layout()
            mlflow.log_figure(fig, "performance_plots.png")
            plt.close()

        except ImportError:
            print("Matplotlib/Seaborn not available for plotting")

    def get_best_model(
        self, metric: str = "accuracy", ascending: bool = False
    ) -> str | None:
        """Get the best model based on a metric"""
        try:
            # Get all runs for the experiment
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                return None

            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            )

            if len(runs) > 0:
                return runs.iloc[0]["run_id"]
            return None

        except Exception as e:
            print(f"Error getting best model: {e}")
            return None

    def load_model(self, run_id: str, model_name: str = "model"):
        """Load a model from MLflow"""
        try:
            model_uri = f"runs:/{run_id}/{model_name}"
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def register_model(self, run_id: str, model_name: str, model_path: str):
        """Register a model in the MLflow model registry"""
        try:
            model_uri = f"runs:/{run_id}/{model_path}"
            mlflow.register_model(model_uri, model_name)
            print(f"Model {model_name} registered successfully")
        except Exception as e:
            print(f"Error registering model: {e}")


# Usage example
if __name__ == "__main__":
    # Initialize MLflow integration
    mlflow_mlb = MLflowMLBIntegration()

    # Example: Log a training run
    # mlflow_mlb.log_training_run(
    #     model=your_model,
    #     model_name="mlb-betting-model",
    #     metrics={"accuracy": 0.85, "precision": 0.82},
    #     parameters={"n_estimators": 100, "max_depth": 10},
    #     features=your_features,
    #     target=your_target
    # )
