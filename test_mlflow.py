"""
Test script for MLflow integration
"""

from datetime import datetime

import mlflow
import numpy as np
import pandas as pd


def test_mlflow_integration():
    """Test MLflow integration with sample data"""

    print("🧪 Testing MLflow Integration...")

    # Set MLflow tracking URI
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlb-betting-test")

    # Create sample data
    np.random.seed(42)
    n_samples = 1000

    # Simulate betting data
    data = {
        "home_team_win_rate": np.random.uniform(0.3, 0.7, n_samples),
        "away_team_win_rate": np.random.uniform(0.3, 0.7, n_samples),
        "home_team_odds": np.random.uniform(1.5, 3.0, n_samples),
        "away_team_odds": np.random.uniform(1.5, 3.0, n_samples),
        "weather_factor": np.random.uniform(0.8, 1.2, n_samples),
        "injury_factor": np.random.uniform(0.9, 1.1, n_samples),
        "home_win": np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
    }

    df = pd.DataFrame(data)

    # Simulate model training
    with mlflow.start_run(run_name="test-mlb-model") as run:
        print(f"📊 MLflow Run ID: {run.info.run_id}")

        # Log parameters
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("test_date", datetime.now().isoformat())
        mlflow.log_param("model_type", "random_forest")
        mlflow.log_param("random_state", 42)

        # Simulate model training
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, precision_score, recall_score
        from sklearn.model_selection import train_test_split

        # Prepare features and target
        X = df.drop("home_win", axis=1)
        y = df["home_win"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", 2 * (precision * recall) / (precision + recall))

        # Log feature importance
        feature_importance = pd.DataFrame(
            {"feature": X.columns, "importance": model.feature_importances_}
        ).sort_values("importance", ascending=False)

        mlflow.log_artifact(feature_importance.to_csv(), "feature_importance.csv")

        # Log sample predictions
        sample_predictions = pd.DataFrame(
            {
                "actual": y_test.head(20),
                "predicted": y_pred[:20],
                "confidence": model.predict_proba(X_test)[:20, 1],
            }
        )

        mlflow.log_artifact(sample_predictions.to_csv(), "sample_predictions.csv")

        # Log model
        mlflow.sklearn.log_model(model, "mlb-betting-model")

        print(f"✅ Model logged successfully!")
        print(f"📈 Accuracy: {accuracy:.3f}")
        print(f"🎯 Precision: {precision:.3f}")
        print(f"📊 Recall: {recall:.3f}")

        return run.info.run_id


def test_betting_opportunity_logging():
    """Test logging betting opportunities"""

    print("\n🎲 Testing Betting Opportunity Logging...")

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("mlb-betting-opportunities")

    # Simulate betting opportunities
    opportunities = [
        {
            "game_id": "MLB_2025_001",
            "home_team": "Yankees",
            "away_team": "Red Sox",
            "home_odds": 1.85,
            "away_odds": 2.15,
            "predicted_winner": "home",
            "confidence": 0.78,
            "recommendation": "STRONG_BUY",
            "stake_percentage": 0.05,
        },
        {
            "game_id": "MLB_2025_002",
            "home_team": "Dodgers",
            "away_team": "Giants",
            "home_odds": 1.65,
            "away_odds": 2.45,
            "predicted_winner": "home",
            "confidence": 0.82,
            "recommendation": "STRONG_BUY",
            "stake_percentage": 0.08,
        },
    ]

    for i, opp in enumerate(opportunities):
        with mlflow.start_run(run_name=f"opportunity-{opp['game_id']}") as run:
            # Log opportunity details
            mlflow.log_params(
                {
                    "game_id": opp["game_id"],
                    "home_team": opp["home_team"],
                    "away_team": opp["away_team"],
                    "predicted_winner": opp["predicted_winner"],
                    "recommendation": opp["recommendation"],
                }
            )

            # Log metrics
            mlflow.log_metric("confidence", opp["confidence"])
            mlflow.log_metric("home_odds", opp["home_odds"])
            mlflow.log_metric("away_odds", opp["away_odds"])
            mlflow.log_metric("stake_percentage", opp["stake_percentage"])

            # Log opportunity as artifact
            opp_df = pd.DataFrame([opp])
            mlflow.log_artifact(opp_df.to_csv(), f"opportunity_{opp['game_id']}.csv")

            print(
                f"✅ Logged opportunity {i+1}: {opp['home_team']} vs {opp['away_team']}"
            )


def main():
    """Main test function"""

    print("🚀 MLflow Integration Test")
    print("=" * 50)

    try:
        # Test basic MLflow functionality
        run_id = test_mlflow_integration()

        # Test betting opportunity logging
        test_betting_opportunity_logging()

        print("\n🎉 All tests completed successfully!")
        print(f"📊 Check MLflow UI at: http://localhost:5000")
        print(f"🔍 Run ID: {run_id}")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(
            "💡 Make sure MLflow server is running: mlflow server --host 0.0.0.0 --port 5000"
        )


if __name__ == "__main__":
    main()
