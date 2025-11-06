import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load MLB historical data from a CSV file."""
    return pd.read_csv(csv_path)


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_features = df.select_dtypes(include=["object", "category"]).columns

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def train_random_forest(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train a Random Forest model with preprocessing."""
    preprocessor = build_preprocessor(X)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X, y)
    return clf


def train_gradient_boosting(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Train a Gradient Boosting model with preprocessing."""
    preprocessor = build_preprocessor(X)
    model = GradientBoostingClassifier(random_state=42)
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X, y)
    return clf


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float:
    """Evaluate model accuracy on a test set."""
    preds = model.predict(X_test)
    return accuracy_score(y_test, preds)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train MLB prediction models")
    parser.add_argument("csv_path", help="Path to historical MLB data in CSV format")
    parser.add_argument(
        "--model", choices=["rf", "gb"], default="rf", help="Model type"
    )
    args = parser.parse_args()

    df = load_dataset(args.csv_path)
    target = df["target"]  # replace with correct target column
    features = df.drop(columns=["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    if args.model == "rf":
        model = train_random_forest(X_train, y_train)
    else:
        model = train_gradient_boosting(X_train, y_train)

    acc = evaluate_model(model, X_test, y_test)
    print(f"Test Accuracy: {acc:.3f}")
