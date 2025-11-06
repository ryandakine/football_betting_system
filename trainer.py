import argparse
import os
import sqlite3

import joblib
import numpy as np
import pandas as pd
import requests

try:
    import lightgbm as lgb

    LGBM_AVAILABLE = True
except ImportError:
    from sklearn.linear_model import LogisticRegression

    LGBM_AVAILABLE = False


def fetch_data(db_path, days):
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT r.game_id, r.home_final, r.away_final, b.fd_price_home, b.fd_price_away, b.edge_home, b.edge_away, b.sentiment_home, b.sentiment_away
        FROM results r
        JOIN bets b ON r.game_id = b.game_id
        WHERE r.game_date >= date('now', '-{days} day')
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df


def train_model(df):
    # Features must be pre-bet only (no label leakage)
    X = df[
        [
            "fd_price_home",
            "fd_price_away",
            "edge_home",
            "edge_away",
            "sentiment_home",
            "sentiment_away",
        ]
    ].values
    y = (df["home_final"] > df["away_final"]).astype(int)
    if LGBM_AVAILABLE:
        model = lgb.LGBMClassifier()
        model.fit(X, y)
        print("Trained LightGBM model.")
    else:
        model = LogisticRegression()
        model.fit(X, y)
        print("Trained LogisticRegression model.")
    return model, X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--db", type=str, default="db/bets.sqlite")
    parser.add_argument("--out", type=str, default="models/current.pkl")
    args = parser.parse_args()

    df = fetch_data(args.db, args.days)
    if df.empty:
        print("No data to train on.")
        return

    model, X, y = train_model(df)
    joblib.dump(model, args.out)
    print(f"Model saved to {args.out}")

    # ROC-AUC
    from sklearn.metrics import roc_auc_score

    y_pred = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    print(f"ROC-AUC: {auc:.3f}")

    # Hot-swap model
    with open(args.out, "rb") as f:
        resp = requests.put("http://predictor:8799/model", files={"file": f})
        print("Model hot-swap response:", resp.text)


if __name__ == "__main__":
    main()
