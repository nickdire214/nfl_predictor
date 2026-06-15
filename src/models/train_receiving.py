"""Receiving yards model training module.

Trains a Ridge regression model to predict receiving yards from
data/features/receiving_matrix.parquet, mirroring src/models/train.py's
structure for the QB passing-yards model.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.receiving import RECEIVING_ROLLING_COLS
from src.ingestion._common import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

ROLLING_SUFFIXES = ("_l3", "_l8", "_std")
ROLLING_COLS = [f"{stat}{suffix}" for stat in RECEIVING_ROLLING_COLS for suffix in ROLLING_SUFFIXES]

FEATURE_COLS = ROLLING_COLS + [
    "is_home", "team_rest", "opponent_rest", "div_game", "team_implied_total",
    "spread_line", "total_line", "games_into_season", "temp", "wind",
    "is_dome", "is_playoff", "pos_WR", "pos_TE",
]
TARGET = "receiving_yards"


def preprocess_receiving_matrix(df):
    df = df.dropna(subset=["targets_l3"]).copy()

    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["temp"] = df["temp"].fillna(70)
    df["wind"] = df["wind"].fillna(0)

    for stat in RECEIVING_ROLLING_COLS:
        df[f"{stat}_std"] = df[f"{stat}_std"].fillna(df[f"{stat}_l8"])

    df["is_playoff"] = (df["game_type"] != "REG").astype(int)

    df["pos_WR"] = (df["position"] == "WR").astype(int)
    df["pos_TE"] = (df["position"] == "TE").astype(int)

    return df


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _rmse(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))


def _bias(pred, actual):
    return np.mean(pred - actual)


def main():
    df = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
    df = preprocess_receiving_matrix(df)

    train = df[df["season"] <= 2024]
    test = df[df["season"] == 2025]
    print(f"Train rows: {len(train)} (seasons 2021-2024)")
    print(f"Test rows: {len(test)} (season 2025)")

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    ridge_pred = pipeline.predict(X_test)
    naive_l8_pred = test["receiving_yards_l8"]
    naive_season_pred = test["receiving_yards_std"]

    print("\nEvaluation on 2025 test set:")
    print(f"{'predictor':<15} {'MAE':>8} {'RMSE':>8} {'bias':>8}")
    for name, pred in [
        ("naive-L8", naive_l8_pred),
        ("naive-season", naive_season_pred),
        ("ridge", ridge_pred),
    ]:
        print(f"{name:<15} {_mae(pred, y_test):>8.2f} {_rmse(pred, y_test):>8.2f} {_bias(pred, y_test):>8.2f}")

    coefs = pipeline.named_steps["ridge"].coef_
    coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": coefs})
    coef_df["abs_coef"] = coef_df["coef"].abs()
    coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15)

    print("\nTop 15 Ridge coefficients by absolute value:")
    print(coef_df[["feature", "coef"]].to_string(index=False))

    print("\nFinal production fit (all seasons):")
    X_all, y_all = df[FEATURE_COLS], df[TARGET]
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    final_pipeline.fit(X_all, y_all)
    print(f"Train rows: {len(df)} (seasons {df['season'].min()}-{df['season'].max()})")

    final_coefs = final_pipeline.named_steps["ridge"].coef_
    final_coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": final_coefs})
    final_coef_df["abs_coef"] = final_coef_df["coef"].abs()
    final_coef_df = final_coef_df.sort_values("abs_coef", ascending=False).head(15)

    print("\nTop 15 Ridge coefficients by absolute value (final production fit):")
    print(final_coef_df[["feature", "coef"]].to_string(index=False))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "receiving_yards_ridge.pkl"
    joblib.dump(final_pipeline, model_path)
    print(f"\nSaved pipeline to: {model_path}")


if __name__ == "__main__":
    main()
