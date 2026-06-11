"""Model training module.

Trains a Ridge regression model to predict QB passing yards from
data/features/qb_matrix.parquet, evaluates it against naive baselines
on the 2025 holdout season, and persists the fitted pipeline.
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.ingestion._common import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

STAT_COLS = [
    "passing_yards", "attempts", "completions", "passing_tds",
    "passing_interceptions", "passing_epa", "passing_cpoe", "sacks_suffered",
]
ROLLING_COLS = [f"{stat}{suffix}" for stat in STAT_COLS for suffix in ("_l3", "_l8", "_std")]

FEATURE_COLS = ROLLING_COLS + [
    "is_home", "team_rest", "opponent_rest", "div_game", "team_implied_total",
    "spread_line", "total_line", "games_into_season", "temp", "wind",
    "is_dome", "is_playoff",
]
TARGET = "passing_yards"


def preprocess_qb_matrix(df):
    df = df.dropna(subset=["passing_yards_l3"]).copy()

    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["temp"] = df["temp"].fillna(70)
    df["wind"] = df["wind"].fillna(0)

    for stat in STAT_COLS:
        df[f"{stat}_std"] = df[f"{stat}_std"].fillna(df[f"{stat}_l8"])

    df["is_playoff"] = (df["game_type"] != "REG").astype(int)

    return df


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _rmse(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))


def _bias(pred, actual):
    return np.mean(pred - actual)


def main():
    df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
    df = preprocess_qb_matrix(df)

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
    naive_l8_pred = test["passing_yards_l8"]
    naive_season_pred = test["passing_yards_std"]

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "qb_passing_yards_ridge.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved pipeline to: {model_path}")


if __name__ == "__main__":
    main()
