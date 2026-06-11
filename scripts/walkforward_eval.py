"""Walk-forward evaluation diagnostic for the QB passing yards Ridge model.

Read-only diagnostic, no changes to train.py. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\walkforward_eval.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

print(f"{'season':<8} {'train_rows':>10} {'test_rows':>10} {'ridge_mae':>10} {'ridge_bias':>11} {'naiveL8_mae':>12} {'naiveL8_bias':>13}")
for season in [2022, 2023, 2024, 2025]:
    train = df_processed[df_processed["season"] < season]
    test = df_processed[df_processed["season"] == season]

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    ridge_pred = pipeline.predict(X_test)
    naive_l8_pred = test["passing_yards_l8"]

    print(
        f"{season:<8} {len(train):>10} {len(test):>10} "
        f"{_mae(ridge_pred, y_test):>10.2f} {_bias(ridge_pred, y_test):>11.2f} "
        f"{_mae(naive_l8_pred, y_test):>12.2f} {_bias(naive_l8_pred, y_test):>13.2f}"
    )

print("\nLeague-wide mean passing yards per start, by season (from qb_matrix, pre-dropna):")
print(df.groupby("season")["passing_yards"].mean().to_string())
