"""Build the pooled out-of-sample residual distribution for the probability layer.

Runs the standard walk-forward (2022-2025, Ridge + RollingCalibrator(n0=150)
applied week-by-week) exactly as in analyze_residuals.py, mean-centers the
pooled residuals, and saves them to models/qb_residual_pool.npy.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\build_residual_pool.py
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
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.calibrate import RollingCalibrator
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

residual_chunks = []

for season in SEASONS:
    train = df_processed[df_processed["season"] < season]
    test = df_processed[df_processed["season"] == season].sort_values("week").reset_index(drop=True)

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    raw_pred = pipeline.predict(X_test)
    test = test.copy()
    test["raw_pred"] = raw_pred
    test["y_test"] = y_test.values

    calibrator = RollingCalibrator(n0=N0)
    calibrated = np.zeros(len(test))

    for week in sorted(test["week"].unique()):
        week_mask = (test["week"] == week).values
        week_raw_pred = test.loc[week_mask, "raw_pred"].values
        week_actual = test.loc[week_mask, "y_test"].values

        calibrated[week_mask] = calibrator.calibrate(week_raw_pred)
        calibrator.update(week_raw_pred, week_actual)

    residual_chunks.append(test["y_test"].values - calibrated)

residuals = np.concatenate(residual_chunks)
residual_pool = residuals - np.mean(residuals)

MODELS_DIR.mkdir(parents=True, exist_ok=True)
pool_path = MODELS_DIR / "qb_residual_pool.npy"
np.save(pool_path, residual_pool)

print(f"count: {len(residual_pool)}")
print(f"std: {np.std(residual_pool, ddof=1):.4f}")
print(f"mean: {np.mean(residual_pool):.2e} (~0)")
print(f"saved to: {pool_path}")
