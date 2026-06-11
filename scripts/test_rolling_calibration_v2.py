"""Shrunk in-season calibration grid test for the QB passing yards Ridge model.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_rolling_calibration_v2.py
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

SEASONS = [2022, 2023, 2024, 2025]
N0_VALUES = [50, 100, 150, 200, 300]


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


def calibrated_predictions(test, n0):
    weeks = sorted(test["week"].unique())
    calibrated = np.zeros(len(test))

    for week in weeks:
        week_mask = (test["week"] == week).values
        prior_mask = (test["week"] < week).values
        n = prior_mask.sum()

        if n > 0:
            mean_bias = test.loc[prior_mask, "error"].mean()
        else:
            mean_bias = 0.0

        offset = mean_bias * (n / (n + n0))
        calibrated[week_mask] = test.loc[week_mask, "raw_pred"] - offset

    return calibrated


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

season_data = {}
raw_results = {}

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
    test["error"] = test["raw_pred"] - test["y_test"]

    season_data[season] = test
    raw_results[season] = (_mae(raw_pred, y_test.values), _bias(raw_pred, y_test.values))


print("Reference: raw (uncalibrated) Ridge")
print(f"{'season':<8} {'raw_mae':>8} {'raw_bias':>9}")
for season in SEASONS:
    mae, bias = raw_results[season]
    print(f"{season:<8} {mae:>8.2f} {bias:>9.2f}")

summary_rows = []

for n0 in N0_VALUES:
    print(f"\n=== n0 = {n0} ===")
    print(f"{'season':<8} {'raw_mae':>8} {'raw_bias':>9} {'cal_mae':>8} {'cal_bias':>9}")

    sum_mae = 0.0
    sum_abs_bias = 0.0
    worst_degradation = -np.inf

    for season in SEASONS:
        test = season_data[season]
        raw_mae, raw_bias = raw_results[season]

        cal_pred = calibrated_predictions(test, n0)
        cal_mae = _mae(cal_pred, test["y_test"].values)
        cal_bias = _bias(cal_pred, test["y_test"].values)

        print(f"{season:<8} {raw_mae:>8.2f} {raw_bias:>9.2f} {cal_mae:>8.2f} {cal_bias:>9.2f}")

        sum_mae += cal_mae
        sum_abs_bias += abs(cal_bias)
        worst_degradation = max(worst_degradation, cal_mae - raw_mae)

    summary_rows.append((n0, sum_mae, sum_abs_bias, worst_degradation))

raw_sum_mae = sum(raw_results[s][0] for s in SEASONS)
raw_sum_abs_bias = sum(abs(raw_results[s][1]) for s in SEASONS)

print("\n=== Summary ===")
print(f"{'n0':<10} {'sum_mae':>10} {'sum_abs_bias':>14} {'worst_mae_degradation':>22}")
print(f"{'raw':<10} {raw_sum_mae:>10.2f} {raw_sum_abs_bias:>14.2f} {'-':>22}")
for n0, sum_mae, sum_abs_bias, worst_degradation in summary_rows:
    print(f"{n0:<10} {sum_mae:>10.2f} {sum_abs_bias:>14.2f} {worst_degradation:>22.2f}")
