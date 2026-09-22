"""Hard-assertion tests for RollingCalibrator.

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_calibrate.py
"""

import math
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
from src.models.calibrate import RollingCalibrator
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# --- (a) offset is 0.0 with no data ---
section("(a) offset is 0.0 with no data")
calibrator = RollingCalibrator(n0=150)
assert calibrator.offset == 0.0, f"expected 0.0, got {calibrator.offset}"
print("PASS: offset == 0.0 with no data")


# --- (b) offset matches hand-computed shrunk value ---
section("(b) offset matches hand-computed shrunk value")
calibrator = RollingCalibrator(n0=150)
errors = [10, 20, 30]
calibrator.update(predicted=[110, 120, 130], actual=[100, 100, 100])  # errors = [10, 20, 30]

mean_error = np.mean(errors)
n = len(errors)
expected_offset = mean_error * n / (n + 150)

assert math.isclose(calibrator.offset, expected_offset), (
    f"offset ({calibrator.offset}) != hand-computed expected ({expected_offset})"
)
print(f"PASS: offset ({calibrator.offset:.6f}) == hand-computed expected ({expected_offset:.6f})")


# --- (c) full replay of 2025 walk-forward ---
section("(c) full 2025 walk-forward replay matches v2 diagnostic (n0=150)")

df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

train = df_processed[df_processed["season"] < 2025]
test = df_processed[df_processed["season"] == 2025].sort_values("week").reset_index(drop=True)

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

calibrator = RollingCalibrator(n0=150)
calibrated = np.zeros(len(test))

for week in sorted(test["week"].unique()):
    week_mask = (test["week"] == week).values
    week_raw_pred = test.loc[week_mask, "raw_pred"].values
    week_actual = test.loc[week_mask, "y_test"].values

    calibrated[week_mask] = calibrator.calibrate(week_raw_pred)
    calibrator.update(week_raw_pred, week_actual)

mae = np.mean(np.abs(calibrated - test["y_test"].values))
bias = np.mean(calibrated - test["y_test"].values)

# Re-pinned 2026-09-22 for the step-95 leading-passer correction. The matrix
# previously labelled a team-game with schedules' designated starter, which on
# 127 of 2,912 played team-games was not the man who actually threw the ball --
# so the model was trained partly on a backup's four-attempt cameo labelled as
# that team's start. Deriving the starter from the leading passer corrected
# those labels and added 50 previously-dropped team-games, which moved this
# replay: MAE 57.73 -> 55.75 (better by ~2 yards), bias 3.72 -> 3.67.
expected_mae, expected_bias = 55.75, 3.67
assert round(mae, 2) == expected_mae, f"MAE ({mae:.2f}) != expected ({expected_mae})"
assert round(bias, 2) == expected_bias, f"bias ({bias:.2f}) != expected ({expected_bias})"
print(f"PASS: 2025 calibrated MAE ({mae:.2f}) == expected ({expected_mae})")
print(f"PASS: 2025 calibrated bias ({bias:.2f}) == expected ({expected_bias})")


# --- (d) reset() returns offset to 0.0 ---
section("(d) reset() returns offset to 0.0")
assert calibrator.offset != 0.0, "calibrator should have a nonzero offset before reset"
calibrator.reset()
assert calibrator.offset == 0.0, f"expected 0.0 after reset, got {calibrator.offset}"
print("PASS: offset == 0.0 after reset()")


print("\nAll tests passed.")
