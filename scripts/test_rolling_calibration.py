"""In-season rolling calibration diagnostic for the QB passing yards Ridge model.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_rolling_calibration.py
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

MIN_GAMES_FOR_OFFSET = 30
OFFSET_CAP = 12


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

print(f"{'season':<8} {'raw_mae':>8} {'raw_bias':>9} {'cal_mae':>8} {'cal_bias':>9} {'final_offset':>13}")

season_2025_log = None

for season in [2022, 2023, 2024, 2025]:
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
    test["error"] = test["raw_pred"] - y_test.values

    weeks = sorted(test["week"].unique())
    calibrated_pred = np.zeros(len(test))
    offsets_log = []
    final_offset = 0.0

    for week in weeks:
        week_mask = test["week"] == week
        prior_mask = test["week"] < week
        n_completed = prior_mask.sum()

        if n_completed >= MIN_GAMES_FOR_OFFSET:
            offset = test.loc[prior_mask, "error"].mean()
            offset = np.clip(offset, -OFFSET_CAP, OFFSET_CAP)
        else:
            offset = 0.0

        calibrated_pred[week_mask.values] = test.loc[week_mask, "raw_pred"] - offset
        final_offset = offset
        offsets_log.append((week, n_completed, offset))

    cal_mae = _mae(calibrated_pred, y_test.values)
    cal_bias = _bias(calibrated_pred, y_test.values)

    print(
        f"{season:<8} {_mae(raw_pred, y_test):>8.2f} {_bias(raw_pred, y_test):>9.2f} "
        f"{cal_mae:>8.2f} {cal_bias:>9.2f} {final_offset:>13.2f}"
    )

    if season == 2025:
        season_2025_log = offsets_log

print("\n2025 week-by-week offset evolution:")
print(f"{'week':<6} {'completed_games':>16} {'offset':>8}")
for week, n_completed, offset in season_2025_log:
    print(f"{week:<6} {n_completed:>16} {offset:>8.2f}")
