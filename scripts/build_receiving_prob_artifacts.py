"""Step 33 (part 1): build the receiving probability-layer artifacts.

Runs the standard walk-forward (2022-2025, Ridge + RollingCalibrator(n0=150)
applied week-by-week, as adopted in step 27) and collects ALL calibrated
out-of-sample (prediction, position, residual) triples for the prop-relevant
slice across all four seasons. Fits position x prediction-quintile sigma bins
(step 29's winning architecture) on this full pool, and builds the z-pool
(residual / sigma_pos, mean-centered) that supplies distribution shape.

Saves:
    models/receiving_sigma_table.parquet (position, bin_median_pred, std, n)
    models/receiving_z_pool.npy

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\build_receiving_prob_artifacts.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.calibrate import RollingCalibrator
from src.models.train_receiving import FEATURE_COLS, TARGET, preprocess_receiving_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150
PROP_MIN_TARGETS_L8 = 3
N_QUINTILES = 5
# The receiving model distinguishes only these positions (pos_WR/pos_TE dummies,
# everything else encoded as the RB baseline). Canonical position (step 35) can
# now surface a handful of FB/CB pass-catchers in the prop pool (e.g. Travis
# Hunter); they stay in the matrix/training as baseline-encoded rows but are not
# given their own sigma bins -- too few to estimate, and predict_receiving maps
# any unmodeled position to the RB baseline.
MODELED_POSITIONS = ["RB", "TE", "WR"]


df = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
df_processed = preprocess_receiving_matrix(df)


# --- Walk-forward (2022-2025): collect calibrated OOS (pred, position, residual) ---
records = []

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

    test["pred"] = calibrated
    test["residual"] = test["y_test"] - test["pred"]

    records.append(test[["pred", "residual", "targets_l8", "position"]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["targets_l8"] >= PROP_MIN_TARGETS_L8].copy()

excluded = prop_data[~prop_data["position"].isin(MODELED_POSITIONS)]
if len(excluded):
    print(
        f"Excluding {len(excluded)} prop-pool rows with unmodeled position "
        f"(not in {MODELED_POSITIONS}) from sigma/z-pool fitting:"
    )
    print(excluded["position"].value_counts().to_string())
prop_data = prop_data[prop_data["position"].isin(MODELED_POSITIONS)].copy()

print(f"OOS prop-relevant pool (2022-2025): {len(prop_data)} rows")


# --- Position x prediction-quintile sigma bins, fit on the full pool ---
positions = sorted(prop_data["position"].unique())
print(f"positions present: {positions}")

sigma_rows = []
position_bins = {}

for pos in positions:
    pos_data = prop_data[prop_data["position"] == pos].copy()
    n_bins = min(N_QUINTILES, pos_data["pred"].nunique())
    pos_data["quintile"] = pd.qcut(pos_data["pred"], n_bins, duplicates="drop")
    pos_stats = pos_data.groupby("quintile", observed=True).agg(
        n=("pred", "size"),
        median_pred=("pred", "median"),
        std=("residual", "std"),
    ).reset_index()
    pos_stats = pos_stats.sort_values("median_pred").reset_index(drop=True)

    # Enforce non-decreasing std across bins (step 28 found heteroskedasticity is
    # smooth and monotonic; small dips in thinner bins are sampling noise).
    pos_stats["std"] = np.maximum.accumulate(pos_stats["std"].values)

    for _, row in pos_stats.iterrows():
        sigma_rows.append({
            "position": pos,
            "bin_median_pred": row["median_pred"],
            "std": row["std"],
            "n": int(row["n"]),
        })
    position_bins[pos] = (pos_stats["median_pred"].values, pos_stats["std"].values)

sigma_table = pd.DataFrame(sigma_rows)

print(f"\n{'position':<10} {'bin_median_pred':>16} {'std':>10} {'n':>6}")
for _, row in sigma_table.iterrows():
    print(f"{row['position']:<10} {row['bin_median_pred']:>16.2f} {row['std']:>10.2f} {row['n']:>6}")


# --- z-pool: residual / sigma_pos(pred), mean-centered ---
def sigma_position_vec(preds, positions_arr):
    out = np.zeros(len(preds))
    for pos in position_bins:
        mask = positions_arr == pos
        if mask.any():
            out[mask] = np.interp(preds[mask], position_bins[pos][0], position_bins[pos][1])
    return out


prop_data["sigma_pos"] = sigma_position_vec(prop_data["pred"].values, prop_data["position"].values)
z = prop_data["residual"].values / prop_data["sigma_pos"].values
z_centered = z - np.mean(z)

print(f"\nz-pool: count={len(z_centered)}, std={np.std(z_centered, ddof=1):.4f}, "
      f"skewness={stats.skew(z_centered):.4f}, mean={np.mean(z_centered):.2e} (~0)")


# --- Save artifacts ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)

sigma_table_path = MODELS_DIR / "receiving_sigma_table.parquet"
z_pool_path = MODELS_DIR / "receiving_z_pool.npy"

sigma_table.to_parquet(sigma_table_path, index=False)
np.save(z_pool_path, z_centered)

print(f"\nSaved sigma table to: {sigma_table_path}")
print(f"Saved z-pool to: {z_pool_path}")
