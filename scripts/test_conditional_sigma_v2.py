"""Step 29: position-conditioned sigma(pred) probability layer for receiving yards - validation.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_conditional_sigma_v2.py
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

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.calibrate import RollingCalibrator
from src.models.train_receiving import FEATURE_COLS, TARGET, preprocess_receiving_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150
PROP_MIN_TARGETS_L8 = 3
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COVERAGE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
N_DECILES = 10
N_QUINTILES = 5
ACCEPTANCE_BAR = 0.05  # 5 percentage points


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
df_processed = preprocess_receiving_matrix(df)


# --- Reproduce step 27 walk-forward (calibrated), prop-relevant slice ---
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
    test["analysis_season"] = season

    records.append(test[["analysis_season", "pred", "residual", "targets_l8", "position"]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["targets_l8"] >= PROP_MIN_TARGETS_L8].copy()

train_data = prop_data[prop_data["analysis_season"] < 2025].copy()
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy()


# --- Step 28 reproduction: pred-only decile sigma (for the comparison column) ---
section("Step 28 reproduction: pred-only decile table (2022-2024 prop-relevant)")

train_data["decile"] = pd.qcut(train_data["pred"], N_DECILES, duplicates="drop")
decile_stats = train_data.groupby("decile", observed=True).agg(
    n=("pred", "size"),
    median_pred=("pred", "median"),
    std=("residual", "std"),
).reset_index()

print(f"{'bin_range':<20} {'n':>6} {'median_pred':>12} {'std':>10}")
for _, row in decile_stats.iterrows():
    print(f"{str(row['decile']):<20} {row['n']:>6} {row['median_pred']:>12.2f} {row['std']:>10.2f}")

pred_bin_medians = decile_stats["median_pred"].values
pred_bin_stds = decile_stats["std"].values


def sigma_pred(pred):
    return np.interp(pred, pred_bin_medians, pred_bin_stds)


# --- Step 29: position x prediction-quintile sigma table ---
section("Step 29: position x prediction-quintile sigma table (2022-2024 prop-relevant)")

positions = sorted(train_data["position"].unique())
print(f"positions present: {positions}")

position_bins = {}

for pos in positions:
    pos_data = train_data[train_data["position"] == pos].copy()
    n_bins = min(N_QUINTILES, pos_data["pred"].nunique())
    pos_data["quintile"] = pd.qcut(pos_data["pred"], n_bins, duplicates="drop")
    pos_stats = pos_data.groupby("quintile", observed=True).agg(
        n=("pred", "size"),
        median_pred=("pred", "median"),
        std=("residual", "std"),
    ).reset_index()

    print(f"\n  position {pos}:")
    print(f"  {'bin_range':<22} {'n':>6} {'median_pred':>12} {'std':>10}")
    for _, row in pos_stats.iterrows():
        print(f"  {str(row['quintile']):<22} {row['n']:>6} {row['median_pred']:>12.2f} {row['std']:>10.2f}")

    position_bins[pos] = (pos_stats["median_pred"].values, pos_stats["std"].values)


def sigma_position(pred, position):
    medians, stds = position_bins[position]
    return np.interp(pred, medians, stds)


def sigma_position_vec(preds, positions_arr):
    out = np.zeros(len(preds))
    for pos in position_bins:
        mask = positions_arr == pos
        if mask.any():
            out[mask] = np.interp(preds[mask], position_bins[pos][0], position_bins[pos][1])
    return out


# --- Standardize 2022-2024 residuals with position-aware sigma ---
section("Standardized 2022-2024 residuals: z = residual / sigma_position(pred, position)")

train_data["sigma_pos"] = sigma_position_vec(train_data["pred"].values, train_data["position"].values)
train_data["z_pos"] = train_data["residual"] / train_data["sigma_pos"]

z = train_data["z_pos"].values
print(f"overall z-pool: count={len(z)}, std={np.std(z, ddof=1):.4f} (expect ~1), "
      f"skewness={stats.skew(z):.4f} (expect ~1.1, preserved)")

print(f"\nper-position skew (visibility only):")
print(f"{'position':<10} {'n':>6} {'skew':>8}")
for pos in positions:
    pos_z = train_data.loc[train_data["position"] == pos, "z_pos"].values
    print(f"{pos:<10} {len(pos_z):>6} {stats.skew(pos_z):>8.4f}")

z_quantiles = np.quantile(z, QUANTILE_LEVELS)
print(f"\n{'quantile':>10} {'z value':>10}")
for q, zv in zip(QUANTILE_LEVELS, z_quantiles):
    print(f"{q * 100:>9.0f}% {zv:>10.4f}")

z_coverage = {level: np.quantile(z, level) for level in COVERAGE_LEVELS}


# --- Step 28 z-pool (for the comparison column) ---
train_data["sigma_pred"] = sigma_pred(train_data["pred"].values)
train_data["z_pred"] = train_data["residual"] / train_data["sigma_pred"]
z_pred_coverage = {level: np.quantile(train_data["z_pred"].values, level) for level in COVERAGE_LEVELS}


# --- Apply to 2025: 20-cell coverage table, three columns ---
section("Coverage on 2025 prop-slice: pooled (27) vs pred-only conditional (28) vs position+pred conditional (29)")

test_2025["sigma_pred"] = sigma_pred(test_2025["pred"].values)
test_2025["sigma_pos"] = sigma_position_vec(test_2025["pred"].values, test_2025["position"].values)
test_2025["pred_quartile"] = pd.qcut(test_2025["pred"], 4, duplicates="drop")

pooled_thresholds = {level: np.quantile(train_data["residual"].values, level) for level in COVERAGE_LEVELS}

max_abs_dev = 0.0
cells = []

for quartile, group in test_2025.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {quartile} (n={len(group)}):")
    print(f"  {'nominal':>8} {'pooled':>9} {'cond_pred':>10} {'cond_pos':>9} {'dev_pos':>9}")
    for level in COVERAGE_LEVELS:
        pooled_obs = np.mean(group["residual"].values <= pooled_thresholds[level])

        cond_pred_threshold = group["sigma_pred"].values * z_pred_coverage[level]
        cond_pred_obs = np.mean(group["residual"].values <= cond_pred_threshold)

        cond_pos_threshold = group["sigma_pos"].values * z_coverage[level]
        cond_pos_obs = np.mean(group["residual"].values <= cond_pos_threshold)

        dev = abs(cond_pos_obs - level)
        max_abs_dev = max(max_abs_dev, dev)
        cells.append((quartile, level, cond_pos_obs, dev))

        print(f"  {level * 100:>7.0f}% {pooled_obs * 100:>8.2f}% {cond_pred_obs * 100:>9.2f}% "
              f"{cond_pos_obs * 100:>8.2f}% {dev * 100:>8.2f}pp")


# --- Per-position coverage (visibility only) ---
section("Per-position coverage on 2025 prop-slice (position+pred conditional), visibility only")

for pos in positions:
    pos_group = test_2025[test_2025["position"] == pos]
    print(f"\n  position {pos} (n={len(pos_group)}):")
    print(f"  {'nominal':>8} {'cond_pos_obs':>14}")
    for level in COVERAGE_LEVELS:
        cond_pos_threshold = pos_group["sigma_pos"].values * z_coverage[level]
        cond_pos_obs = np.mean(pos_group["residual"].values <= cond_pos_threshold)
        print(f"  {level * 100:>7.0f}% {cond_pos_obs * 100:>13.2f}%")


# --- p10/p90 by predicted-value quartile ---
section("p10/p90 of (prediction + residual), by predicted-value quartile: pooled vs cond-pred vs cond-pos")

prop_data["pred_quartile_all"] = pd.qcut(prop_data["pred"], 4, duplicates="drop")
pooled_residuals_all_train = train_data["residual"].values

print(f"{'quartile':<20} {'median_pred':>12} {'pooled_p10':>11} {'pooled_p90':>11} "
      f"{'cpred_p10':>10} {'cpred_p90':>10} {'cpos_p10':>9} {'cpos_p90':>9}")
for quartile, group in prop_data.groupby("pred_quartile_all", observed=True):
    median_pred = group["pred"].median()
    majority_pos = group["position"].mode().iloc[0]

    pooled_p10 = median_pred + np.quantile(pooled_residuals_all_train, 0.10)
    pooled_p90 = median_pred + np.quantile(pooled_residuals_all_train, 0.90)

    sigma_pred_med = sigma_pred(median_pred)
    cpred_p10 = median_pred + sigma_pred_med * z_pred_coverage[0.10]
    cpred_p90 = median_pred + sigma_pred_med * z_pred_coverage[0.90]

    sigma_pos_med = sigma_position(median_pred, majority_pos)
    cpos_p10 = median_pred + sigma_pos_med * z_coverage[0.10]
    cpos_p90 = median_pred + sigma_pos_med * z_coverage[0.90]

    print(f"{str(quartile):<20} {median_pred:>12.2f} {pooled_p10:>11.2f} {pooled_p90:>11.2f} "
          f"{cpred_p10:>10.2f} {cpred_p90:>10.2f} {cpos_p10:>9.2f} {cpos_p90:>9.2f}")


# --- Acceptance criterion ---
section("Acceptance criterion: max |observed - nominal| across all 20 cells (position+pred conditional) vs 5-point bar")
print(f"max absolute deviation: {max_abs_dev * 100:.2f}pp")
verdict = "PASS" if max_abs_dev <= ACCEPTANCE_BAR else "FAIL"
print(f"verdict: {verdict} ({'<=' if verdict == 'PASS' else '>'} {ACCEPTANCE_BAR * 100:.0f}pp bar)")
if verdict == "FAIL":
    print("\nNo further variants per pre-registration. Quantile regression becomes the documented next architecture.")
