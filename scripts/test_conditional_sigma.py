"""Step 28: conditional sigma(pred) probability layer for receiving yards — validation.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_conditional_sigma.py
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

    records.append(test[["analysis_season", "pred", "residual", "targets_l8"]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["targets_l8"] >= PROP_MIN_TARGETS_L8].copy()

train_data = prop_data[prop_data["analysis_season"] < 2025].copy()
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy()


# --- Decile binning: bin-median-prediction -> bin-std ---
section("Decile table (2022-2024 prop-relevant): bin range, n, std")

train_data["decile"] = pd.qcut(train_data["pred"], N_DECILES, duplicates="drop")
decile_stats = train_data.groupby("decile", observed=True).agg(
    n=("pred", "size"),
    median_pred=("pred", "median"),
    std=("residual", "std"),
).reset_index()

print(f"{'bin_range':<20} {'n':>6} {'median_pred':>12} {'std':>10}")
for _, row in decile_stats.iterrows():
    print(f"{str(row['decile']):<20} {row['n']:>6} {row['median_pred']:>12.2f} {row['std']:>10.2f}")

bin_medians = decile_stats["median_pred"].values
bin_stds = decile_stats["std"].values


def sigma(pred):
    return np.interp(pred, bin_medians, bin_stds)


# --- Standardize 2022-2024 residuals ---
section("Standardized 2022-2024 residuals: z = residual / sigma(pred)")

train_data["sigma"] = sigma(train_data["pred"].values)
train_data["z"] = train_data["residual"] / train_data["sigma"]

z = train_data["z"].values
z_count = len(z)
z_std = np.std(z, ddof=1)
z_skew = stats.skew(z)

print(f"count: {z_count}")
print(f"std: {z_std:.4f} (expect ~1)")
print(f"skewness: {z_skew:.4f} (expect ~1.1, preserved from step 27)")

z_quantiles = np.quantile(z, QUANTILE_LEVELS)
print(f"\n{'quantile':>10} {'z value':>10}")
for q, zv in zip(QUANTILE_LEVELS, z_quantiles):
    print(f"{q * 100:>9.0f}% {zv:>10.4f}")

z_coverage = {level: np.quantile(z, level) for level in COVERAGE_LEVELS}


# --- Apply to 2025: conditional coverage by predicted-value quartile ---
section("Coverage on 2025 prop-slice: conditional sigma(pred) vs step-27 pooled, by predicted-value quartile")

test_2025["sigma"] = sigma(test_2025["pred"].values)
test_2025["pred_quartile"] = pd.qcut(test_2025["pred"], 4, duplicates="drop")

# Step 27 pooled thresholds (residual-space, from 2022-2024 pooled residuals)
pooled_thresholds = {level: np.quantile(train_data["residual"].values, level) for level in COVERAGE_LEVELS}

max_abs_dev = 0.0
cells = []

for quartile, group in test_2025.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {quartile} (n={len(group)}):")
    print(f"  {'nominal':>8} {'pooled_obs':>12} {'conditional_obs':>16} {'cond_dev':>10}")
    for level in COVERAGE_LEVELS:
        pooled_obs = np.mean(group["residual"].values <= pooled_thresholds[level])

        cond_threshold = group["sigma"].values * z_coverage[level]
        conditional_obs = np.mean(group["residual"].values <= cond_threshold)

        dev = abs(conditional_obs - level)
        max_abs_dev = max(max_abs_dev, dev)
        cells.append((quartile, level, conditional_obs, dev))

        print(f"  {level * 100:>7.0f}% {pooled_obs * 100:>11.2f}% {conditional_obs * 100:>15.2f}% {dev * 100:>9.2f}pp")


# --- p10/p90 by predicted-value quartile: conditional vs pooled ---
section("p10/p90 of (prediction + residual), by predicted-value quartile: conditional vs pooled (step 27)")

prop_data["pred_quartile_all"] = pd.qcut(prop_data["pred"], 4, duplicates="drop")
pooled_residuals_all_train = train_data["residual"].values  # 2022-2024 pool, matches step 27's pooled approach basis

print(f"{'quartile':<20} {'median_pred':>12} {'pooled_p10':>11} {'pooled_p90':>11} {'cond_p10':>10} {'cond_p90':>10}")
for quartile, group in prop_data.groupby("pred_quartile_all", observed=True):
    median_pred = group["pred"].median()

    pooled_p10 = median_pred + np.quantile(pooled_residuals_all_train, 0.10)
    pooled_p90 = median_pred + np.quantile(pooled_residuals_all_train, 0.90)

    sigma_med = sigma(median_pred)
    cond_p10 = median_pred + sigma_med * z_coverage[0.10]
    cond_p90 = median_pred + sigma_med * z_coverage[0.90]

    print(f"{str(quartile):<20} {median_pred:>12.2f} {pooled_p10:>11.2f} {pooled_p90:>11.2f} "
          f"{cond_p10:>10.2f} {cond_p90:>10.2f}")


# --- Acceptance criterion ---
section("Acceptance criterion: max |observed - nominal| across all 20 cells vs 5-point bar")
print(f"max absolute deviation: {max_abs_dev * 100:.2f}pp")
verdict = "PASS" if max_abs_dev <= ACCEPTANCE_BAR else "FAIL"
print(f"verdict: {verdict} ({'<=' if verdict == 'PASS' else '>'} {ACCEPTANCE_BAR * 100:.0f}pp bar)")
