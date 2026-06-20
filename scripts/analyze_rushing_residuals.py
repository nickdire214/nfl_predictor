"""Step 45: rushing calibration + residual/heteroskedasticity diagnostics.

Read-only evidence-gathering, no pipeline changes. Mirrors
analyze_receiving_residuals.py for the RB rushing stack. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\analyze_rushing_residuals.py
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
from src.models.train_rushing import FEATURE_COLS, TARGET, preprocess_rushing_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150
PROP_MIN_CARRIES_L8 = 8
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COVERAGE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


df = pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet")
df_processed = preprocess_rushing_matrix(df)


# --- Part A: calibration test ---
section("Part A: Calibration (raw vs RollingCalibrator n0=150, fed ALL rows)")
print(f"{'season':<8} {'raw_mae':>8} {'raw_bias':>9} {'cal_mae':>8} {'cal_bias':>9}")

records = []
sum_raw_mae = sum_cal_mae = 0.0
sum_abs_raw_bias = sum_abs_cal_bias = 0.0

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

    test["calibrated_pred"] = calibrated
    test["raw_residual"] = test["y_test"] - test["raw_pred"]
    test["cal_residual"] = test["y_test"] - test["calibrated_pred"]
    test["analysis_season"] = season

    raw_mae, raw_bias = _mae(test["raw_pred"], test["y_test"]), _bias(test["raw_pred"], test["y_test"])
    cal_mae, cal_bias = _mae(test["calibrated_pred"], test["y_test"]), _bias(test["calibrated_pred"], test["y_test"])

    print(f"{season:<8} {raw_mae:>8.2f} {raw_bias:>9.2f} {cal_mae:>8.2f} {cal_bias:>9.2f}")

    sum_raw_mae += raw_mae
    sum_cal_mae += cal_mae
    sum_abs_raw_bias += abs(raw_bias)
    sum_abs_cal_bias += abs(cal_bias)

    records.append(test[[
        "analysis_season", "raw_pred", "calibrated_pred", "y_test",
        "raw_residual", "cal_residual", "carries_l8",
    ]])

all_data = pd.concat(records, ignore_index=True)

mae_cost = sum_cal_mae - sum_raw_mae
bias_reduction = sum_abs_raw_bias - sum_abs_cal_bias
use_calibrated = bias_reduction > 0 and mae_cost <= 0.5

print(f"\nSum |bias|: raw={sum_abs_raw_bias:.2f}, calibrated={sum_abs_cal_bias:.2f} "
      f"(reduction={bias_reduction:.2f})")
print(f"Sum MAE: raw={sum_raw_mae:.2f}, calibrated={sum_cal_mae:.2f} (cost={mae_cost:.2f})")

if use_calibrated:
    print("-> Calibration helps (reduces bias at <=0.5 MAE cost): using CALIBRATED predictions/residuals for Part B+")
    pred_col, residual_col = "calibrated_pred", "cal_residual"
else:
    print("-> Calibration does not clearly help: using RAW (uncalibrated) predictions/residuals for Part B+")
    pred_col, residual_col = "raw_pred", "raw_residual"


# --- Part B: pooled residual distribution (prop-relevant slice) ---
prop_data = all_data[all_data["carries_l8"] >= PROP_MIN_CARRIES_L8].copy()

section(f"Part B: Pooled residual distribution, prop-relevant slice (carries_l8 >= {PROP_MIN_CARRIES_L8}), "
        f"using {residual_col}")

residuals = prop_data[residual_col].values
count = len(residuals)
mean = np.mean(residuals)
std = np.std(residuals, ddof=1)
skewness = stats.skew(residuals)
excess_kurtosis = stats.kurtosis(residuals)

print(f"count: {count}")
print(f"mean: {mean:.4f}")
print(f"std: {std:.4f}")
print(f"skewness: {skewness:.4f}")
print(f"excess kurtosis: {excess_kurtosis:.4f}")

empirical_q = np.quantile(residuals, QUANTILE_LEVELS)
normal_q = stats.norm.ppf(QUANTILE_LEVELS, loc=mean, scale=std)

print(f"\n{'quantile':>10} {'empirical':>12} {'normal_fit':>12}")
for q, e, n in zip(QUANTILE_LEVELS, empirical_q, normal_q):
    print(f"{q * 100:>9.0f}% {e:>12.2f} {n:>12.2f}")


# --- Heteroskedasticity (RB only, no position split) ---
section("Heteroskedasticity: residual std by predicted-value quartile (prop-relevant slice)")
prop_data["pred_quartile"] = pd.qcut(prop_data[pred_col], 4, duplicates="drop")
print(prop_data.groupby("pred_quartile", observed=True)[residual_col].agg(["count", "std"]).to_string())

section("Heteroskedasticity: residual std by carries_l8 quartile (prop-relevant slice)")
prop_data["carries_quartile"] = pd.qcut(prop_data["carries_l8"], 4, duplicates="drop")
print(prop_data.groupby("carries_quartile", observed=True)[residual_col].agg(["count", "std"]).to_string())


section("Pooled-residual p10/p90 of (prediction + residual), by predicted-value quartile")
print(f"{'quartile':<24} {'median_pred':>12} {'p10':>10} {'p90':>10}")
bottom_quartile_p10 = None
for i, (quartile, group) in enumerate(prop_data.groupby("pred_quartile", observed=True)):
    median_pred = group[pred_col].median()
    p10 = median_pred + np.quantile(residuals, 0.10)
    p90 = median_pred + np.quantile(residuals, 0.90)
    if i == 0:
        bottom_quartile_p10 = p10
    print(f"{str(quartile):<24} {median_pred:>12.2f} {p10:>10.2f} {p90:>10.2f}")

print(f"\nNegative-yardage / zero-floor check (bottom predicted-value quartile p10 = {bottom_quartile_p10:.2f}):")
if bottom_quartile_p10 <= 3.0:
    print(f"  FLAG: bottom-quartile p10 sits at/near 0 ({bottom_quartile_p10:.2f}). Rushing has a large")
    print("  zero/near-zero mass (dressed backs with 1-2 carries for 0-3 yards). Per the receiving")
    print("  step-31 point-mass-at-zero lesson, coverage evaluation next step needs TIE-AWARE treatment.")
else:
    print(f"  OK: bottom-quartile p10 = {bottom_quartile_p10:.2f}, not at/near the zero floor.")


# --- Honest coverage test ---
section("Honest-coverage test: pooled 2022-2024 prop-slice residual quantiles, evaluated on 2025 prop-slice")

train_residuals = prop_data.loc[prop_data["analysis_season"] < 2025, residual_col].values
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy()
test_2025_residuals = test_2025[residual_col].values

thresholds = np.quantile(train_residuals, COVERAGE_LEVELS)

print("Overall:")
print(f"{'nominal':>8} {'threshold':>10} {'observed_coverage':>18}")
for nominal, threshold in zip(COVERAGE_LEVELS, thresholds):
    observed = np.mean(test_2025_residuals <= threshold)
    print(f"{nominal * 100:>7.0f}% {threshold:>10.2f} {observed * 100:>17.2f}%")

print("\nBy predicted-value quartile (2025 prop-slice):")
test_2025["pred_quartile"] = pd.qcut(test_2025[pred_col], 4, duplicates="drop")
for quartile, group in test_2025.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {quartile} (n={len(group)}):")
    print(f"  {'nominal':>8} {'threshold':>10} {'observed_coverage':>18}")
    group_residuals = group[residual_col].values
    for nominal, threshold in zip(COVERAGE_LEVELS, thresholds):
        observed = np.mean(group_residuals <= threshold)
        print(f"  {nominal * 100:>7.0f}% {threshold:>10.2f} {observed * 100:>17.2f}%")
