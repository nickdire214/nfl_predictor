"""Out-of-sample residual analysis for the calibrated QB passing yards model.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\analyze_residuals.py
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
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df_processed = preprocess_qb_matrix(df)

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

    test["calibrated_pred"] = calibrated
    test["residual"] = test["y_test"] - test["calibrated_pred"]
    test["analysis_season"] = season

    records.append(test[[
        "analysis_season", "residual", "calibrated_pred",
        "attempts_l8", "games_into_season", "y_test",
    ]])

all_data = pd.concat(records, ignore_index=True)
residuals = all_data["residual"].values


# --- Distribution shape ---
section("Residual distribution (pooled, calibrated out-of-sample, 2022-2025)")

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

quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
empirical_q = np.quantile(residuals, quantile_levels)
normal_q = stats.norm.ppf(quantile_levels, loc=mean, scale=std)

print(f"\n{'quantile':>10} {'empirical':>12} {'normal_fit':>12}")
for q, e, n in zip(quantile_levels, empirical_q, normal_q):
    print(f"{q * 100:>9.0f}% {e:>12.2f} {n:>12.2f}")


# --- Heteroskedasticity checks ---
section("Heteroskedasticity: residual std by calibrated-prediction quartile")
all_data["pred_quartile"] = pd.qcut(all_data["calibrated_pred"], 4, duplicates="drop")
print(all_data.groupby("pred_quartile", observed=True)["residual"].agg(["count", "std"]).to_string())

section("Heteroskedasticity: residual std by attempts_l8 quartile")
all_data["attempts_quartile"] = pd.qcut(all_data["attempts_l8"], 4, duplicates="drop")
print(all_data.groupby("attempts_quartile", observed=True)["residual"].agg(["count", "std"]).to_string())

section("Heteroskedasticity: residual std by games_into_season (0-2 vs 3+)")
all_data["season_phase"] = np.where(all_data["games_into_season"] <= 2, "0-2", "3+")
print(all_data.groupby("season_phase", observed=True)["residual"].agg(["count", "std"]).to_string())


# --- Honest coverage test ---
section("Honest-coverage test: quantiles from 2022-2024 residuals, evaluated on 2025")

train_residuals = all_data.loc[all_data["analysis_season"] < 2025, "residual"].values
test_2025_residuals = all_data.loc[all_data["analysis_season"] == 2025, "residual"].values

nominal_levels = [0.10, 0.25, 0.50, 0.75, 0.90]
thresholds = np.quantile(train_residuals, nominal_levels)

print(f"{'nominal':>8} {'threshold':>10} {'observed_coverage':>18}")
for nominal, threshold in zip(nominal_levels, thresholds):
    observed = np.mean(test_2025_residuals <= threshold)
    print(f"{nominal * 100:>7.0f}% {threshold:>10.2f} {observed * 100:>17.2f}%")
