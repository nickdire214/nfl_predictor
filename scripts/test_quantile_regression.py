"""Step 30: quantile regression validation for receiving yards probability layer.

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_quantile_regression.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
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
RANDOM_STATE = 42


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


df = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
df_processed = preprocess_receiving_matrix(df)


# --- Reproduce steps 27-29 walk-forward (calibrated Ridge), prop-relevant slice ---
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

    records.append(test[["analysis_season", "pred", "residual", "y_test", "targets_l8", "position", "week", "gsis_id"]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["targets_l8"] >= PROP_MIN_TARGETS_L8].copy()

train_data = prop_data[prop_data["analysis_season"] < 2025].copy()
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy()


# --- Step 28: pred-only decile sigma ---
train_data["decile"] = pd.qcut(train_data["pred"], N_DECILES, duplicates="drop")
decile_stats = train_data.groupby("decile", observed=True).agg(
    n=("pred", "size"),
    median_pred=("pred", "median"),
    std=("residual", "std"),
).reset_index()

pred_bin_medians = decile_stats["median_pred"].values
pred_bin_stds = decile_stats["std"].values


def sigma_pred(pred):
    return np.interp(pred, pred_bin_medians, pred_bin_stds)


# --- Step 29: position x prediction-quintile sigma ---
positions = sorted(train_data["position"].unique())
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
    position_bins[pos] = (pos_stats["median_pred"].values, pos_stats["std"].values)


def sigma_position_vec(preds, positions_arr):
    out = np.zeros(len(preds))
    for pos in position_bins:
        mask = positions_arr == pos
        if mask.any():
            out[mask] = np.interp(preds[mask], position_bins[pos][0], position_bins[pos][1])
    return out


train_data["sigma_pred"] = sigma_pred(train_data["pred"].values)
train_data["z_pred"] = train_data["residual"] / train_data["sigma_pred"]
z_pred_coverage = {level: np.quantile(train_data["z_pred"].values, level) for level in COVERAGE_LEVELS}

train_data["sigma_pos"] = sigma_position_vec(train_data["pred"].values, train_data["position"].values)
train_data["z_pos"] = train_data["residual"] / train_data["sigma_pos"]
z_pos_coverage = {level: np.quantile(train_data["z_pos"].values, level) for level in COVERAGE_LEVELS}

pooled_thresholds = {level: np.quantile(train_data["residual"].values, level) for level in COVERAGE_LEVELS}

test_2025["sigma_pred"] = sigma_pred(test_2025["pred"].values)
test_2025["sigma_pos"] = sigma_position_vec(test_2025["pred"].values, test_2025["position"].values)
test_2025["pred_quartile"] = pd.qcut(test_2025["pred"], 4, duplicates="drop")


# --- Step 30: quantile regression ---
section("Step 30: HistGradientBoostingRegressor quantile models (train: 2021-2024, test: 2025)")

train_qr = df_processed[df_processed["season"] <= 2024]
test_qr_full = df_processed[df_processed["season"] == 2025].sort_values("week").reset_index(drop=True)

X_train_qr, y_train_qr = train_qr[FEATURE_COLS], train_qr[TARGET]
X_test_qr = test_qr_full[FEATURE_COLS]

q_cols = []
for q in QUANTILE_LEVELS:
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_depth=3, random_state=RANDOM_STATE,
    )
    model.fit(X_train_qr, y_train_qr)
    col = f"q{int(round(q * 100)):02d}"
    test_qr_full[col] = model.predict(X_test_qr)
    q_cols.append(col)

print(f"Train rows: {len(train_qr)} (seasons 2021-2024)")
print(f"Test rows: {len(test_qr_full)} (season 2025)")
print(f"Quantile models fit for q in {QUANTILE_LEVELS}")


# --- Crossing check (before sorting) ---
section("Quantile crossing check (before sorting)")

raw_q = test_qr_full[q_cols].values
is_sorted_row = np.all(np.diff(raw_q, axis=1) >= 0, axis=1)
frac_crossing = 1.0 - is_sorted_row.mean()
print(f"rows: {len(test_qr_full)}")
print(f"fraction with any crossing (non-monotonic quantiles): {frac_crossing:.4f}")

sorted_q = np.sort(raw_q, axis=1)
for i, col in enumerate(q_cols):
    test_qr_full[col] = sorted_q[:, i]


# --- Restrict to prop-relevant 2025 slice and merge with Ridge-based pred_quartile ---
prop_mask_qr = (test_qr_full["targets_l8"] >= PROP_MIN_TARGETS_L8).values
test_2025_qr = test_qr_full[prop_mask_qr].reset_index(drop=True)

merged = test_2025.merge(
    test_2025_qr[["gsis_id", "week", TARGET] + q_cols],
    on=["gsis_id", "week"],
    how="inner",
    suffixes=("", "_qr"),
)
print(f"\nmerged rows: {len(merged)} (test_2025: {len(test_2025)}, test_2025_qr: {len(test_2025_qr)})")
assert len(merged) == len(test_2025), "merge dropped rows -- gsis_id/week not a unique join key"


# --- Context metric: MAE of q50 vs Ridge ---
section("Context: MAE of quantile-regression q50 vs Ridge (calibrated), 2025 prop-slice")

qr_q50_mae = _mae(merged["q50"].values, merged["y_test"].values)
ridge_mae = _mae(merged["pred"].values, merged["y_test"].values)
print(f"Ridge (calibrated) MAE: {ridge_mae:.2f}")
print(f"Quantile-reg q50 MAE:   {qr_q50_mae:.2f}")


# --- 20-cell coverage table: 4-way comparison ---
section("Coverage on 2025 prop-slice: pooled (27) / cond-pred (28) / cond-pos (29) / quantile-reg (30)")

level_to_col = {level: f"q{int(round(level * 100)):02d}" for level in COVERAGE_LEVELS}

max_abs_dev_qr = 0.0

for quartile, group in merged.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {quartile} (n={len(group)}):")
    print(f"  {'nominal':>8} {'pooled':>9} {'cond_pred':>10} {'cond_pos':>9} {'qreg':>9} {'dev_qreg':>9}")
    for level in COVERAGE_LEVELS:
        pooled_obs = np.mean(group["residual"].values <= pooled_thresholds[level])

        cond_pred_threshold = group["sigma_pred"].values * z_pred_coverage[level]
        cond_pred_obs = np.mean(group["residual"].values <= cond_pred_threshold)

        cond_pos_threshold = group["sigma_pos"].values * z_pos_coverage[level]
        cond_pos_obs = np.mean(group["residual"].values <= cond_pos_threshold)

        qreg_col = level_to_col[level]
        qreg_obs = np.mean(group["y_test"].values <= group[qreg_col].values)

        dev_qreg = abs(qreg_obs - level)
        max_abs_dev_qr = max(max_abs_dev_qr, dev_qreg)

        print(f"  {level * 100:>7.0f}% {pooled_obs * 100:>8.2f}% {cond_pred_obs * 100:>9.2f}% "
              f"{cond_pos_obs * 100:>8.2f}% {qreg_obs * 100:>8.2f}% {dev_qreg * 100:>8.2f}pp")


# --- p10/p90 by predicted-value quartile ---
section("p10/p90 by predicted-value quartile: pooled / cond-pred / cond-pos / quantile-reg")

print(f"{'quartile':<20} {'median_pred':>12} {'pooled_p10':>11} {'pooled_p90':>11} "
      f"{'cpred_p10':>10} {'cpred_p90':>10} {'cpos_p10':>9} {'cpos_p90':>9} {'qr_p10':>8} {'qr_p90':>8}")

for quartile, group in merged.groupby("pred_quartile", observed=True):
    median_pred = group["pred"].median()
    majority_pos = group["position"].mode().iloc[0]

    pooled_p10 = median_pred + np.quantile(train_data["residual"].values, 0.10)
    pooled_p90 = median_pred + np.quantile(train_data["residual"].values, 0.90)

    sigma_pred_med = sigma_pred(median_pred)
    cpred_p10 = median_pred + sigma_pred_med * z_pred_coverage[0.10]
    cpred_p90 = median_pred + sigma_pred_med * z_pred_coverage[0.90]

    sigma_pos_med = np.interp(median_pred, position_bins[majority_pos][0], position_bins[majority_pos][1])
    cpos_p10 = median_pred + sigma_pos_med * z_pos_coverage[0.10]
    cpos_p90 = median_pred + sigma_pos_med * z_pos_coverage[0.90]

    qr_p10 = group["q10"].median()
    qr_p90 = group["q90"].median()

    print(f"{str(quartile):<20} {median_pred:>12.2f} {pooled_p10:>11.2f} {pooled_p90:>11.2f} "
          f"{cpred_p10:>10.2f} {cpred_p90:>10.2f} {cpos_p10:>9.2f} {cpos_p90:>9.2f} "
          f"{qr_p10:>8.2f} {qr_p90:>8.2f}")


# --- Acceptance criterion ---
section("Acceptance criterion: max |observed - nominal| across all 20 cells (quantile-reg) vs 5-point bar")
print(f"max absolute deviation: {max_abs_dev_qr * 100:.2f}pp")
verdict = "PASS" if max_abs_dev_qr <= ACCEPTANCE_BAR else "FAIL"
print(f"verdict: {verdict} ({'<=' if verdict == 'PASS' else '>'} {ACCEPTANCE_BAR * 100:.0f}pp bar)")

if verdict == "FAIL":
    section("Side-by-side: max |dev| across all four architectures (steps 27-30)")
    max_dev_pooled = max(
        abs(np.mean(g["residual"].values <= pooled_thresholds[level]) - level)
        for _, g in merged.groupby("pred_quartile", observed=True)
        for level in COVERAGE_LEVELS
    )
    max_dev_cond_pred = max(
        abs(np.mean(g["residual"].values <= g["sigma_pred"].values * z_pred_coverage[level]) - level)
        for _, g in merged.groupby("pred_quartile", observed=True)
        for level in COVERAGE_LEVELS
    )
    max_dev_cond_pos = max(
        abs(np.mean(g["residual"].values <= g["sigma_pos"].values * z_pos_coverage[level]) - level)
        for _, g in merged.groupby("pred_quartile", observed=True)
        for level in COVERAGE_LEVELS
    )
    print(f"{'architecture':<28} {'max |dev|':>10}")
    print(f"{'27 pooled':<28} {max_dev_pooled * 100:>9.2f}pp")
    print(f"{'28 cond-pred':<28} {max_dev_cond_pred * 100:>9.2f}pp")
    print(f"{'29 cond-pos':<28} {max_dev_cond_pos * 100:>9.2f}pp")
    print(f"{'30 quantile-reg':<28} {max_abs_dev_qr * 100:>9.2f}pp")
