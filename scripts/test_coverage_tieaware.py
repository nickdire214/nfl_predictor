"""Step 31: tie-aware coverage re-evaluation for all four probability-layer architectures.

Measurement correction, not a new architecture -- same predictions/thresholds as steps 27-30,
just a tie-aware observed-coverage formula: P(actual < threshold) + 0.5 * P(actual == threshold).

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_coverage_tieaware.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
MATERIAL_CHANGE = 0.005  # 0.5 percentage points


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
section("Step 30 reproduction: HistGradientBoostingRegressor quantile models (train: 2021-2024, test: 2025)")

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

raw_q = test_qr_full[q_cols].values
sorted_q = np.sort(raw_q, axis=1)
for i, col in enumerate(q_cols):
    test_qr_full[col] = sorted_q[:, i]

prop_mask_qr = (test_qr_full["targets_l8"] >= PROP_MIN_TARGETS_L8).values
test_2025_qr = test_qr_full[prop_mask_qr].reset_index(drop=True)

merged = test_2025.merge(
    test_2025_qr[["gsis_id", "week", TARGET] + q_cols],
    on=["gsis_id", "week"],
    how="inner",
    suffixes=("", "_qr"),
)
assert len(merged) == len(test_2025), "merge dropped rows -- gsis_id/week not a unique join key"

level_to_col = {level: f"q{int(round(level * 100)):02d}" for level in COVERAGE_LEVELS}


# --- Step 31: factual check on bottom quartile ---
section("Step 31: factual check -- bottom-quartile (2025 prop-slice) actual-yards point mass at zero")

bottom_quartile = sorted(merged["pred_quartile"].unique())[0]
bottom = merged[merged["pred_quartile"] == bottom_quartile]
actual = bottom["y_test"].values

p_lt0 = np.mean(actual < 0)
p_eq0 = np.mean(actual == 0)
p_le0 = np.mean(actual <= 0)

print(f"bottom quartile: {bottom_quartile} (n={len(bottom)})")
print(f"P(actual < 0)  = {p_lt0:.4f}")
print(f"P(actual == 0) = {p_eq0:.4f}")
print(f"P(actual <= 0) = {p_le0:.4f}")

nominal = 0.10
in_range = p_lt0 <= nominal <= p_le0
print(f"\nnominal 10% in [P(actual<0), P(actual<=0)] = [{p_lt0:.4f}, {p_le0:.4f}]? {in_range}")
if in_range:
    print("-> q10=0 is a CORRECT 10th percentile by definition; the step-30 cell reading "
          "(21.28% observed vs 10% nominal) was a metric artifact of the strict '<=' rule "
          "colliding with a discrete point mass at 0, not a model miscalibration.")
else:
    print("-> nominal 10% does NOT fall inside the point-mass range; step-30 finding stands "
          "as a genuine miscalibration, not a metric artifact.")


# --- Tie-aware coverage: recompute the 20-cell table for all four architectures ---
section("20-cell table: original vs tie-aware deviations, all four architectures")

max_dev = {"pooled": 0.0, "cond_pred": 0.0, "cond_pos": 0.0, "qreg": 0.0}
max_dev_tieaware = {"pooled": 0.0, "cond_pred": 0.0, "cond_pos": 0.0, "qreg": 0.0}
changed_cells = []

for quartile, group in merged.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {quartile} (n={len(group)}):")
    print(f"  {'nominal':>8} {'arch':>10} {'orig_obs':>9} {'orig_dev':>9} {'tie_obs':>9} {'tie_dev':>9} {'delta':>7}")
    for level in COVERAGE_LEVELS:
        residual = group["residual"].values
        actual = group["y_test"].values

        # pooled
        thr = pooled_thresholds[level]
        orig = np.mean(residual <= thr)
        tie = np.mean(residual < thr) + 0.5 * np.mean(residual == thr)
        dev_o, dev_t = abs(orig - level), abs(tie - level)
        max_dev["pooled"] = max(max_dev["pooled"], dev_o)
        max_dev_tieaware["pooled"] = max(max_dev_tieaware["pooled"], dev_t)
        delta = abs(tie - orig)
        if delta > MATERIAL_CHANGE:
            changed_cells.append((quartile, level, "pooled", delta))
        print(f"  {level * 100:>7.0f}% {'pooled':>10} {orig * 100:>8.2f}% {dev_o * 100:>8.2f}pp "
              f"{tie * 100:>8.2f}% {dev_t * 100:>8.2f}pp {delta * 100:>6.2f}pp")

        # cond_pred
        thr = group["sigma_pred"].values * z_pred_coverage[level]
        orig = np.mean(residual <= thr)
        tie = np.mean(residual < thr) + 0.5 * np.mean(residual == thr)
        dev_o, dev_t = abs(orig - level), abs(tie - level)
        max_dev["cond_pred"] = max(max_dev["cond_pred"], dev_o)
        max_dev_tieaware["cond_pred"] = max(max_dev_tieaware["cond_pred"], dev_t)
        delta = abs(tie - orig)
        if delta > MATERIAL_CHANGE:
            changed_cells.append((quartile, level, "cond_pred", delta))
        print(f"  {'':>8} {'cond_pred':>10} {orig * 100:>8.2f}% {dev_o * 100:>8.2f}pp "
              f"{tie * 100:>8.2f}% {dev_t * 100:>8.2f}pp {delta * 100:>6.2f}pp")

        # cond_pos
        thr = group["sigma_pos"].values * z_pos_coverage[level]
        orig = np.mean(residual <= thr)
        tie = np.mean(residual < thr) + 0.5 * np.mean(residual == thr)
        dev_o, dev_t = abs(orig - level), abs(tie - level)
        max_dev["cond_pos"] = max(max_dev["cond_pos"], dev_o)
        max_dev_tieaware["cond_pos"] = max(max_dev_tieaware["cond_pos"], dev_t)
        delta = abs(tie - orig)
        if delta > MATERIAL_CHANGE:
            changed_cells.append((quartile, level, "cond_pos", delta))
        print(f"  {'':>8} {'cond_pos':>10} {orig * 100:>8.2f}% {dev_o * 100:>8.2f}pp "
              f"{tie * 100:>8.2f}% {dev_t * 100:>8.2f}pp {delta * 100:>6.2f}pp")

        # qreg
        thr = group[level_to_col[level]].values
        orig = np.mean(actual <= thr)
        tie = np.mean(actual < thr) + 0.5 * np.mean(actual == thr)
        dev_o, dev_t = abs(orig - level), abs(tie - level)
        max_dev["qreg"] = max(max_dev["qreg"], dev_o)
        max_dev_tieaware["qreg"] = max(max_dev_tieaware["qreg"], dev_t)
        delta = abs(tie - orig)
        if delta > MATERIAL_CHANGE:
            changed_cells.append((quartile, level, "qreg", delta))
        print(f"  {'':>8} {'qreg':>10} {orig * 100:>8.2f}% {dev_o * 100:>8.2f}pp "
              f"{tie * 100:>8.2f}% {dev_t * 100:>8.2f}pp {delta * 100:>6.2f}pp")


# --- Cells that changed materially ---
section("Cells that changed by more than 0.5pp under tie-aware coverage")

if changed_cells:
    for quartile, level, arch, delta in changed_cells:
        print(f"  quartile={quartile}, level={level * 100:.0f}%, arch={arch}, delta={delta * 100:.2f}pp")
else:
    print("  none")

non_qreg_changes = [c for c in changed_cells if c[2] != "qreg"]
if non_qreg_changes:
    print("\n*** WARNING: a sigma-architecture cell moved materially -- diagnosis may be wrong, STOP ***")
    for quartile, level, arch, delta in non_qreg_changes:
        print(f"  quartile={quartile}, level={level * 100:.0f}%, arch={arch}, delta={delta * 100:.2f}pp")
else:
    print("\nAll material changes are confined to quantile-reg cells, as expected if the "
          "point-mass diagnosis is correct.")


# --- Side-by-side max-|dev| summary, recomputed ---
section("Side-by-side: max |dev| across all four architectures, original vs tie-aware")

print(f"{'architecture':<28} {'orig max|dev|':>14} {'tieaware max|dev|':>18} {'verdict (tie-aware, 5pp bar)':>30}")
arch_names = {"pooled": "27 pooled", "cond_pred": "28 cond-pred", "cond_pos": "29 cond-pos", "qreg": "30 quantile-reg"}
for key, name in arch_names.items():
    verdict = "PASS" if max_dev_tieaware[key] <= ACCEPTANCE_BAR else "FAIL"
    print(f"{name:<28} {max_dev[key] * 100:>13.2f}pp {max_dev_tieaware[key] * 100:>17.2f}pp {verdict:>30}")
