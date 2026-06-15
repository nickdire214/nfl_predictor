"""Step 32: finalist tiebreaker on product-realistic probability quality (cond-pos vs quantile-reg).

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\tiebreak_prob_quality.py
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
N_QUINTILES = 5
RANDOM_STATE = 42
LINE_FLOOR_SUBSET = 25.5
BRIER_TIE_THRESHOLD = 0.002


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


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

    records.append(test[[
        "analysis_season", "pred", "residual", "y_test", "targets_l8", "position",
        "week", "gsis_id", "receiving_yards_l8",
    ]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["targets_l8"] >= PROP_MIN_TARGETS_L8].copy()

train_data = prop_data[prop_data["analysis_season"] < 2025].copy()
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy()


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


train_data["sigma_pos"] = sigma_position_vec(train_data["pred"].values, train_data["position"].values)
train_data["z_pos"] = train_data["residual"] / train_data["sigma_pos"]
z_pos_pool = np.sort(train_data["z_pos"].values)
n_pool = len(z_pos_pool)

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


# --- Step 32: synthetic line ---
section("Step 32: synthetic line = round(receiving_yards_l8) + 0.5")

merged["line"] = np.round(merged["receiving_yards_l8"].values) + 0.5

print(f"line distribution (n={len(merged)}):")
print(f"  min:    {merged['line'].min():.1f}")
print(f"  median: {merged['line'].median():.1f}")
print(f"  max:    {merged['line'].max():.1f}")


# --- P(over) for cond-pos ---
z_line = (merged["line"].values - merged["pred"].values) / merged["sigma_pos"].values
# P(z_pool > z_line) via empirical survival function (searchsorted on sorted pool)
idx = np.searchsorted(z_pos_pool, z_line, side="right")
p_over_pos = (n_pool - idx) / n_pool

# --- P(over) for quantile-reg: piecewise-linear interpolation through 7 sorted quantiles ---
levels_arr = np.array(QUANTILE_LEVELS)
q_values = merged[q_cols].values  # (n, 7), sorted ascending per row

p_over_qreg = np.empty(len(merged))
lines = merged["line"].values
for i in range(len(merged)):
    qvals = q_values[i]
    line = lines[i]
    if line < qvals[0]:
        cdf = 0.01
    elif line > qvals[-1]:
        cdf = 0.99
    else:
        cdf = np.interp(line, qvals, levels_arr)
    p_over_qreg[i] = 1.0 - cdf

merged["p_over_pos"] = p_over_pos
merged["p_over_qreg"] = p_over_qreg
merged["over"] = (merged["y_test"].values > lines).astype(int)


# --- Brier scores ---
section("Brier scores: overall, per prediction-quartile, and line >= 25.5 subset")

brier_pos_overall = np.mean((merged["p_over_pos"].values - merged["over"].values) ** 2)
brier_qreg_overall = np.mean((merged["p_over_qreg"].values - merged["over"].values) ** 2)

print(f"{'subset':<28} {'n':>6} {'brier_cond_pos':>15} {'brier_qreg':>12}")
print(f"{'overall':<28} {len(merged):>6} {brier_pos_overall:>15.5f} {brier_qreg_overall:>12.5f}")

for quartile, group in merged.groupby("pred_quartile", observed=True):
    b_pos = np.mean((group["p_over_pos"].values - group["over"].values) ** 2)
    b_qreg = np.mean((group["p_over_qreg"].values - group["over"].values) ** 2)
    print(f"{str(quartile):<28} {len(group):>6} {b_pos:>15.5f} {b_qreg:>12.5f}")

subset = merged[merged["line"] >= LINE_FLOOR_SUBSET]
brier_pos_subset = np.mean((subset["p_over_pos"].values - subset["over"].values) ** 2)
brier_qreg_subset = np.mean((subset["p_over_qreg"].values - subset["over"].values) ** 2)
print(f"\n{'line >= 25.5':<28} {len(subset):>6} {brier_pos_subset:>15.5f} {brier_qreg_subset:>12.5f}")


# --- Reliability tables (10 bins) ---
for name, col in [("cond-pos", "p_over_pos"), ("quantile-reg", "p_over_qreg")]:
    section(f"Reliability table (10 bins): {name}")
    bins = np.linspace(0, 1, 11)
    merged["_bin"] = pd.cut(merged[col], bins=bins, include_lowest=True)
    print(f"{'bin':<16} {'n':>6} {'mean_pred':>10} {'observed_over_rate':>20}")
    for b, group in merged.groupby("_bin", observed=True):
        if len(group) == 0:
            continue
        mean_pred = group[col].mean()
        obs_rate = group["over"].mean()
        print(f"{str(b):<16} {len(group):>6} {mean_pred:>10.4f} {obs_rate:>20.4f}")
    merged.drop(columns="_bin", inplace=True)


# --- Decision rule ---
section("Decision rule and verdict")

print(f"overall Brier:    cond-pos={brier_pos_overall:.5f}  quantile-reg={brier_qreg_overall:.5f}")
print(f"line>=25.5 Brier: cond-pos={brier_pos_subset:.5f}  quantile-reg={brier_qreg_subset:.5f}")

diff_overall = brier_pos_overall - brier_qreg_overall

if abs(diff_overall) > BRIER_TIE_THRESHOLD:
    winner = "quantile-reg" if brier_qreg_overall < brier_pos_overall else "cond-pos"
    print(f"\noverall Brier difference ({diff_overall:.5f}) exceeds tie threshold ({BRIER_TIE_THRESHOLD}) "
          f"-> winner decided on overall Brier: {winner}")
else:
    diff_subset = brier_pos_subset - brier_qreg_subset
    print(f"\noverall Brier difference ({diff_overall:.5f}) within tie threshold ({BRIER_TIE_THRESHOLD}) "
          f"-> tiebreak on line>=25.5 Brier")
    if abs(diff_subset) > 1e-9:
        winner = "quantile-reg" if brier_qreg_subset < brier_pos_subset else "cond-pos"
        print(f"line>=25.5 Brier difference ({diff_subset:.5f}) -> winner: {winner}")
    else:
        winner = "cond-pos"
        print(f"line>=25.5 Brier tied exactly -> winner on simplicity: {winner}")

print(f"\nVERDICT: {winner}")
