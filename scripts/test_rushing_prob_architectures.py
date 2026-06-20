"""Step 46: rushing probability architecture head-to-head, tie-aware from the start.

Architecture A: prediction-conditional sigma (decile bins, no position axis -- RB only).
Architecture B: quantile regression (7 HistGradientBoostingRegressor models).

Both evaluated on the 2025 prop-slice (carries_l8 >= 8) under TIE-AWARE coverage
(observed = mean(actual < thr) + 0.5*mean(actual == thr)) across a 20-cell table
(pred-quartile x [10,25,50,75,90]). Pre-registered bar: max |dev| <= 5pp.

Read-only, no production code. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_rushing_prob_architectures.py
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
from src.models.train_rushing import FEATURE_COLS, TARGET, preprocess_rushing_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150
PROP_MIN_CARRIES_L8 = 8
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
COVERAGE_LEVELS = [0.10, 0.25, 0.50, 0.75, 0.90]
N_DECILES = 10
ACCEPTANCE_BAR = 0.05
RANDOM_STATE = 42
LINE_FLOOR_SUBSET = 40.5  # workhorse-RB rushing-prop territory (analogue of receiving's 25.5)
BRIER_TIE_THRESHOLD = 0.002


def section(title):
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def tie_aware(actual, threshold):
    return np.mean(actual < threshold) + 0.5 * np.mean(actual == threshold)


df = pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet")
df_processed = preprocess_rushing_matrix(df)


# --- Calibrated walk-forward, collect OOS (pred, residual, carries_l8) on prop slice ---
records = []
for season in SEASONS:
    train = df_processed[df_processed["season"] < season]
    test = df_processed[df_processed["season"] == season].sort_values("week").reset_index(drop=True)

    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])
    pipeline.fit(train[FEATURE_COLS], train[TARGET])

    test = test.copy()
    test["raw_pred"] = pipeline.predict(test[FEATURE_COLS])
    test["y_test"] = test[TARGET].values

    calibrator = RollingCalibrator(n0=N0)
    calibrated = np.zeros(len(test))
    for week in sorted(test["week"].unique()):
        m = (test["week"] == week).values
        calibrated[m] = calibrator.calibrate(test.loc[m, "raw_pred"].values)
        calibrator.update(test.loc[m, "raw_pred"].values, test.loc[m, "y_test"].values)

    test["pred"] = calibrated
    test["residual"] = test["y_test"] - test["pred"]
    test["analysis_season"] = season
    records.append(test[[
        "analysis_season", "pred", "residual", "y_test", "carries_l8",
        "week", "gsis_id", "rushing_yards_l8",
    ]])

all_data = pd.concat(records, ignore_index=True)
prop_data = all_data[all_data["carries_l8"] >= PROP_MIN_CARRIES_L8].copy()
train_data = prop_data[prop_data["analysis_season"] < 2025].copy()
test_2025 = prop_data[prop_data["analysis_season"] == 2025].copy().reset_index(drop=True)

print(f"prop-slice pool: train(22-24)={len(train_data)}, test(2025)={len(test_2025)}")


# --- Architecture A: prediction-conditional sigma (decile bins, no position) ---
train_data["decile"] = pd.qcut(train_data["pred"], N_DECILES, duplicates="drop")
decile_stats = train_data.groupby("decile", observed=True).agg(
    n=("pred", "size"), median_pred=("pred", "median"), std=("residual", "std"),
).reset_index()
pred_bin_medians = decile_stats["median_pred"].values
pred_bin_stds = decile_stats["std"].values


def sigma_pred(pred):
    return np.interp(pred, pred_bin_medians, pred_bin_stds)


train_data["sigma_pred"] = sigma_pred(train_data["pred"].values)
z_pool = np.sort((train_data["residual"] / train_data["sigma_pred"]).values)
n_pool = len(z_pool)
z_q = {lvl: np.quantile(z_pool, lvl) for lvl in QUANTILE_LEVELS}

test_2025["sigma_pred"] = sigma_pred(test_2025["pred"].values)
test_2025["pred_quartile"] = pd.qcut(test_2025["pred"], 4, duplicates="drop")


# --- Architecture B: quantile regression (train 2021-2024 prop-slice) ---
section("Architecture B: HistGradientBoostingRegressor quantile models")
train_qr_full = df_processed[df_processed["season"] <= 2024]
train_qr = train_qr_full[train_qr_full["carries_l8"] >= PROP_MIN_CARRIES_L8]
test_qr_full = df_processed[df_processed["season"] == 2025].sort_values("week").reset_index(drop=True)

q_cols = []
for q in QUANTILE_LEVELS:
    model = HistGradientBoostingRegressor(loss="quantile", quantile=q, max_depth=3, random_state=RANDOM_STATE)
    model.fit(train_qr[FEATURE_COLS], train_qr[TARGET])
    col = f"q{int(round(q * 100)):02d}"
    test_qr_full[col] = model.predict(test_qr_full[FEATURE_COLS])
    q_cols.append(col)

print(f"Train rows: {len(train_qr)} (2021-2024 prop-slice), Test rows (full 2025): {len(test_qr_full)}")

raw_q = test_qr_full[q_cols].values
crossing_rate = np.mean(np.any(np.diff(raw_q, axis=1) < 0, axis=1))
sorted_q = np.sort(raw_q, axis=1)
for i, col in enumerate(q_cols):
    test_qr_full[col] = sorted_q[:, i]
print(f"Crossing rate (rows needing a sort): {crossing_rate:.4%}")

test_2025_qr = test_qr_full[test_qr_full["carries_l8"] >= PROP_MIN_CARRIES_L8].reset_index(drop=True)
merged = test_2025.merge(
    test_2025_qr[["gsis_id", "week"] + q_cols], on=["gsis_id", "week"], how="inner",
)
assert len(merged) == len(test_2025), "merge dropped rows -- gsis_id/week not a unique key"
merged["pred_quartile"] = test_2025["pred_quartile"].values
level_to_col = {lvl: f"q{int(round(lvl * 100)):02d}" for lvl in COVERAGE_LEVELS}


# --- Context: q50 MAE vs calibrated-Ridge MAE ---
section("Context: q50 MAE vs calibrated-Ridge MAE (2025 prop-slice)")
actual = merged["y_test"].values
ridge_mae = _mae(merged["pred"].values, actual)
a_q50 = merged["pred"].values + merged["sigma_pred"].values * z_q[0.50]
b_q50 = merged["q50"].values
print(f"calibrated-Ridge point MAE: {ridge_mae:.2f}")
print(f"Arch A (cond-sigma) q50 MAE: {_mae(a_q50, actual):.2f}")
print(f"Arch B (quantile-reg) q50 MAE: {_mae(b_q50, actual):.2f}")


# --- Context: p10/p90 by predicted-value quartile ---
section("p10 / p90 by predicted-value quartile (bottom-quartile p10 should be >= ~0)")
print(f"{'quartile':<22} {'median_pred':>11} {'A_p10':>8} {'A_p90':>8} {'B_p10':>8} {'B_p90':>8}")
bottom_a_p10 = None
for i, (qt, g) in enumerate(merged.groupby("pred_quartile", observed=True)):
    mp = g["pred"].median()
    a_p10 = mp + sigma_pred(mp) * z_q[0.10]
    a_p90 = mp + sigma_pred(mp) * z_q[0.90]
    b_p10 = g["q10"].median()
    b_p90 = g["q90"].median()
    if i == 0:
        bottom_a_p10 = a_p10
    print(f"{str(qt):<22} {mp:>11.2f} {a_p10:>8.2f} {a_p90:>8.2f} {b_p10:>8.2f} {b_p90:>8.2f}")
print(f"\nBottom-quartile Arch-A p10 = {bottom_a_p10:.2f} "
      f"({'OK, >= ~0' if bottom_a_p10 >= -2 else 'FLAG: below zero floor'})")


# --- Tie-aware 20-cell coverage table ---
section("Tie-aware coverage (20 cells: pred-quartile x [10,25,50,75,90]), A vs B")
max_dev = {"A_cond_sigma": 0.0, "B_quantile_reg": 0.0}
for qt, g in merged.groupby("pred_quartile", observed=True):
    print(f"\n  quartile {qt} (n={len(g)}):")
    print(f"  {'nominal':>8} {'A_obs':>8} {'A_dev':>8} {'B_obs':>8} {'B_dev':>8}")
    g_actual = g["y_test"].values
    for lvl in COVERAGE_LEVELS:
        # Arch-A threshold is the per-row predicted quantile (pred + sigma(pred)*z_q).
        a_thr = g["pred"].values + g["sigma_pred"].values * z_q[lvl]
        a_obs = np.mean(g_actual < a_thr) + 0.5 * np.mean(g_actual == a_thr)
        a_dev = abs(a_obs - lvl)
        max_dev["A_cond_sigma"] = max(max_dev["A_cond_sigma"], a_dev)

        b_thr = g[level_to_col[lvl]].values
        b_obs = np.mean(g_actual < b_thr) + 0.5 * np.mean(g_actual == b_thr)
        b_dev = abs(b_obs - lvl)
        max_dev["B_quantile_reg"] = max(max_dev["B_quantile_reg"], b_dev)

        print(f"  {lvl * 100:>7.0f}% {a_obs * 100:>7.2f}% {a_dev * 100:>6.2f}pp "
              f"{b_obs * 100:>7.2f}% {b_dev * 100:>6.2f}pp")


# --- Verdict vs the 5pp bar ---
section("Verdict vs pre-registered 5pp bar (max |dev| across 20 cells)")
results = {}
for arch, md in max_dev.items():
    verdict = "PASS" if md <= ACCEPTANCE_BAR else "FAIL"
    results[arch] = verdict
    print(f"  {arch:<18} max|dev| = {md * 100:.2f}pp -> {verdict}")

passers = [a for a, v in results.items() if v == "PASS"]


# --- Tiebreak / floor rule ---
section("Decision")
if len(passers) == 2:
    print("Both PASS -> tiebreak on Brier at synthetic lines (round(rushing_yards_l8)+0.5)")
    lines = np.round(merged["rushing_yards_l8"].values) + 0.5
    over = (merged["y_test"].values > lines).astype(int)
    print(f"  synthetic line: min={lines.min():.1f}, median={np.median(lines):.1f}, max={lines.max():.1f}")

    # Arch A P(over) via empirical z-pool survival
    z_line = (lines - merged["pred"].values) / merged["sigma_pred"].values
    idx = np.searchsorted(z_pool, z_line, side="right")
    p_over_a = (n_pool - idx) / n_pool

    # Arch B P(over) via interpolation through 7 sorted quantiles
    levels_arr = np.array(QUANTILE_LEVELS)
    qv = merged[q_cols].values
    p_over_b = np.empty(len(merged))
    for i in range(len(merged)):
        if lines[i] < qv[i][0]:
            cdf = 0.01
        elif lines[i] > qv[i][-1]:
            cdf = 0.99
        else:
            cdf = np.interp(lines[i], qv[i], levels_arr)
        p_over_b[i] = 1.0 - cdf

    brier_a = np.mean((p_over_a - over) ** 2)
    brier_b = np.mean((p_over_b - over) ** 2)
    sub = merged["rushing_yards_l8"].notna() & (lines >= LINE_FLOOR_SUBSET)
    brier_a_sub = np.mean((p_over_a[sub.values] - over[sub.values]) ** 2)
    brier_b_sub = np.mean((p_over_b[sub.values] - over[sub.values]) ** 2)

    print(f"  overall Brier:        A={brier_a:.5f}  B={brier_b:.5f}")
    print(f"  line>={LINE_FLOOR_SUBSET} Brier:  A={brier_a_sub:.5f}  B={brier_b_sub:.5f} (n={int(sub.sum())})")
    if abs(brier_a - brier_b) > BRIER_TIE_THRESHOLD:
        winner = "A_cond_sigma" if brier_a < brier_b else "B_quantile_reg"
        print(f"  -> overall Brier decides: {winner}")
    else:
        winner = "A_cond_sigma" if brier_a_sub < brier_b_sub else "B_quantile_reg"
        print(f"  -> within tie threshold; line-floor subset decides: {winner}")
    print(f"\nVERDICT: {winner} (both passed the 5pp bar)")
elif len(passers) == 1:
    print(f"VERDICT: {passers[0]} wins (sole architecture passing the 5pp bar)")
else:
    best = min(max_dev, key=max_dev.get)
    print("Neither passes the 5pp bar -> registered floor rule: best-by-max-dev ships with documented blemish")
    print(f"VERDICT: {best} (max|dev| {max_dev[best] * 100:.2f}pp) ships with documented blemish")
