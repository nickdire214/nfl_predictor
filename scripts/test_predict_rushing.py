"""Hard-assertion tests for the rushing probability layer (src/models/predict_rushing.py).

Read-only verification. Exits nonzero if any assertion fails. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_predict_rushing.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.predict_rushing import QUANTILE_LEVELS, load_artifacts, predict_quantiles, prob_over
from src.models.train_rushing import FEATURE_COLS, preprocess_rushing_matrix


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


artifacts = load_artifacts()
print(f"loaded {len(artifacts['models'])} quantile models at levels {artifacts['levels']}")

# Real feature rows: a high-volume workhorse and a low-volume back (2025).
df = preprocess_rushing_matrix(pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet"))
df_2025 = df[df["season"] == 2025]

workhorse = df_2025.sort_values("carries_l8", ascending=False).iloc[[0]]
lowvol = df_2025[df_2025["carries_l8"] >= 8].sort_values("carries_l8").iloc[[0]]
print(f"workhorse: {workhorse.iloc[0]['player']} wk{workhorse.iloc[0]['week']} "
      f"(carries_l8={workhorse.iloc[0]['carries_l8']:.1f})")
print(f"low-volume: {lowvol.iloc[0]['player']} wk{lowvol.iloc[0]['week']} "
      f"(carries_l8={lowvol.iloc[0]['carries_l8']:.1f})")

# Lowest-PREDICTION row within the PROP SLICE (carries_l8 >= 8) for the floor test.
# (The layer is built for and evaluated on the prop slice; a near-zero-volume back
# outside it is out-of-distribution and the quantile models would extrapolate.)
prop_2025 = df_2025[df_2025["carries_l8"] >= 8].copy()
prop_2025["_q50"] = predict_quantiles(artifacts, prop_2025, levels=[0.50])
lowpred = prop_2025.sort_values("_q50").iloc[[0]]


# --- (a) prob_over monotone decreasing in line (fixed feature row) ---
section("(a) prob_over monotonically decreasing in line (fixed workhorse row)")
# Lines spanning the row's interior quantile range (outside [p5,p95] prob_over
# legitimately plateaus on the 0.01/0.99 clamps -- that is checked in (c)).
qw = predict_quantiles(artifacts, workhorse)
lines = np.linspace(qw[0] + 2, qw[-1] - 2, 6)
probs = prob_over(artifacts, workhorse, lines)
assert np.all(np.diff(probs) < 0), f"prob_over not monotonically decreasing: {probs}"
print(f"PASS: lines={np.round(lines, 1)} -> prob_over={np.round(probs, 4)}")


# --- (b) predict_quantiles strictly increasing after sort ---
section("(b) predict_quantiles strictly increasing (monotone on output)")
q = predict_quantiles(artifacts, workhorse)
assert np.all(np.diff(q) > 0), f"quantiles not strictly increasing: {q}"
print(f"PASS: levels={QUANTILE_LEVELS} -> quantiles={np.round(q, 2)}")


# --- (c) prob_over within [0.01, 0.99] ---
section("(c) prob_over within [0.01, 0.99] (incl. extreme lines)")
extreme = prob_over(artifacts, workhorse, np.array([-50.0, 0.0, 75.0, 500.0]))
assert np.all((extreme >= 0.01) & (extreme <= 0.99)), f"prob_over out of [0.01,0.99]: {extreme}"
print(f"PASS: extreme lines [-50,0,75,500] -> prob_over={np.round(extreme, 4)} (all in [0.01,0.99])")


# --- (d) workhorse q50 > low-volume q50 ---
section("(d) workhorse q50 > low-volume q50")
q50_workhorse = predict_quantiles(artifacts, workhorse, levels=[0.50])[0]
q50_lowvol = predict_quantiles(artifacts, lowvol, levels=[0.50])[0]
assert q50_workhorse > q50_lowvol, f"workhorse q50 ({q50_workhorse}) not > low-volume q50 ({q50_lowvol})"
print(f"PASS: workhorse q50={q50_workhorse:.2f} > low-volume q50={q50_lowvol:.2f}")


# --- (e) floor sanity: a low-prediction row's q10 >= 0 ---
section("(e) floor sanity: low-prediction row q10 >= 0 (the thing cond-sigma failed)")
q10_lowpred = predict_quantiles(artifacts, lowpred, levels=[0.10])[0]
q50_lowpred = predict_quantiles(artifacts, lowpred, levels=[0.50])[0]
assert q10_lowpred >= 0, f"low-prediction q10 ({q10_lowpred}) is negative"
print(f"PASS: lowest-prediction row ({lowpred.iloc[0]['player']} wk{lowpred.iloc[0]['week']}, "
      f"q50={q50_lowpred:.2f}) has q10={q10_lowpred:.2f} >= 0")


print("\nAll tests passed.")
