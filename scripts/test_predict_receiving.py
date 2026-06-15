"""Hard-assertion tests for the receiving probability layer (src/models/predict_receiving.py).

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_predict_receiving.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.predict_receiving import QUANTILE_LEVELS, load_artifacts, predict_quantiles, prob_over, sigma_for


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


artifacts = load_artifacts()
print(f"z-pool: count={len(artifacts['z_pool'])}, std={np.std(artifacts['z_pool'], ddof=1):.4f}, "
      f"mean={np.mean(artifacts['z_pool']):.2e}")
print(f"positions in sigma table: {sorted(artifacts['sigma_bins'].keys())}")


# --- (a) prob_over monotonically increasing in pred, fixed line ---
section("(a) prob_over monotonically increasing in pred (fixed line)")
line = 50.0
preds = np.array([20.0, 30.0, 40.0, 50.0, 60.0])
probs = prob_over(artifacts, preds, "WR", line)

assert np.all(np.diff(probs) > 0), f"prob_over not monotonically increasing: {probs}"
print(f"PASS: prob_over(preds={preds}, WR, line={line}) = {probs}")


# --- (b) prob_over monotonically decreasing in line, fixed pred ---
section("(b) prob_over monotonically decreasing in line (fixed pred)")
pred = 50.0
lines = np.array([20.0, 35.0, 50.0, 65.0, 80.0])
probs = prob_over(artifacts, pred, "WR", lines)

assert np.all(np.diff(probs) < 0), f"prob_over not monotonically decreasing: {probs}"
print(f"PASS: prob_over(pred={pred}, WR, lines={lines}) = {probs}")


# --- (c) skew-aware center test ---
section("(c) skew-aware center: prob_over(pred,pred) in [0.41, 0.50], p50 quantile < pred")
pred = 40.0
p_center = prob_over(artifacts, pred, "WR", pred)
quantiles = predict_quantiles(artifacts, pred, "WR")
p50 = quantiles[QUANTILE_LEVELS.index(0.5)]

assert 0.41 <= p_center <= 0.50, f"prob_over(pred,pred) = {p_center} not in [0.41, 0.50]"
assert p50 < pred, f"p50 quantile {p50} not < pred {pred}"
print(f"PASS: prob_over({pred},{pred}) = {p_center:.4f} in [0.41, 0.50]; "
      f"p50 = {p50:.2f} < pred = {pred}")


# --- (d) quantiles strictly increasing ---
section("(d) predict_quantiles strictly increasing (no crossing)")
pred = 45.0
quantiles = predict_quantiles(artifacts, pred, "TE")

assert np.all(np.diff(quantiles) > 0), f"quantiles not strictly increasing: {quantiles}"
print(f"PASS: predict_quantiles({pred}, TE, levels={QUANTILE_LEVELS}) = {quantiles}")


# --- (e) sigma_for monotone non-decreasing within position; RB < WR at mid-range pred ---
section("(e) sigma_for monotone non-decreasing in pred within each position; "
        "RB sigma < WR sigma at same mid-range pred")

for pos in sorted(artifacts["sigma_bins"].keys()):
    medians, stds = artifacts["sigma_bins"][pos]
    grid = np.linspace(medians.min(), medians.max(), 25)
    sigmas = sigma_for(artifacts, pos, grid)
    assert np.all(np.diff(sigmas) >= 0), f"{pos} sigma_for not non-decreasing over grid: {sigmas}"
    print(f"  {pos}: sigma_for non-decreasing over pred in [{medians.min():.2f}, {medians.max():.2f}] "
          f"(sigma range [{sigmas[0]:.2f}, {sigmas[-1]:.2f}])")

mid_pred = 28.0
sigma_rb = sigma_for(artifacts, "RB", mid_pred)
sigma_wr = sigma_for(artifacts, "WR", mid_pred)
assert sigma_rb < sigma_wr, f"sigma_for(RB, {mid_pred})={sigma_rb:.2f} not < sigma_for(WR, {mid_pred})={sigma_wr:.2f}"
print(f"PASS: sigma_for(RB, {mid_pred}) = {sigma_rb:.2f} < sigma_for(WR, {mid_pred}) = {sigma_wr:.2f}")


# --- (f) WR pred=60: P(over 75.5) sanity band ---
section("(f) WR pred=60: P(over 75.5) in [0.25, 0.40] (sigma ~= 40)")
pred = 60.0
line = 75.5
sigma = sigma_for(artifacts, "WR", pred)
p = prob_over(artifacts, pred, "WR", line)

print(f"sigma_for(WR, {pred}) = {sigma:.2f}")
assert 0.25 <= p <= 0.40, f"prob_over(WR, pred={pred}, line={line}) = {p} not in [0.25, 0.40]"
print(f"PASS: prob_over(WR, pred={pred}, line={line}) = {p:.4f} in [0.25, 0.40]")


print("\nAll tests passed.")
