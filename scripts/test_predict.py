"""Hard-assertion tests for the probability layer (src/models/predict.py).

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_predict.py
"""

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.predict import load_residual_pool, predict_quantiles, prob_over


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


residual_pool = load_residual_pool()
print(f"Residual pool: count={len(residual_pool)}, std={np.std(residual_pool, ddof=1):.4f}")


# --- (a) prob_over monotonically increasing in the prediction, fixed line ---
section("(a) prob_over monotonically increasing in prediction (fixed line)")
line = 250.0
preds = np.array([200.0, 225.0, 250.0, 275.0, 300.0])
probs = prob_over(preds, line, residual_pool)

assert np.all(np.diff(probs) > 0), f"prob_over not monotonically increasing: {probs}"
print(f"PASS: prob_over({preds}, line={line}) = {probs} is monotonically increasing")


# --- (b) prob_over(pred, line=pred) ~ 0.50 ---
section("(b) prob_over(pred, line=pred) within 0.03 of 0.50")
pred = 230.0
p = prob_over(pred, pred, residual_pool)

assert abs(p - 0.50) <= 0.03, f"prob_over(pred, line=pred) = {p} not within 0.03 of 0.50"
print(f"PASS: prob_over({pred}, line={pred}) = {p:.4f} is within 0.03 of 0.50")


# --- (c) prob_over monotonically decreasing in the line, fixed prediction ---
section("(c) prob_over monotonically decreasing in line (fixed prediction)")
pred = 250.0
lines = np.array([200.0, 225.0, 250.0, 275.0, 300.0])
probs = prob_over(pred, lines, residual_pool)

assert np.all(np.diff(probs) < 0), f"prob_over not monotonically decreasing: {probs}"
print(f"PASS: prob_over(pred={pred}, {lines}) = {probs} is monotonically decreasing")


# --- (d) quantiles strictly increasing (no crossing) ---
section("(d) predict_quantiles strictly increasing (no crossing)")
pred = 250.0
levels = (0.1, 0.25, 0.5, 0.75, 0.9)
quantiles = predict_quantiles(pred, residual_pool, levels=levels)

assert np.all(np.diff(quantiles) > 0), f"quantiles not strictly increasing: {quantiles}"
print(f"PASS: predict_quantiles({pred}, levels={levels}) = {quantiles} is strictly increasing")


# --- (e) sanity probability for a line 75 yards above the prediction ---
section("(e) P(over) for line = pred + 75, given residual std ~74")
pred = 250.0
line = pred + 75.0
p = prob_over(pred, line, residual_pool)

assert 0.13 <= p <= 0.20, f"P(over line=pred+75) = {p} not in [0.13, 0.20]"
print(f"PASS: prob_over(pred={pred}, line={line}) = {p:.4f} is in [0.13, 0.20]")


print("\nAll tests passed.")
