"""Step 47 (part 1): build the rushing probability-layer artifacts.

Trains the 7 HistGradientBoostingRegressor(loss="quantile") models (the
step-46 winner) on ALL prop-relevant rows (carries_l8 >= 8) across all
seasons 2021-2025, on FEATURE_COLS / target rushing_yards, and saves them
as a dict {q: model}.

Unlike receiving (sigma table + z-pool), the rushing layer ships the fitted
quantile models directly -- the predictive distribution is the 7 quantiles,
sorted on output.

Saves:
    models/rushing_quantile_models.pkl  ({quantile_level: fitted model})

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\build_rushing_prob_artifacts.py
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.train_rushing import FEATURE_COLS, TARGET, preprocess_rushing_matrix

PROP_MIN_CARRIES_L8 = 8
QUANTILE_LEVELS = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
RANDOM_STATE = 42
MODEL_PATH = MODELS_DIR / "rushing_quantile_models.pkl"


df = pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet")
df_processed = preprocess_rushing_matrix(df)
prop = df_processed[df_processed["carries_l8"] >= PROP_MIN_CARRIES_L8].copy()

X, y = prop[FEATURE_COLS], prop[TARGET].values
print(f"Training rows (prop-slice, carries_l8 >= {PROP_MIN_CARRIES_L8}, seasons "
      f"{prop['season'].min()}-{prop['season'].max()}): {len(prop)}")

models = {}
q_cols = []
for q in QUANTILE_LEVELS:
    model = HistGradientBoostingRegressor(
        loss="quantile", quantile=q, max_depth=3, random_state=RANDOM_STATE,
    )
    model.fit(X, y)
    models[q] = model
    col = f"q{int(round(q * 100)):02d}"
    prop[col] = model.predict(X)
    q_cols.append(col)


# --- Crossing diagnostics on the training rows: rate AND magnitude ---
raw_q = prop[q_cols].values
diffs = np.diff(raw_q, axis=1)               # adjacent (q[i+1]-q[i]); negative => crossing
crossing_rows = np.any(diffs < 0, axis=1)
crossing_rate = np.mean(crossing_rows)
inversions = -diffs[diffs < 0]               # positive yards by which a pair inverts
crossing_magnitude = inversions.mean() if inversions.size else 0.0

print(f"\nCrossing diagnostics (training rows, n={len(prop)}):")
print(f"  crossing rate (rows needing a sort): {crossing_rate:.4%}")
print(f"  crossing magnitude (mean yards an inverted adjacent pair crosses by): {crossing_magnitude:.3f}")
print(f"  max single inversion: {inversions.max():.2f} yards" if inversions.size else "  (no inversions)")


# --- Per-quantile training coverage sanity (on sorted output) ---
sorted_q = np.sort(raw_q, axis=1)
print("\nPer-quantile training coverage (sorted output, P(actual <= q-pred)):")
print(f"  {'level':>6} {'nominal':>8} {'observed':>9}")
for i, q in enumerate(QUANTILE_LEVELS):
    observed = np.mean(y <= sorted_q[:, i])
    print(f"  {q * 100:>5.0f}% {q * 100:>7.0f}% {observed * 100:>8.2f}%")


# --- Save ---
MODELS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(models, MODEL_PATH)
print(f"\nSaved {len(models)} quantile models to: {MODEL_PATH}")
