"""Step 26: walk-forward evaluation for the receiving-yards Ridge model.

Read-only evidence-gathering, no saved model. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\walkforward_receiving.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.train_receiving import FEATURE_COLS, TARGET, preprocess_receiving_matrix

SEASONS = [2022, 2023, 2024, 2025]
PROP_RELEVANT_MIN_TARGETS_L8 = 3


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


df = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
df = preprocess_receiving_matrix(df)

final_pipeline = None
final_coefs = None

for season in SEASONS:
    train = df[df["season"] < season]
    test = df[df["season"] == season].sort_values("week").reset_index(drop=True)

    X_train, y_train = train[FEATURE_COLS], train[TARGET]
    X_test, y_test = test[FEATURE_COLS], test[TARGET]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    ridge_pred = pipeline.predict(X_test)
    naive_l8_pred = test["receiving_yards_l8"].values
    naive_season_pred = test["receiving_yards_std"].values
    y_test_vals = y_test.values

    prop_mask = (test["targets_l8"] >= PROP_RELEVANT_MIN_TARGETS_L8).values
    n_prop = prop_mask.sum()

    print("\n" + "=" * 60)
    print(f"Test season {season} (train rows: {len(train)}, test rows: {len(test)}, "
          f"prop-relevant rows: {n_prop} ({n_prop / len(test):.1%}))")
    print("=" * 60)

    print(f"{'predictor':<15} {'slice':<14} {'MAE':>8} {'bias':>8}")
    for name, pred in [
        ("naive-L8", naive_l8_pred),
        ("naive-season", naive_season_pred),
        ("ridge", ridge_pred),
    ]:
        mae_all = _mae(pred, y_test_vals)
        bias_all = _bias(pred, y_test_vals)
        print(f"{name:<15} {'all':<14} {mae_all:>8.2f} {bias_all:>8.2f}")

        mae_prop = _mae(pred[prop_mask], y_test_vals[prop_mask])
        bias_prop = _bias(pred[prop_mask], y_test_vals[prop_mask])
        print(f"{name:<15} {'prop-relevant':<14} {mae_prop:>8.2f} {bias_prop:>8.2f}")

    if season == SEASONS[-1]:
        final_pipeline = pipeline


coefs = final_pipeline.named_steps["ridge"].coef_
coef_df = pd.DataFrame({"feature": FEATURE_COLS, "coef": coefs})
coef_df["abs_coef"] = coef_df["coef"].abs()
coef_df = coef_df.sort_values("abs_coef", ascending=False).head(15)

print("\n" + "=" * 60)
print(f"Top 15 Ridge coefficients by absolute value (final fold, test season {SEASONS[-1]})")
print("=" * 60)
print(coef_df[["feature", "coef"]].to_string(index=False))
