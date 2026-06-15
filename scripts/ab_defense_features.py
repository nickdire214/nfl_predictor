"""Step 23: A/B verdict on opponent pass-defense features (DEFENSE_COLS).

Read-only evidence-gathering, no pipeline changes. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\ab_defense_features.py

Decision rule (pre-registered): adopt DEFENSE_COLS if arm B's mean MAE
improves at all (mean delta < 0) AND no season degrades by more than 0.3
MAE. Otherwise reject.
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
from src.features.engineer import DEFENSE_COLS
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix

SEASONS = [2022, 2023, 2024, 2025]
N0 = 150

ARM_A_COLS = FEATURE_COLS
ARM_B_COLS = FEATURE_COLS + DEFENSE_COLS


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _bias(pred, actual):
    return np.mean(pred - actual)


def calibrated_predictions(test, n0=N0):
    weeks = sorted(test["week"].unique())
    calibrated = np.zeros(len(test))

    for week in weeks:
        week_mask = (test["week"] == week).values
        prior_mask = (test["week"] < week).values
        n = prior_mask.sum()

        if n > 0:
            mean_bias = test.loc[prior_mask, "error"].mean()
        else:
            mean_bias = 0.0

        offset = mean_bias * (n / (n + n0))
        calibrated[week_mask] = test.loc[week_mask, "raw_pred"] - offset

    return calibrated


def run_arm(df, cols, season):
    train = df[df["season"] < season]
    test = df[df["season"] == season].sort_values("week").reset_index(drop=True)

    X_train, y_train = train[cols], train[TARGET]
    X_test, y_test = test[cols], test[TARGET]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    pipeline.fit(X_train, y_train)

    raw_pred = pipeline.predict(X_test)
    test = test.copy()
    test["raw_pred"] = raw_pred
    test["y_test"] = y_test.values
    test["error"] = test["raw_pred"] - test["y_test"]

    cal_pred = calibrated_predictions(test)
    mae = _mae(cal_pred, test["y_test"].values)
    bias = _bias(cal_pred, test["y_test"].values)
    return mae, bias


df = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
df = preprocess_qb_matrix(df)

before_rows = len(df)
for stat in ("def_pass_yards_allowed", "def_pass_epa_allowed"):
    df[f"{stat}_std"] = df[f"{stat}_std"].fillna(df[f"{stat}_l8"])

df = df.dropna(subset=["def_pass_yards_allowed_l8"]).copy()
dropped = before_rows - len(df)
print(f"Rows dropped for missing def_pass_yards_allowed_l8: {dropped} (before={before_rows}, after={len(df)})")


print("\nPer-season results:")
print(f"{'season':<8} {'MAE_A':>8} {'bias_A':>8} {'MAE_B':>8} {'bias_B':>8} {'delta_MAE':>10}")

deltas = []
for season in SEASONS:
    mae_a, bias_a = run_arm(df, ARM_A_COLS, season)
    mae_b, bias_b = run_arm(df, ARM_B_COLS, season)
    delta = mae_b - mae_a
    deltas.append(delta)
    print(f"{season:<8} {mae_a:>8.2f} {bias_a:>8.2f} {mae_b:>8.2f} {bias_b:>8.2f} {delta:>10.3f}")

mean_delta = np.mean(deltas)
worst_delta = max(deltas)

print("\nSummary:")
print(f"Mean MAE delta (B - A) across {len(SEASONS)} seasons: {mean_delta:.4f}")
print(f"Worst single-season degradation for B (max delta): {worst_delta:.4f}")

if mean_delta < 0 and worst_delta <= 0.3:
    print("\nVerdict: ADOPT (mean MAE improves and no season degrades by more than 0.3)")
else:
    print("\nVerdict: REJECT (decision rule not met)")


# --- Fit arm B once on all data, inspect standardized defense coefficients ---
print("\n" + "=" * 60)
print("Arm B fit on all data: standardized defense coefficients")
print("=" * 60)

X_all, y_all = df[ARM_B_COLS], df[TARGET]
pipeline_b = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0)),
])
pipeline_b.fit(X_all, y_all)

coefs = pipeline_b.named_steps["ridge"].coef_
coef_df = pd.DataFrame({"feature": ARM_B_COLS, "coef": coefs})
defense_coef_df = coef_df[coef_df["feature"].isin(DEFENSE_COLS)]
print(defense_coef_df.to_string(index=False))
