"""Probability layer for the RB rushing-yards model.

Implements the quantile-regression architecture adopted in DECISIONS.md
(steps 45-46): seven HistGradientBoostingRegressor(loss="quantile") models
(see scripts/build_rushing_prob_artifacts.py) supply the predictive
distribution directly as 7 quantiles, sorted on output to guarantee
monotonicity. prob_over interpolates the CDF piecewise-linearly through the
sorted quantiles.

INTERFACE NOTE (differs from receiving): this layer takes the FULL feature
row (all FEATURE_COLS) -- the quantile models need every feature to predict.
Receiving's layer needed only the scalar prediction (sigma was a function of
the prediction alone). Callers must pass a DataFrame/row carrying FEATURE_COLS,
not a precomputed point prediction.

Why quantile-reg and not sigma+z-pool (the receiving choice): rushing has a
large near-zero mass (low-volume backs), and a symmetric z-pool scaled by
sigma(pred) produces negative-yardage lower tails for low-prediction rows.
Quantile regression learns the zero-floor directly. Neither architecture
cleared the 5pp tie-aware coverage bar; quantile-reg shipped per the floor
rule (6.29pp vs cond-sigma's 8.83pp). Documented blemish: bottom predicted-
value-quartile p50 over-coverage; out-of-sample crossing rate ~25.8% on the
2025 test (handled by the output sort). Tracked in live grading.
"""

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.rushing import build_rushing_prediction_features
from src.ingestion._common import PROJECT_ROOT
from src.models.calibrate import RollingCalibrator
from src.models.train_rushing import FEATURE_COLS, TARGET, preprocess_rushing_matrix

MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
QUANTILE_MODELS_PATH = MODELS_DIR / "rushing_quantile_models.pkl"

N0 = 150
QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
CDF_CLAMP_LOW = 0.01
CDF_CLAMP_HIGH = 0.99


def load_artifacts():
    """Load the fitted rushing quantile models.

    Returns a dict with keys:
        "models": dict {quantile_level: fitted HistGradientBoostingRegressor}
        "levels": tuple of quantile levels in ascending order
    """
    models = joblib.load(QUANTILE_MODELS_PATH)
    levels = tuple(sorted(models.keys()))
    return {"models": models, "levels": levels}


def predict_quantiles(artifacts, features, levels=None):
    """Predict rushing-yards quantiles for each feature row, sorted per row.

    `features` is a DataFrame carrying FEATURE_COLS (the full feature row(s) --
    see the module interface note). Returns an array of shape (n, len(levels)),
    or (len(levels),) for a single-row DataFrame.

    All trained quantiles are predicted and sorted per row first (monotonicity
    guaranteed on output); if `levels` is given (a subset of the trained
    levels), only those columns are returned.
    """
    all_levels = artifacts["levels"]
    X = features[FEATURE_COLS]

    preds = np.column_stack([artifacts["models"][q].predict(X) for q in all_levels])
    preds = np.sort(preds, axis=1)

    if levels is not None:
        idx = [all_levels.index(q) for q in levels]
        preds = preds[:, idx]

    if preds.shape[0] == 1:
        return preds[0]
    return preds


def prob_over(artifacts, features, line):
    """Probability the rushing-yards outcome exceeds `line`, per feature row.

    CDF is interpolated piecewise-linearly through the row's 7 sorted
    quantiles; below p5 the CDF is clamped to 0.01 and above p95 to 0.99
    (mirroring the receiving interpolation). Returns 1 - CDF(line).

    `line` may be a scalar (applied to all rows) or an array aligned to the
    rows of `features`. Returns a scalar for a single-row, scalar-line call,
    else an array.
    """
    all_levels = np.array(artifacts["levels"])
    quantiles = np.atleast_2d(predict_quantiles(artifacts, features))
    n_rows = quantiles.shape[0]

    line = np.atleast_1d(line).astype(float)
    if line.shape[0] == 1:
        line = np.repeat(line, n_rows)

    if n_rows == 1 and line.shape[0] > 1:
        # one feature row evaluated against many lines
        quantiles = np.repeat(quantiles, line.shape[0], axis=0)
    elif line.shape[0] != quantiles.shape[0]:
        raise ValueError(
            f"line length ({line.shape[0]}) must be 1 or match feature rows ({n_rows})"
        )

    n = quantiles.shape[0]
    probs = np.empty(n)
    for i in range(n):
        qvals = quantiles[i]
        ln = line[i]
        if ln < qvals[0]:
            cdf = CDF_CLAMP_LOW
        elif ln > qvals[-1]:
            cdf = CDF_CLAMP_HIGH
        else:
            cdf = np.interp(ln, qvals, all_levels)
        probs[i] = 1.0 - cdf

    if n == 1:
        return probs[0]
    return probs


def run_week_rushing(season, week, line_overrides=None, lines=None, label=None, force=False):
    """Generate and save RB rushing-yards predictions for (season, week).

    Hybrid point/distribution design (mirrors run_week_receiving operationally
    but with two distinct estimators):

    1. Trains the rushing Ridge pipeline on all rows strictly before
       (season, week) and replays the current season's completed weeks through
       a single global RollingCalibrator(n0=N0) (stateless train-before-each-week
       replay) -> the calibrated point prediction `pred_rushing_yards` (headline).
    2. Loads the pre-fit quantile models via load_artifacts() -- these are STATIC
       (built once by scripts/build_rushing_prob_artifacts.py on 2021-2025), NOT
       retrained per week. In-season staleness caveat: unlike the Ridge point
       model (refit each week on all prior data) and the calibrator (replayed on
       the current season), the quantile distribution does not absorb the current
       season's games until the artifacts are rebuilt. Acceptable for a within-
       season weekly cadence; revisit if a season drifts materially.
    3. The distribution (q05..q95) and prob_over come entirely from the quantile
       models -- so q50 (quantile model) and pred_rushing_yards (calibrated Ridge)
       are two distinct estimates, both surfaced, deliberately named by source.

    `line_overrides`: optional dict {team: (spread_line, total_line)}.
    `lines`: optional dict {gsis_id: line} of sportsbook rushing-yards lines.
    `label`: optional filename suffix for non-canonical/test runs.

    The saved log carries `carries_l8` alongside `team_implied_total` and
    `as_of` so the prop slice (carries_l8 >= 8) is selectable directly from the
    log without rejoining the matrix.
    """
    artifacts = load_artifacts()

    matrix = pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet")
    df_processed = preprocess_rushing_matrix(matrix)

    # --- Global calibrator replay over the current season's completed weeks ---
    calibrator = RollingCalibrator(n0=N0)
    history_train = df_processed[df_processed["season"] < season]
    season_data = df_processed[df_processed["season"] == season]

    if not history_train.empty:
        history_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        history_pipeline.fit(history_train[FEATURE_COLS], history_train[TARGET])

        for w in range(1, week):
            week_data = season_data[season_data["week"] == w]
            if week_data.empty:
                continue
            raw_pred = history_pipeline.predict(week_data[FEATURE_COLS])
            calibrator.update(raw_pred, week_data[TARGET].values)

    offset = calibrator.offset

    # --- Final Ridge pipeline: trained on all rows strictly before (season, week) ---
    final_train = df_processed[
        (df_processed["season"] < season) | ((df_processed["season"] == season) & (df_processed["week"] < week))
    ]
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    final_pipeline.fit(final_train[FEATURE_COLS], final_train[TARGET])

    # --- Target-week features ---
    features = build_rushing_prediction_features(season, week, line_overrides=line_overrides)
    features = preprocess_rushing_matrix(features).reset_index(drop=True)

    # Headline point prediction: calibrated Ridge.
    raw_pred = final_pipeline.predict(features[FEATURE_COLS])
    calibrated_pred = calibrator.calibrate(raw_pred)

    # Distribution: static quantile models (sorted on output).
    quantiles = np.atleast_2d(predict_quantiles(artifacts, features))

    # carries_l8 is carried into the log as the prop-slice selector (the prop
    # population is carries_l8 >= 8, per steps 46/51). It is already a model
    # FEATURE_COL; surfacing it here is log metadata only -- no model change --
    # and it lets a live log be filtered to the priced population without
    # rejoining the rushing matrix.
    result = features[[
        "season", "week", "team", "opponent", "gsis_id", "player", "position",
        "team_implied_total", "as_of", "carries_l8",
    ]].copy()
    result["pred_rushing_yards"] = calibrated_pred
    for i, level in enumerate(QUANTILE_LEVELS):
        result[f"q{int(round(level * 100)):02d}"] = quantiles[:, i]
    result["calibrator_offset"] = offset

    if lines:
        result["line"] = result["gsis_id"].map(lines)
        result["prob_over"] = np.nan
        has_line = result["line"].notna().values
        if has_line.any():
            result.loc[has_line, "prob_over"] = prob_over(
                artifacts, features[has_line], result.loc[has_line, "line"].values
            )

    result["generated_at_utc"] = datetime.now(timezone.utc)
    result = result.sort_values("pred_rushing_yards", ascending=False).reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_rushing_yards{suffix}.parquet"
    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"{out_path} already exists. Prediction logs are immutable; pass force=True to overwrite."
            )
        print(f"WARNING: {out_path} already exists — overwriting (force=True).")

    result.to_parquet(out_path)
    print(f"\nSaved to: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate weekly RB rushing-yards predictions.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing prediction log")
    parser.add_argument("--label", default=None, help="suffix for non-canonical/test runs (avoids overwriting real weekly logs)")
    args = parser.parse_args()

    result = run_week_rushing(args.season, args.week, force=args.force, label=args.label)
    print(f"\nPredictions ({len(result)} rows):")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
