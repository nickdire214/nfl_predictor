"""Probability layer and weekly prediction runner for the receiving yards model.

Implements the position+prediction-conditional sigma architecture adopted
in DECISIONS.md (steps 27-32): sigma_for(position, pred) interpolates the
out-of-sample residual std within a position over prediction-quintile bin
medians (clamped at the ends), and the z-pool (residual / sigma_pos,
mean-centered, see scripts/build_receiving_prob_artifacts.py) supplies the
distribution shape -- including its right skew -- for prob_over and
predict_quantiles.

Also implements the weekly runner (run_week_receiving) that trains the
receiving Ridge pipeline on history, replays a single global
RollingCalibrator over the current season's completed weeks, builds
prediction-time features for the resolved per-team rosters, attaches a
per-row conditional-sigma probability layer, and writes an immutable
per-week prediction log -- mirroring src.models.predict.run_week.

Known blemish (documented, not fixed): bottom-quartile median (50%-nominal)
coverage runs ~5.7pp hot. Tracked via live grading.
"""

import argparse
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.receiving import build_receiving_prediction_features
from src.ingestion._common import PROJECT_ROOT
from src.models.calibrate import RollingCalibrator
from src.models.train_receiving import FEATURE_COLS, TARGET, preprocess_receiving_matrix

MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
RIDGE_PATH = MODELS_DIR / "receiving_yards_ridge.pkl"
SIGMA_TABLE_PATH = MODELS_DIR / "receiving_sigma_table.parquet"
Z_POOL_PATH = MODELS_DIR / "receiving_z_pool.npy"

N0 = 150
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)

# The point model encodes any non-WR/non-TE position as the RB baseline
# (pos_WR=0, pos_TE=0), and the sigma table is fit only on RB/TE/WR. A canonical
# position outside these (e.g. a two-way CB like Travis Hunter, or a FB) is
# therefore mapped to RB for sigma/quantile purposes -- consistent with how its
# point prediction is produced.
SIGMA_FALLBACK_POSITION = "RB"


def load_artifacts():
    """Load the receiving model pipeline and probability-layer artifacts.

    Returns a dict with keys:
        "pipeline":   the fitted Ridge pipeline (models/receiving_yards_ridge.pkl)
        "sigma_bins": dict of position -> (bin_median_pred array, std array),
                       sorted by bin_median_pred, for use with sigma_for
        "z_pool":     mean-centered residual/sigma_pos values (models/receiving_z_pool.npy)
    """
    pipeline = joblib.load(RIDGE_PATH)
    sigma_table = pd.read_parquet(SIGMA_TABLE_PATH)
    z_pool = np.load(Z_POOL_PATH)

    sigma_bins = {}
    for position, group in sigma_table.groupby("position"):
        group = group.sort_values("bin_median_pred")
        sigma_bins[position] = (group["bin_median_pred"].to_numpy(), group["std"].to_numpy())

    return {"pipeline": pipeline, "sigma_bins": sigma_bins, "z_pool": z_pool}


def sigma_for(artifacts, position, pred):
    """Interpolate the residual std for `position` at prediction value(s) `pred`.

    Interpolates within the position's prediction-quintile bin medians;
    out-of-range predictions are clamped to the nearest end bin's std
    (np.interp default behavior). `pred` may be a scalar or array-like.
    Positions absent from the sigma table fall back to the RB baseline
    (see SIGMA_FALLBACK_POSITION).
    """
    if position not in artifacts["sigma_bins"]:
        position = SIGMA_FALLBACK_POSITION
    medians, stds = artifacts["sigma_bins"][position]
    return np.interp(pred, medians, stds)


def prob_over(artifacts, pred, position, line):
    """Probability that the realized outcome is strictly above `line`.

    Computed as the fraction of (pred + sigma_for(position, pred) * z_pool)
    that is strictly greater than `line`. `pred` and `line` may be scalars
    or array-likes (broadcast against each other); `position` is a single
    position string applying to all rows of `pred`.
    """
    pred = np.atleast_1d(pred).astype(float)
    line = np.atleast_1d(line).astype(float)
    z_pool = artifacts["z_pool"]

    sigma = np.atleast_1d(sigma_for(artifacts, position, pred))
    samples = pred[:, None] + sigma[:, None] * z_pool[None, :]
    probs = np.mean(samples > line[:, None], axis=1)

    if probs.shape[0] == 1:
        return probs[0]
    return probs


def predict_quantiles(artifacts, pred, position, levels=QUANTILE_LEVELS):
    """Predicted outcome quantiles: pred + sigma_for(position, pred) * z_pool quantiles.

    `pred` may be a scalar or array-like. Returns an array of shape
    (len(levels),) for a scalar prediction, or (n, len(levels)) for an
    array of n predictions.
    """
    z_pool = artifacts["z_pool"]
    pool_quantiles = np.quantile(z_pool, levels)

    pred = np.asarray(pred, dtype=float)
    sigma = sigma_for(artifacts, position, pred)

    if pred.ndim == 0:
        return pred + sigma * pool_quantiles

    return pred[:, None] + sigma[:, None] * pool_quantiles[None, :]


def run_week_receiving(season, week, line_overrides=None, lines=None, label=None, force=False):
    """Generate and save receiving-yards predictions for (season, week).

    Mirrors src.models.predict.run_week, adapted for the receiving stack:

    1. Trains the receiving Ridge pipeline on all rows strictly before
       (season, week) — the pipeline used to predict the target week.
    2. Separately, trains a pipeline on rows strictly before `season` and
       replays the current season's completed weeks (1..week-1) through a
       single global RollingCalibrator(n0=N0), one week at a time,
       accumulating prediction errors (stateless train-before-each-week
       replay). One offset is applied across all receivers.
    3. Builds target-week features via build_receiving_prediction_features
       (resolved per-team rosters), preprocesses, predicts, and calibrates.
    4. Attaches a per-row conditional-sigma probability layer: sigma_for
       (position, calibrated_pred), the five quantiles via predict_quantiles,
       and — if `lines` is given — line and prob_over per row. Unmodeled
       positions (FB/CB) fall back to RB sigma (see sigma_for).
    5. Saves to
       data/predictions/{season}_w{week:02d}_receiving_yards{_label}.parquet,
       raising FileExistsError if it already exists unless force=True.

    `line_overrides`: optional dict {team: (spread_line, total_line)},
    passed through to build_receiving_prediction_features.

    `lines`: optional dict {gsis_id: line} of sportsbook receiving-yards lines.

    `label`: optional string; if set the filename gains a `_{label}` suffix,
    for non-canonical/test runs that mustn't collide with real weekly logs.
    """
    artifacts = load_artifacts()

    matrix = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
    df_processed = preprocess_receiving_matrix(matrix)

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

    # --- Final pipeline: trained on all rows strictly before (season, week) ---
    final_train = df_processed[
        (df_processed["season"] < season) | ((df_processed["season"] == season) & (df_processed["week"] < week))
    ]
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    final_pipeline.fit(final_train[FEATURE_COLS], final_train[TARGET])

    # --- Target-week features ---
    features = build_receiving_prediction_features(season, week, line_overrides=line_overrides)
    features = preprocess_receiving_matrix(features)

    raw_pred = final_pipeline.predict(features[FEATURE_COLS])
    calibrated_pred = calibrator.calibrate(raw_pred)

    positions = features["position"].to_numpy()

    # --- Per-row conditional-sigma probability layer ---
    sigmas = np.array([sigma_for(artifacts, pos, p) for pos, p in zip(positions, calibrated_pred)])
    quantiles = np.array([
        predict_quantiles(artifacts, p, pos, levels=QUANTILE_LEVELS)
        for p, pos in zip(calibrated_pred, positions)
    ])

    result = features[[
        "season", "week", "team", "opponent", "gsis_id", "player", "position",
        "team_implied_total", "as_of",
    ]].reset_index(drop=True).copy()
    result["pred_receiving_yards"] = calibrated_pred
    for i, level in enumerate(QUANTILE_LEVELS):
        result[f"p{int(round(level * 100))}"] = quantiles[:, i]
    result["sigma"] = sigmas
    result["calibrator_offset"] = offset

    if lines:
        result["line"] = result["gsis_id"].map(lines)
        result["prob_over"] = np.nan
        has_line = result["line"].notna()
        for idx in result.index[has_line]:
            result.loc[idx, "prob_over"] = prob_over(
                artifacts,
                result.loc[idx, "pred_receiving_yards"],
                result.loc[idx, "position"],
                result.loc[idx, "line"],
            )

    result["generated_at_utc"] = datetime.now(timezone.utc)
    result = result.sort_values("pred_receiving_yards", ascending=False).reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_receiving_yards{suffix}.parquet"
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
    parser = argparse.ArgumentParser(description="Generate weekly receiving-yards predictions.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing prediction log")
    parser.add_argument("--label", default=None, help="suffix for non-canonical/test runs (avoids overwriting real weekly logs)")
    args = parser.parse_args()

    result = run_week_receiving(args.season, args.week, force=args.force, label=args.label)
    print(f"\nPredictions ({len(result)} rows):")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
