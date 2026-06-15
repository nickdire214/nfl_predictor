"""Probability layer for the receiving yards model.

Implements the position+prediction-conditional sigma architecture adopted
in DECISIONS.md (steps 27-32): sigma_for(position, pred) interpolates the
out-of-sample residual std within a position over prediction-quintile bin
medians (clamped at the ends), and the z-pool (residual / sigma_pos,
mean-centered, see scripts/build_receiving_prob_artifacts.py) supplies the
distribution shape -- including its right skew -- for prob_over and
predict_quantiles.

Known blemish (documented, not fixed): bottom-quartile median (50%-nominal)
coverage runs ~5.7pp hot. Tracked via live grading.
"""

import joblib
import numpy as np
import pandas as pd

from src.ingestion._common import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
RIDGE_PATH = MODELS_DIR / "receiving_yards_ridge.pkl"
SIGMA_TABLE_PATH = MODELS_DIR / "receiving_sigma_table.parquet"
Z_POOL_PATH = MODELS_DIR / "receiving_z_pool.npy"

QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)


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
    """
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
