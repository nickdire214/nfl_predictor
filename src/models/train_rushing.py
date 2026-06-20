"""RB rushing-yards model training module.

Provides the preprocessing, feature list, and target for a Ridge model
predicting RB rushing yards from data/features/rushing_matrix.parquet,
mirroring src/models/train_receiving.py. RB-only, so no position dummies.

No model is trained/saved here yet (walk-forward evidence is gathered by
scripts/walkforward_rushing.py first).
"""

import numpy as np

from src.features.rushing import RUSHING_ROLLING_COLS
from src.ingestion._common import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

ROLLING_SUFFIXES = ("_l3", "_l8", "_std")
ROLLING_COLS = [f"{stat}{suffix}" for stat in RUSHING_ROLLING_COLS for suffix in ROLLING_SUFFIXES]

FEATURE_COLS = ROLLING_COLS + [
    "is_home", "team_rest", "opponent_rest", "div_game", "team_implied_total",
    "spread_line", "total_line", "games_into_season", "temp", "wind",
    "is_dome", "is_playoff",
]
TARGET = "rushing_yards"


def preprocess_rushing_matrix(df):
    df = df.dropna(subset=["carries_l3"]).copy()

    df["is_dome"] = df["roof"].isin(["dome", "closed"]).astype(int)
    df["temp"] = df["temp"].fillna(70)
    df["wind"] = df["wind"].fillna(0)

    for stat in RUSHING_ROLLING_COLS:
        df[f"{stat}_std"] = df[f"{stat}_std"].fillna(df[f"{stat}_l8"])

    df["is_playoff"] = (df["game_type"] != "REG").astype(int)

    return df


def _mae(pred, actual):
    return np.mean(np.abs(pred - actual))


def _rmse(pred, actual):
    return np.sqrt(np.mean((pred - actual) ** 2))


def _bias(pred, actual):
    return np.mean(pred - actual)
