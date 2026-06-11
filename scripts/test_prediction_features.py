"""Hard-assertion tests for build_prediction_features (src/features/engineer.py).

Read-only verification script (test (b) temporarily rewrites and restores
data/raw/player_stats.parquet). Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_prediction_features.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.engineer import STAT_COLS, build_prediction_features
from src.models.train import ROLLING_COLS


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def values_match(a, b, tol=1e-6):
    if pd.isna(a) and pd.isna(b):
        return True
    if pd.isna(a) or pd.isna(b):
        return False
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return a == b


COMPARE_COLS = ROLLING_COLS + [
    "is_home", "team_rest", "opponent_rest", "div_game", "team_implied_total",
    "spread_line", "total_line", "games_into_season", "temp", "wind",
]


# --- (a) Replay equivalence for 2025 weeks 5 and 14 ---
section("(a) Replay equivalence: build_prediction_features vs qb_matrix.parquet")

qb_matrix = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")

for week in (5, 14):
    pred_df = build_prediction_features(2025, week)
    actual_df = qb_matrix[(qb_matrix["season"] == 2025) & (qb_matrix["week"] == week)]

    assert len(pred_df) == len(actual_df), (
        f"week {week}: row count mismatch (pred={len(pred_df)}, actual={len(actual_df)})"
    )

    mismatches = []
    for _, pred_row in pred_df.iterrows():
        team = pred_row["team"]
        actual_rows = actual_df[actual_df["team"] == team]
        assert len(actual_rows) == 1, f"week {week}, team {team}: expected 1 actual row, got {len(actual_rows)}"
        actual_row = actual_rows.iloc[0]

        for col in COMPARE_COLS:
            if not values_match(pred_row[col], actual_row[col]):
                mismatches.append((week, team, col, pred_row[col], actual_row[col]))

        # is_dome / is_playoff are derived from roof / game_type by preprocess_qb_matrix
        pred_is_dome = pred_row["roof"] in ("dome", "closed")
        actual_is_dome = actual_row["roof"] in ("dome", "closed")
        if pred_is_dome != actual_is_dome:
            mismatches.append((week, team, "is_dome", pred_is_dome, actual_is_dome))

        pred_is_playoff = pred_row["game_type"] != "REG"
        actual_is_playoff = actual_row["game_type"] != "REG"
        if pred_is_playoff != actual_is_playoff:
            mismatches.append((week, team, "is_playoff", pred_is_playoff, actual_is_playoff))

    if mismatches:
        print(f"\nWeek {week}: {len(mismatches)} mismatching (team, column) values:")
        for week_, team, col, pred_val, actual_val in mismatches:
            print(f"  team={team} col={col}: pred={pred_val} actual={actual_val}")

    assert not mismatches, f"week {week}: {len(mismatches)} feature mismatches (see above)"
    print(f"PASS: week {week} ({len(pred_df)} rows) — all feature columns match qb_matrix.parquet")


# --- (b) No self-leakage for 2025 week 10 ---
section("(b) No self-leakage: 2025 week 10 with/without its own player_stats rows")

player_stats_path = RAW_DATA_DIR / "player_stats.parquet"
original_player_stats = pd.read_parquet(player_stats_path)

normal_result = build_prediction_features(2025, 10)

modified_player_stats = original_player_stats[
    ~((original_player_stats["season"] == 2025) & (original_player_stats["week"] == 10))
]

try:
    modified_player_stats.to_parquet(player_stats_path)
    leakage_result = build_prediction_features(2025, 10)
finally:
    original_player_stats.to_parquet(player_stats_path)

normal_sorted = normal_result.sort_values("team").reset_index(drop=True)
leakage_sorted = leakage_result.sort_values("team").reset_index(drop=True)

assert len(normal_sorted) == len(leakage_sorted), "row count differs after removing week-10 stats"

mismatches = []
for col in normal_sorted.columns:
    for i in range(len(normal_sorted)):
        a, b = normal_sorted.loc[i, col], leakage_sorted.loc[i, col]
        if not values_match(a, b):
            mismatches.append((normal_sorted.loc[i, "team"], col, a, b))

if mismatches:
    print(f"\n{len(mismatches)} mismatching values after removing week-10 player_stats rows:")
    for team, col, a, b in mismatches:
        print(f"  team={team} col={col}: with_week10_stats={a} without_week10_stats={b}")

assert not mismatches, "week 10 features changed when its own player_stats rows were removed"
print("PASS: 2025 week 10 features identical with and without that week's own player_stats rows")


# --- (c) Forward sanity: a week with no games / no stats at all ---
section("(c) Forward sanity: week with no games (2025 week 23)")

future_df = build_prediction_features(2025, 23, starters={"KC": "00-0033873"})

assert list(future_df.columns) == list(qb_matrix.columns), (
    f"column mismatch: {set(future_df.columns) ^ set(qb_matrix.columns)}"
)
assert len(future_df) == 0, f"expected 0 rows for a non-existent week, got {len(future_df)}"
print(f"PASS: build_prediction_features(2025, 23, starters=...) ran without error, "
      f"returned shape {future_df.shape} with matching columns")


print("\nAll tests passed.")
