"""Hard-assertion tests for build_receiving_prediction_features (src/features/receiving.py).

Read-only verification script (test (b) temporarily rewrites and restores
data/raw/snap_counts.parquet and data/raw/player_stats.parquet). Exits
nonzero if any assertion fails. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_receiving_prediction_features.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.receiving import build_receiving_prediction_features
from src.models.train_receiving import ROLLING_COLS


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
section("(a) Replay equivalence: build_receiving_prediction_features vs receiving_matrix.parquet")

receiving_matrix = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")

for week in (5, 14):
    pred_df = build_receiving_prediction_features(2025, week)
    actual_df = receiving_matrix[(receiving_matrix["season"] == 2025) & (receiving_matrix["week"] == week)]

    pred_keys = set(zip(pred_df["team"], pred_df["gsis_id"]))
    actual_keys = set(zip(actual_df["team"], actual_df["gsis_id"]))
    common_keys = pred_keys & actual_keys

    # Predicted roster is "each team's roster from its most recent prior game"
    # — week-to-week roster churn (returns from rest/injury, healthy
    # scratches) means pred and actual row sets are not identical. Feature
    # equivalence is checked over the intersection; coverage is informational.
    print(f"week {week}: pred={len(pred_df)} rows, actual={len(actual_df)} rows, "
          f"common={len(common_keys)}, only_in_actual={len(actual_keys - pred_keys)}, "
          f"only_in_pred={len(pred_keys - actual_keys)}")

    mismatches = []
    for _, pred_row in pred_df.iterrows():
        team, gsis_id = pred_row["team"], pred_row["gsis_id"]
        if (team, gsis_id) not in common_keys:
            continue
        actual_rows = actual_df[(actual_df["team"] == team) & (actual_df["gsis_id"] == gsis_id)]
        assert len(actual_rows) == 1, (
            f"week {week}, team {team}, gsis_id {gsis_id}: expected 1 actual row, got {len(actual_rows)}"
        )
        actual_row = actual_rows.iloc[0]

        for col in COMPARE_COLS:
            if not values_match(pred_row[col], actual_row[col]):
                mismatches.append((week, team, gsis_id, col, pred_row[col], actual_row[col]))

        # is_dome / is_playoff are derived from roof / game_type, sourced from the
        # current week's schedule (not the roster snapshot) -- always hard-checked.
        pred_is_dome = pred_row["roof"] in ("dome", "closed")
        actual_is_dome = actual_row["roof"] in ("dome", "closed")
        if pred_is_dome != actual_is_dome:
            mismatches.append((week, team, gsis_id, "is_dome", pred_is_dome, actual_is_dome))

        pred_is_playoff = pred_row["game_type"] != "REG"
        actual_is_playoff = actual_row["game_type"] != "REG"
        if pred_is_playoff != actual_is_playoff:
            mismatches.append((week, team, gsis_id, "is_playoff", pred_is_playoff, actual_is_playoff))

        # pos_WR / pos_TE derive from `position`, now sourced canonically from
        # players.parquet in BOTH paths (step 35), so they must match exactly --
        # the prior per-week snap-label flip (e.g. Waller 2025 wk13) is gone.
        for pos_col, pos_val in (("pos_WR", "WR"), ("pos_TE", "TE")):
            pred_pos = pred_row["position"] == pos_val
            actual_pos = actual_row["position"] == pos_val
            if pred_pos != actual_pos:
                mismatches.append((week, team, gsis_id, pos_col, pred_pos, actual_pos))

    if mismatches:
        print(f"\nWeek {week}: {len(mismatches)} mismatching (team, gsis_id, column) values:")
        for week_, team, gsis_id, col, pred_val, actual_val in mismatches:
            print(f"  team={team} gsis_id={gsis_id} col={col}: pred={pred_val} actual={actual_val}")

    assert not mismatches, f"week {week}: {len(mismatches)} feature mismatches (see above)"
    assert len(common_keys) > 0, f"week {week}: no overlap between predicted and actual rosters"
    print(f"PASS: week {week} ({len(common_keys)} common rows) — all feature columns match receiving_matrix.parquet")


# --- (b) No self-leakage for 2025 week 10 ---
section("(b) No self-leakage: 2025 week 10 with/without its own snap + stats inputs")

snap_counts_path = RAW_DATA_DIR / "snap_counts.parquet"
player_stats_path = RAW_DATA_DIR / "player_stats.parquet"
original_snap_counts = pd.read_parquet(snap_counts_path)
original_player_stats = pd.read_parquet(player_stats_path)

normal_result = build_receiving_prediction_features(2025, 10)

modified_snap_counts = original_snap_counts[
    ~((original_snap_counts["season"] == 2025) & (original_snap_counts["week"] == 10))
]
modified_player_stats = original_player_stats[
    ~((original_player_stats["season"] == 2025) & (original_player_stats["week"] == 10))
]

try:
    modified_snap_counts.to_parquet(snap_counts_path)
    modified_player_stats.to_parquet(player_stats_path)
    leakage_result = build_receiving_prediction_features(2025, 10)
finally:
    original_snap_counts.to_parquet(snap_counts_path)
    original_player_stats.to_parquet(player_stats_path)

normal_sorted = normal_result.sort_values(["team", "gsis_id"]).reset_index(drop=True)
leakage_sorted = leakage_result.sort_values(["team", "gsis_id"]).reset_index(drop=True)

assert len(normal_sorted) == len(leakage_sorted), (
    f"row count differs after removing week-10 snap/stats rows "
    f"(normal={len(normal_sorted)}, leakage={len(leakage_sorted)})"
)

mismatches = []
for col in normal_sorted.columns:
    for i in range(len(normal_sorted)):
        a, b = normal_sorted.loc[i, col], leakage_sorted.loc[i, col]
        if not values_match(a, b):
            mismatches.append((normal_sorted.loc[i, "team"], normal_sorted.loc[i, "gsis_id"], col, a, b))

if mismatches:
    print(f"\n{len(mismatches)} mismatching values after removing week-10 snap/stats rows:")
    for team, gsis_id, col, a, b in mismatches:
        print(f"  team={team} gsis_id={gsis_id} col={col}: with_week10={a} without_week10={b}")

assert not mismatches, "week 10 features changed when its own snap/stats rows were removed"
print("PASS: 2025 week 10 features identical with and without that week's own snap_counts/player_stats rows")


# --- (c) Forward sanity: a week with no games / no stats at all ---
section("(c) Forward sanity: week with no games (2025 week 23)")

# `as_of` is a prediction-time metadata column (not in receiving_matrix); the
# column set must otherwise match the matrix exactly.
METADATA_COLS = {"as_of"}

future_df = build_receiving_prediction_features(2025, 23)

extra = set(future_df.columns) - set(receiving_matrix.columns)
missing = set(receiving_matrix.columns) - set(future_df.columns)
assert extra == METADATA_COLS and not missing, (
    f"column mismatch: extra={extra} (expected {METADATA_COLS}), missing={missing}"
)
assert len(future_df) == 0, f"expected 0 rows for a non-existent week, got {len(future_df)}"
print(f"PASS: build_receiving_prediction_features(2025, 23) ran without error, "
      f"returned shape {future_df.shape} with matching columns (+ {METADATA_COLS} metadata)")


print("\nAll tests passed.")
