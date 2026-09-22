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
from src.features.engineer import DEFENSE_COLS, STAT_COLS, build_prediction_features
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


COMPARE_COLS = ROLLING_COLS + DEFENSE_COLS + [
    "is_home", "team_rest", "opponent_rest", "div_game", "team_implied_total",
    "spread_line", "total_line", "games_into_season", "temp", "wind",
]


# --- (a) Replay equivalence for 2025 weeks 5 and 14 ---
section("(a) Replay equivalence: build_prediction_features vs qb_matrix.parquet")

qb_matrix = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")

# Attempts per (team, qb) for the disagreement report below. Informational only.
_ps = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
_att = _ps.set_index(["season", "week", "team", "player_id"])["attempts"]
_names = _ps.drop_duplicates("player_id").set_index("player_id")["player_display_name"]


def _qb_label(pid, season, week, team):
    """'Name (N att)' for the disagreement report."""
    name = _names.get(pid, pid)
    try:
        a = _att.get((season, week, team, pid))
        a = int(a) if pd.notna(a) else None
    except (KeyError, TypeError):
        a = None
    return f"{name} ({a} att)" if a is not None else f"{name} (no att)"


for week in (5, 14):
    pred_df = build_prediction_features(2025, week)
    actual_df = qb_matrix[(qb_matrix["season"] == 2025) & (qb_matrix["week"] == week)]

    assert len(pred_df) == len(actual_df), (
        f"week {week}: row count mismatch (pred={len(pred_df)}, actual={len(actual_df)})"
    )

    # COMMON-KEY PARITY, not row-for-row parity (step 96).
    #
    # The two paths label the starting quarterback by DIFFERENT definitions, and
    # that is deliberate rather than a defect:
    #
    #   qb_matrix                 labels by HINDSIGHT  -- the leading passer by
    #                             attempts, which is what actually happened
    #                             (step 95).
    #   build_prediction_features labels by FORECAST   -- schedules' designated
    #                             starter, or a starters_override entry, which
    #                             is all a Wednesday run can know.
    #
    # On the 127 of 2,912 team-games where nflverse's designation was not the
    # leading passer, the two therefore name different quarterbacks and their
    # rolling features legitimately differ -- they describe different players.
    # Asserting row-for-row equality would force the serving path to use
    # hindsight, which would inject lookahead into every backtest.
    #
    # So parity is asserted over the INTERSECTION: team-games where both paths
    # name the same quarterback. That set must match exactly -- any mismatch
    # there is a real feature-construction bug. Disagreements are reported
    # informationally, the same pattern test_receiving_prediction_features and
    # test_rushing_prediction_features use for roster churn.
    pred_qb = dict(zip(pred_df["team"], pred_df["qb_id"]))
    actual_qb = dict(zip(actual_df["team"], actual_df["qb_id"]))
    common_teams = {tm for tm, q in pred_qb.items()
                    if tm in actual_qb and q == actual_qb[tm]}
    disagreements = sorted(tm for tm in pred_qb
                           if tm in actual_qb and pred_qb[tm] != actual_qb[tm])

    print(f"week {week}: pred={len(pred_df)} rows, actual={len(actual_df)} rows, "
          f"common_qb={len(common_teams)}, disagreements={len(disagreements)}")
    if disagreements:
        print(f"  starter disagreements (predicted vs matrix) — informational:")
        for tm in disagreements:
            print(f"    {tm:4s} predicted={_qb_label(pred_qb[tm], 2025, week, tm):32s}"
                  f" matrix={_qb_label(actual_qb[tm], 2025, week, tm)}")

    mismatches = []
    for _, pred_row in pred_df.iterrows():
        team = pred_row["team"]
        if team not in common_teams:
            continue
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
    assert common_teams, f"week {week}: no team-game where both paths name the same QB"
    print(f"PASS: week {week} ({len(common_teams)} common-QB rows, "
          f"{len(disagreements)} starter disagreement(s) skipped) — all feature "
          f"columns match qb_matrix.parquet")


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
