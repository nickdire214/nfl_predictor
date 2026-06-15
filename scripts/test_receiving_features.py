"""Hard-assertion tests for receiving rolling features (receiving_matrix.parquet).

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_receiving_features.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "features" / "receiving_matrix.parquet"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.receiving import RECEIVING_ROLLING_COLS

ROLLING_SUFFIXES = ["_l3", "_l8", "_std"]
ALL_ROLLING_COLS = [f"{stat}{suffix}" for stat in RECEIVING_ROLLING_COLS for suffix in ROLLING_SUFFIXES]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(MATRIX_PATH)
df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)


# --- Test (a): shift correctness ---
section("(a) Shift correctness (second career game's targets_l3 == first game's actual)")
gsis_id = df.iloc[0]["gsis_id"]
g = df[df["gsis_id"] == gsis_id].sort_values(["season", "week"]).reset_index(drop=True)
row1, row2 = g.iloc[0], g.iloc[1]

assert math.isclose(row2["targets_l3"], row1["targets"]), (
    f"second game targets_l3 ({row2['targets_l3']}) != first game targets ({row1['targets']})"
)
print(f"PASS: {row1['player']} ({row1['season']} wk{row1['week']} -> {row2['season']} wk{row2['week']}) "
      f"targets_l3 ({row2['targets_l3']}) == first game targets ({row1['targets']})")


# --- Test (b): first career game all-NaN ---
section("(b) First career game: all rolling cols NaN")
for col in ALL_ROLLING_COLS:
    assert pd.isna(row1[col]), f"{col} is not NaN for {row1['player']}'s first game: {row1[col]}"
print(f"PASS: all {len(ALL_ROLLING_COLS)} rolling cols are NaN for "
      f"{row1['player']}'s first game ({row1['season']} week {row1['week']})")


# --- Test (c): no cross-player bleed ---
section("(c) No cross-player bleed (next player's first game also all-NaN)")
unique_ids = sorted(df["gsis_id"].unique())
idx_a = unique_ids.index(gsis_id)
gsis_id_b = unique_ids[idx_a + 1]
g_b = df[df["gsis_id"] == gsis_id_b].sort_values(["season", "week"]).reset_index(drop=True)
first_row_b = g_b.iloc[0]

for col in ALL_ROLLING_COLS:
    assert pd.isna(first_row_b[col]), (
        f"{col} is not NaN for {first_row_b['player']}'s first game "
        f"(possible bleed from {row1['player']}): {first_row_b[col]}"
    )
print(f"PASS: {first_row_b['player']}'s (gsis_id={gsis_id_b}, immediately after "
      f"{row1['player']} in sort order) first-game rolling cols are all NaN — no bleed")


# --- Test (d): manual L3 recompute, Ja'Marr Chase 2025 week 13 ---
section("(d) Manual L3 recompute: Ja'Marr Chase 2025 week 13 (prior played weeks 8, 9, 11)")
chase = df[(df["player"] == "Ja'Marr Chase") & (df["season"] == 2025)].sort_values("week").reset_index(drop=True)
target_idx = chase.index[chase["week"] == 13][0]
target_row = chase.iloc[target_idx]

prior_3 = chase.iloc[target_idx - 3:target_idx]
assert prior_3["week"].tolist() == [8, 9, 11], f"expected prior weeks [8, 9, 11], got {prior_3['week'].tolist()}"

manual_l3 = prior_3["targets"].mean()
assert math.isclose(target_row["targets_l3"], manual_l3), (
    f"matrix targets_l3 ({target_row['targets_l3']}) != manual recompute "
    f"of weeks 8, 9, 11 ({manual_l3})"
)
print(f"PASS: Ja'Marr Chase 2025 week 13 targets_l3 ({target_row['targets_l3']}) == "
      f"manual mean of weeks 8, 9, 11 targets {prior_3['targets'].tolist()} ({manual_l3})")


# --- Test (e): season-to-date reset ---
section("(e) Season-to-date reset (first game of new season has NaN _std)")
chase_id = chase.iloc[0]["gsis_id"]
chase_all = df[df["gsis_id"] == chase_id].sort_values(["season", "week"]).reset_index(drop=True)
seasons = sorted(chase_all["season"].unique())
assert len(seasons) >= 2, "Ja'Marr Chase only has one season in the data"

next_season_first = chase_all[chase_all["season"] == seasons[1]].sort_values("week").iloc[0]
std_cols = [f"{stat}_std" for stat in RECEIVING_ROLLING_COLS]
for col in std_cols:
    assert pd.isna(next_season_first[col]), (
        f"{col} is not NaN for Ja'Marr Chase's {seasons[1]} week "
        f"{next_season_first['week']} game: {next_season_first[col]}"
    )
print(f"PASS: Ja'Marr Chase's {seasons[1]} week {next_season_first['week']} "
      f"(first game of new season) has NaN for all {len(std_cols)} season-to-date (_std) columns")


# --- Test (f): zero-target inclusion ---
section("(f) Zero-target inclusion (a played-zero-target game counts as 0 in next L3, not skipped)")

found = None
for gid, player_games in df.groupby("gsis_id"):
    player_games = player_games.sort_values(["season", "week"]).reset_index(drop=True)
    for i in range(len(player_games) - 1):
        if player_games.loc[i, "targets"] != 0:
            continue
        prior = player_games.iloc[max(0, i - 2):i + 1]["targets"]
        if len(prior) < 2 or not (prior == 0).any() or not (prior > 0).any():
            continue

        manual_l3_incl = prior.mean()
        manual_l3_excl = prior[prior > 0].mean()
        next_row = player_games.iloc[i + 1]

        if math.isclose(next_row["targets_l3"], manual_l3_incl) and not math.isclose(manual_l3_incl, manual_l3_excl):
            found = (player_games.iloc[i], next_row, manual_l3_incl, manual_l3_excl, prior)
            break
    if found:
        break

assert found is not None, "no zero-target game found whose L3 inclusion could be verified"
zero_row, next_row, manual_l3_incl, manual_l3_excl, prior = found

print(f"PASS: {zero_row['player']} {zero_row['season']} week {zero_row['week']} had 0 targets "
      f"on {zero_row['offense_snaps']} snaps (played but uninvolved); next game "
      f"({next_row['season']} week {next_row['week']}) targets_l3 ({next_row['targets_l3']}) == "
      f"manual mean including the zero {prior.tolist()} ({manual_l3_incl}), "
      f"!= mean excluding it ({manual_l3_excl})")


print("\nAll tests passed.")
