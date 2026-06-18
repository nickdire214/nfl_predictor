"""Hard-assertion tests for RB rushing rolling features (rushing_matrix.parquet).

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_rushing_features.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "features" / "rushing_matrix.parquet"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.rushing import RUSHING_ROLLING_COLS

ROLLING_SUFFIXES = ["_l3", "_l8", "_std"]
ALL_ROLLING_COLS = [f"{stat}{suffix}" for stat in RUSHING_ROLLING_COLS for suffix in ROLLING_SUFFIXES]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(MATRIX_PATH)
df = df.sort_values(["gsis_id", "season", "week"]).reset_index(drop=True)


# --- Test (a): shift correctness ---
section("(a) Shift correctness (second career game's carries_l3 == first game's actual)")
gsis_id = df.iloc[0]["gsis_id"]
g = df[df["gsis_id"] == gsis_id].sort_values(["season", "week"]).reset_index(drop=True)
row1, row2 = g.iloc[0], g.iloc[1]

assert math.isclose(row2["carries_l3"], row1["carries"]), (
    f"second game carries_l3 ({row2['carries_l3']}) != first game carries ({row1['carries']})"
)
print(f"PASS: {row1['player']} ({row1['season']} wk{row1['week']} -> {row2['season']} wk{row2['week']}) "
      f"carries_l3 ({row2['carries_l3']}) == first game carries ({row1['carries']})")


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


# --- Test (d): manual L3 recompute across a missed game (Derrick Henry 2025) ---
section("(d) Manual L3 across a gap: Derrick Henry 2025 week 9 (prior PLAYED weeks skip absent wk7)")
henry = df[(df["player"] == "Derrick Henry") & (df["season"] == 2025)].sort_values("week").reset_index(drop=True)
assert 7 not in henry["week"].tolist(), f"expected Henry to be missing 2025 week 7, weeks present: {henry['week'].tolist()}"

target_idx = henry.index[henry["week"] == 9][0]
target_row = henry.iloc[target_idx]
prior_3 = henry.iloc[target_idx - 3:target_idx]
# The L3 window is the 3 most recent PLAYED games (games, not weeks): 5, 6, 8 —
# explicitly skipping the absent week 7.
assert prior_3["week"].tolist() == [5, 6, 8], f"expected prior played weeks [5, 6, 8], got {prior_3['week'].tolist()}"

manual_l3 = prior_3["carries"].mean()
assert math.isclose(target_row["carries_l3"], manual_l3), (
    f"matrix carries_l3 ({target_row['carries_l3']}) != manual recompute of weeks 5, 6, 8 ({manual_l3})"
)
print(f"PASS: Derrick Henry 2025 week 9 carries_l3 ({target_row['carries_l3']}) == "
      f"manual mean of played weeks 5, 6, 8 carries {prior_3['carries'].tolist()} ({manual_l3}) "
      f"— games-not-weeks indexing crossed the absent week 7")


# --- Test (e): season-to-date reset ---
section("(e) Season-to-date reset (first game of new season has NaN _std)")
henry_id = henry.iloc[0]["gsis_id"]
henry_all = df[df["gsis_id"] == henry_id].sort_values(["season", "week"]).reset_index(drop=True)
seasons = sorted(henry_all["season"].unique())
assert len(seasons) >= 2, "Derrick Henry only has one season in the data"

next_season_first = henry_all[henry_all["season"] == seasons[1]].sort_values("week").iloc[0]
std_cols = [f"{stat}_std" for stat in RUSHING_ROLLING_COLS]
for col in std_cols:
    assert pd.isna(next_season_first[col]), (
        f"{col} is not NaN for Derrick Henry's {seasons[1]} week "
        f"{next_season_first['week']} game: {next_season_first[col]}"
    )
print(f"PASS: Derrick Henry's {seasons[1]} week {next_season_first['week']} "
      f"(first game of new season) has NaN for all {len(std_cols)} season-to-date (_std) columns")


# --- Test (f): zero-carry inclusion ---
section("(f) Zero-carry inclusion (a dressed-zero-carry game counts as 0 in next L3, not skipped)")

found = None
for gid, player_games in df.groupby("gsis_id"):
    player_games = player_games.sort_values(["season", "week"]).reset_index(drop=True)
    for i in range(len(player_games) - 1):
        if player_games.loc[i, "carries"] != 0:
            continue
        prior = player_games.iloc[max(0, i - 2):i + 1]["carries"]
        if len(prior) < 2 or not (prior == 0).any() or not (prior > 0).any():
            continue

        manual_l3_incl = prior.mean()
        manual_l3_excl = prior[prior > 0].mean()
        next_row = player_games.iloc[i + 1]

        if math.isclose(next_row["carries_l3"], manual_l3_incl) and not math.isclose(manual_l3_incl, manual_l3_excl):
            found = (player_games.iloc[i], next_row, manual_l3_incl, manual_l3_excl, prior)
            break
    if found:
        break

assert found is not None, "no dressed-zero-carry game found whose L3 inclusion could be verified"
zero_row, next_row, manual_l3_incl, manual_l3_excl, prior = found

print(f"PASS: {zero_row['player']} {zero_row['season']} week {zero_row['week']} had 0 carries "
      f"on {zero_row['offense_snaps']} snaps (dressed, no carry); next game "
      f"({next_row['season']} week {next_row['week']}) carries_l3 ({next_row['carries_l3']}) == "
      f"manual mean including the zero {prior.tolist()} ({manual_l3_incl}), "
      f"!= mean excluding it ({manual_l3_excl})")


print("\nAll tests passed.")
