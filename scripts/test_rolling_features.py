"""Hard-assertion tests for rolling feature correctness in qb_matrix.parquet.

Read-only verification script. Exits nonzero if any assertion fails.
Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_rolling_features.py
"""

import math
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MATRIX_PATH = PROJECT_ROOT / "data" / "features" / "qb_matrix.parquet"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

sys.path.insert(0, str(PROJECT_ROOT))
from src.features.engineer import DEFENSE_COLS

ROLLING_SUFFIXES = ["_l3", "_l8", "_std"]
STAT_COLS = [
    "passing_yards", "attempts", "completions", "passing_tds",
    "passing_interceptions", "passing_epa", "passing_cpoe", "sacks_suffered",
]
ROLLING_COLS = [f"{stat}{suffix}" for stat in STAT_COLS for suffix in ROLLING_SUFFIXES]


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(MATRIX_PATH)
df = df.sort_values(["qb_id", "season", "week"]).reset_index(drop=True)


# --- Test 1: Shift correctness ---
section("Test 1: Shift correctness (week-2 L3 == week-1 actual)")
qb_id = "00-0019596"  # Tom Brady, 2021 week1 -> week2
g = df[df["qb_id"] == qb_id].sort_values(["season", "week"])
row1 = g.iloc[0]
row2 = g.iloc[1]
assert row1["season"] == row2["season"] == 2021
assert row1["week"] == 1 and row2["week"] == 2

assert math.isclose(row2["passing_yards_l3"], row1["passing_yards"]), (
    f"week2 L3 ({row2['passing_yards_l3']}) != week1 actual ({row1['passing_yards']})"
)
print(f"PASS: {row1['player_display_name']} 2021 week2 passing_yards_l3 "
      f"({row2['passing_yards_l3']}) == week1 actual ({row1['passing_yards']})")


# --- Test 2: First-game NaN ---
section("Test 2: First-game NaN (all rolling features NaN for career-first game)")
first_row = row1
for col in ROLLING_COLS:
    assert pd.isna(first_row[col]), f"{col} is not NaN for {first_row['player_display_name']}'s first game: {first_row[col]}"
print(f"PASS: all {len(ROLLING_COLS)} rolling features are NaN for "
      f"{first_row['player_display_name']}'s career-first game ({first_row['season']} week {first_row['week']})")


# --- Test 3: No cross-player bleed ---
section("Test 3: No cross-player bleed (next QB's first game also NaN)")
unique_qb_ids = sorted(df["qb_id"].unique())
idx_a = unique_qb_ids.index(qb_id)
qb_id_b = unique_qb_ids[idx_a + 1]
g_b = df[df["qb_id"] == qb_id_b].sort_values(["season", "week"])
first_row_b = g_b.iloc[0]

for col in ROLLING_COLS:
    assert pd.isna(first_row_b[col]), (
        f"{col} is not NaN for {first_row_b['player_display_name']}'s first game "
        f"(possible bleed from {first_row['player_display_name']}): {first_row_b[col]}"
    )
print(f"PASS: {first_row_b['player_display_name']}'s (qb_id={qb_id_b}, immediately after "
      f"{first_row['player_display_name']} in sort order) first-game rolling features are all NaN — no bleed")


# --- Test 4: Self-exclusion (Mahomes 2023 manual recompute) ---
section("Test 4: Self-exclusion (Mahomes 2023 L3 manual recompute)")
mahomes_id = "00-0033873"
m = df[(df["qb_id"] == mahomes_id) & (df["season"] == 2023)].sort_values("week").reset_index(drop=True)

target_idx = m.index[m["week"] == 8][0]
target_row = m.iloc[target_idx]

prior_3 = m.iloc[target_idx - 3:target_idx]["passing_yards"]
manual_l3_excluding_current = prior_3.mean()

incl_3 = m.iloc[target_idx - 2:target_idx + 1]["passing_yards"]
manual_l3_including_current = incl_3.mean()

assert math.isclose(target_row["passing_yards_l3"], manual_l3_excluding_current), (
    f"matrix L3 ({target_row['passing_yards_l3']}) != manual recompute "
    f"of prior 3 games ({manual_l3_excluding_current})"
)
assert not math.isclose(target_row["passing_yards_l3"], manual_l3_including_current), (
    f"matrix L3 ({target_row['passing_yards_l3']}) unexpectedly equals the mean "
    f"that includes the current game ({manual_l3_including_current})"
)
print(f"PASS: Mahomes 2023 week 8 passing_yards_l3 ({target_row['passing_yards_l3']}) == "
      f"manual mean of weeks {m.iloc[target_idx-3:target_idx]['week'].tolist()} "
      f"({manual_l3_excluding_current})")
print(f"PASS: matrix L3 ({target_row['passing_yards_l3']}) != mean including current game "
      f"({manual_l3_including_current})")


# --- Test 5: Season-to-date reset ---
section("Test 5: Season-to-date reset (first game of new season has NaN _std)")
brady = df[df["qb_id"] == qb_id].sort_values(["season", "week"])
seasons = sorted(brady["season"].unique())
assert len(seasons) >= 2, f"{first_row['player_display_name']} only has one season in the data"

next_season_first = brady[brady["season"] == seasons[1]].sort_values("week").iloc[0]
std_cols = [f"{stat}_std" for stat in STAT_COLS]
for col in std_cols:
    assert pd.isna(next_season_first[col]), (
        f"{col} is not NaN for {next_season_first['player_display_name']}'s "
        f"{seasons[1]} week-{next_season_first['week']} game: {next_season_first[col]}"
    )
print(f"PASS: {next_season_first['player_display_name']}'s {seasons[1]} week "
      f"{next_season_first['week']} (first game of new season) has NaN for all "
      f"{len(std_cols)} season-to-date (_std) columns")



# --- Test 6: Defense feature correctness + first-appearance NaN ---
section("Test 6: Defense features (CIN 2023 week 10) + first-appearance NaN")

player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")

cin_games = []
for _, srow in schedules.iterrows():
    if srow["home_team"] == "CIN":
        cin_games.append({"season": srow["season"], "week": srow["week"], "offense_team": srow["away_team"]})
    elif srow["away_team"] == "CIN":
        cin_games.append({"season": srow["season"], "week": srow["week"], "offense_team": srow["home_team"]})

cin_games_df = pd.DataFrame(cin_games).sort_values(["season", "week"]).reset_index(drop=True)

target_idx = cin_games_df.index[(cin_games_df["season"] == 2023) & (cin_games_df["week"] == 10)][0]
prior_8 = cin_games_df.iloc[target_idx - 8:target_idx]
assert len(prior_8) == 8, f"expected 8 prior games for CIN before 2023 week 10, got {len(prior_8)}"

allowed = []
for _, g in prior_8.iterrows():
    pys = player_stats[
        (player_stats["season"] == g["season"]) & (player_stats["week"] == g["week"]) & (player_stats["team"] == g["offense_team"])
    ]["passing_yards"].sum()
    allowed.append(pys)

manual_l8 = sum(allowed) / len(allowed)

target_row = df[(df["season"] == 2023) & (df["week"] == 10) & (df["opponent"] == "CIN")].iloc[0]

assert math.isclose(target_row["def_pass_yards_allowed_l8"], manual_l8), (
    f"matrix def_pass_yards_allowed_l8 ({target_row['def_pass_yards_allowed_l8']}) != "
    f"manual recompute of CIN's prior 8 games ({manual_l8})"
)
print(f"PASS: CIN 2023 week 10 def_pass_yards_allowed_l8 ({target_row['def_pass_yards_allowed_l8']}) == "
      f"manual mean of CIN's prior 8 games' passing yards allowed ({manual_l8})")

first_cin_row = df[df["opponent"] == "CIN"].sort_values(["season", "week"]).iloc[0]
for col in DEFENSE_COLS:
    assert pd.isna(first_cin_row[col]), (
        f"{col} is not NaN for CIN's first appearance as a defense "
        f"({first_cin_row['season']} week {first_cin_row['week']}): {first_cin_row[col]}"
    )
print(f"PASS: CIN's first appearance as a defense ({first_cin_row['season']} week {first_cin_row['week']}) "
      f"has NaN for all {len(DEFENSE_COLS)} defense feature columns")


print("\nAll tests passed.")
