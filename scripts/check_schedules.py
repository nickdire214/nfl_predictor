"""Diagnostic script: explore data/raw/schedules.parquet.

Quick standalone, read-only exploration. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\check_schedules.py
"""

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCHEDULES_PATH = PROJECT_ROOT / "data" / "raw" / "schedules.parquet"

TEAM = "CIN"  # Bengals


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


df = pd.read_parquet(SCHEDULES_PATH)

try:
    section("Shape and dtypes")
    print(f"Shape: {df.shape}")
    print(df.dtypes.to_string())
except Exception as e:
    print(f"ERROR printing shape/dtypes: {e}")


try:
    section("First 3 rows, transposed")
    print(df.head(3).T.to_string())
except Exception as e:
    print(f"ERROR printing head transposed: {e}")


try:
    section(f"{TEAM} 2025 regular season - rest day dry run")

    reg_2025 = df[(df["season"] == 2025) & (df["game_type"] == "REG")]

    home = reg_2025[reg_2025["home_team"] == TEAM].copy()
    home["opponent"] = home["away_team"]
    home["home_away"] = "home"

    away = reg_2025[reg_2025["away_team"] == TEAM].copy()
    away["opponent"] = away["home_team"]
    away["home_away"] = "away"

    team_games = pd.concat([home, away], ignore_index=True)
    team_games["gameday"] = pd.to_datetime(team_games["gameday"])
    team_games = team_games.sort_values("gameday").reset_index(drop=True)

    team_games["days_since_previous"] = team_games["gameday"].diff().dt.days

    view = team_games[["week", "gameday", "opponent", "home_away", "weekday", "days_since_previous"]]
    print(view.to_string(index=False))
except Exception as e:
    print(f"ERROR building rest day view: {e}")


try:
    section(f"{TEAM} 2025 - bye week and short/long week flags")

    weeks_present = set(team_games["week"])
    all_weeks = set(range(1, reg_2025["week"].max() + 1))
    bye_weeks = sorted(all_weeks - weeks_present)
    print(f"Bye week(s): {bye_weeks}")

    print("\nFlags:")
    for _, row in team_games.iterrows():
        gap = row["days_since_previous"]
        if pd.isna(gap):
            continue
        if gap <= 5:
            print(f"  Week {row['week']} ({row['gameday'].date()}): SHORT WEEK ({gap:.0f} days rest)")
        elif gap >= 9:
            print(f"  Week {row['week']} ({row['gameday'].date()}): LONG WEEK / post-bye ({gap:.0f} days rest)")
except Exception as e:
    print(f"ERROR computing bye/flags: {e}")
