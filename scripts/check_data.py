"""Diagnostic script: verify nflreadpy data access (player stats and snap counts).

Quick standalone check, not pipeline code. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\check_data.py
"""

import nflreadpy as nfl

SEASON = 2025


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


try:
    section("Loading player stats")
    stats = nfl.load_player_stats([SEASON]).to_pandas()
    print(f"Shape: {stats.shape}")
    print(f"Columns ({len(stats.columns)}):")
    for col in stats.columns:
        print(f"  {col}")
    print(f"Unique players: {stats['player_id'].nunique()}")
    print(f"Unique weeks: {stats['week'].nunique()}")
except Exception as e:
    print(f"ERROR loading player stats: {e}")


try:
    section("Ja'Marr Chase weekly receiving line (2025)")
    chase = stats[stats["player_display_name"].str.contains("Chase", case=False, na=False)]
    chase = chase[chase["position"] == "WR"]
    chase = chase.sort_values("week")
    cols = ["week", "targets", "receptions", "receiving_yards", "receiving_tds"]
    print(chase[cols].to_string(index=False))
except Exception as e:
    print(f"ERROR filtering Ja'Marr Chase: {e}")


try:
    section("Loading snap counts")
    snaps = nfl.load_snap_counts([SEASON]).to_pandas()
    print(f"Shape: {snaps.shape}")
    print(f"Columns ({len(snaps.columns)}):")
    for col in snaps.columns:
        print(f"  {col}")
except Exception as e:
    print(f"ERROR loading snap counts: {e}")
