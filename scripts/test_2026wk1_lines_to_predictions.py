"""Step 21 chain test: saved odds snapshot -> consensus -> overrides -> run_week.

Loads the game-lines snapshot saved in Step 20, re-fetches events (free
endpoint) to recover event_id -> (season, week, team) matches, builds
consensus lines and line_overrides for 2026 week 1, then runs run_week
with label="june_test" so the output doesn't collide with real weekly logs.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_2026wk1_lines_to_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.odds import (
    consensus_game_lines,
    fetch_events,
    lines_to_overrides,
    match_events_to_schedule,
)
from src.models.predict import run_week

LINES_PATH = PROJECT_ROOT / "data" / "raw" / "odds" / "game_lines_2026wk1_test_2026-06-11.parquet"

lines_df = pd.read_parquet(LINES_PATH)
print(f"Loaded {len(lines_df)} lines rows from {LINES_PATH.name}")

events_df = fetch_events()
matched = match_events_to_schedule(events_df)

consensus_df = consensus_game_lines(lines_df)
print(f"\nConsensus lines: {len(consensus_df)} games")
print(consensus_df.to_string(index=False))

# lines_to_overrides now owns the week filter (step 75) — pass the full matched
# frame and the target week; it raises rather than silently spanning weeks.
overrides = lines_to_overrides(consensus_df, matched, 2026, 1)

# force=True: this is an idempotent rehearsal on a labeled (non-canonical) log,
# matching the receiving and rushing sibling scripts.
result = run_week(2026, 1, line_overrides=overrides, label="june_test", force=True)

print("\nPredictions:")
print(result.to_string(index=False))

missing_implied = result[result["team_implied_total"].isna()]
print(f"\nteam_implied_total missing: {len(missing_implied)}/{len(result)}")
if not missing_implied.empty:
    print(missing_implied[["team", "opponent"]].to_string(index=False))
