"""Step 40 chain test: saved odds snapshot -> consensus -> overrides -> run_week_receiving.

Mirrors scripts/test_2026wk1_lines_to_predictions.py (the QB chain from Step 21)
but drives the receiving stack. Loads the game-lines snapshot saved in Step 20,
re-fetches events (free endpoint) to recover event_id -> (season, week, team),
builds consensus lines and line_overrides for 2026 week 1, then runs
run_week_receiving with label="june_test" so the output doesn't collide with
real weekly logs.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_2026wk1_receiving_predictions.py
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.receiving import build_receiving_prediction_features
from src.ingestion.odds import (
    consensus_game_lines,
    fetch_events,
    lines_to_overrides,
    match_events_to_schedule,
)
from src.models.predict_receiving import run_week_receiving

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

# force=True: this is an idempotent rehearsal on a labeled (non-canonical) log.
result = run_week_receiving(2026, 1, line_overrides=overrides, label="june_test", force=True)

# --- Report ---
print(f"\nTotal predicted rows: {len(result)}")

print("\nTop 15 by predicted receiving yards:")
top_cols = ["team", "player", "position", "pred_receiving_yards", "p10", "p50", "p90",
            "sigma", "team_implied_total", "as_of"]
print(result[top_cols].head(15).to_string(index=False))

print("\nPredicted roster by position:")
print(result["position"].value_counts().to_string())

# (a) team_implied_total coverage — now carried in the runner's output (Step 41),
# so checked directly on result.
print("\n" + "=" * 60)
missing_implied = result[result["team_implied_total"].isna()]
print(f"(a) team_implied_total missing: {len(missing_implied)}/{len(result)}")
if missing_implied.empty:
    print("    PASS: every predicted row has a non-NaN team_implied_total")
else:
    print("    FLAG: rows missing implied total:")
    print(missing_implied[["team", "opponent", "player"]].to_string(index=False))

# (b) sigma varies by position as expected
print("\n(b) sigma by position (min / median / max):")
sig = result.groupby("position")["sigma"].agg(["min", "median", "max", "count"])
print(sig.to_string())
top_wr_sigma = result[result["position"] == "WR"].head(20)["sigma"]
if len(top_wr_sigma):
    print(f"    top-WR sigma range: {top_wr_sigma.min():.1f}-{top_wr_sigma.max():.1f} (expect ~38-42)")
if "RB" in sig.index and "WR" in sig.index:
    rb_med, wr_med = sig.loc["RB", "median"], sig.loc["WR", "median"]
    verdict = "PASS" if rb_med < wr_med else "FLAG"
    print(f"    {verdict}: RB median sigma {rb_med:.1f} < WR median sigma {wr_med:.1f} (RBs tighter)")

# (c) stale-roster visibility via the new as_of column
print("\n(c) Roster as_of staleness (2025-final rosters carried into 2026 wk1):")
print("    as_of value counts:")
print(result["as_of"].value_counts().sort_index().to_string().replace("\n", "\n      "))

# Deeply-stale: drawn from before 2025 wk18 (regular-season end) — most likely
# offseason casualties. Zero-padded as_of strings compare chronologically.
deeply_stale = result[result["as_of"] < "2025 wk18"]
print(f"\n    predicted players with as_of older than 2025 wk18: {len(deeply_stale)}/{len(result)}")
if not deeply_stale.empty:
    cols = ["team", "player", "position", "pred_receiving_yards", "as_of"]
    print(deeply_stale.sort_values("as_of")[cols].head(20).to_string(index=False))
print("\n    (as_of makes the September roster review tractable: sort by it, eyeball")
print("     the oldest. 2026 offseason departures are still not detectable from data")
print("     alone — they remain on 2025-final rosters until refreshed.)")
