"""Step 52 chain test: saved odds snapshot -> consensus -> overrides -> run_week_rushing.

Mirrors scripts/test_2026wk1_receiving_predictions.py (Step 40) but drives the
rushing stack. Loads the game-lines snapshot saved in Step 20 (no new API call),
re-fetches events (free endpoint) to recover event_id -> (season, week, team),
builds consensus lines and line_overrides for 2026 week 1, then runs
run_week_rushing with label="aug_test" so the output doesn't collide with real
weekly logs.

Rehearsal / integration check only -- no DECISIONS entry.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\test_2026wk1_rushing_predictions.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.odds import (
    consensus_game_lines,
    fetch_events,
    lines_to_overrides,
    match_events_to_schedule,
)
from src.models.predict_rushing import QUANTILE_LEVELS, run_week_rushing

LINES_PATH = PROJECT_ROOT / "data" / "raw" / "odds" / "game_lines_2026wk1_test_2026-06-11.parquet"

QUANTILE_COLS = [f"q{int(round(level * 100)):02d}" for level in QUANTILE_LEVELS]

# The prop population the rushing layer is fitted on and serves (steps 46 / 51).
PROP_SLICE_MIN_CARRIES_L8 = 8

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
result = run_week_rushing(2026, 1, line_overrides=overrides, label="aug_test", force=True)

# --- Report ---
print(f"\nTotal predicted rows: {len(result)}")

print("\nTop 20 by predicted rushing yards:")
top_cols = ["team", "player", "carries_l8", "pred_rushing_yards", "q10", "q50", "q90",
            "team_implied_total", "as_of"]
print(result[top_cols].head(20).to_string(index=False))

print("\nBacks predicted per team:")
per_team = result.groupby("team").size()
print(f"    teams with rows: {len(per_team)}")
print(f"    min {per_team.min()} / median {per_team.median():.1f} / max {per_team.max()}")
print("    distribution (backs -> n teams):")
print(per_team.value_counts().sort_index().to_string().replace("\n", "\n      "))

# (a) team_implied_total coverage -- carried in the runner's output, checked directly.
print("\n" + "=" * 60)
missing_implied = result[result["team_implied_total"].isna()]
print(f"(a) team_implied_total missing: {len(missing_implied)}/{len(result)}")
if missing_implied.empty:
    print("    PASS: every predicted row has a non-NaN team_implied_total")
else:
    print("    FLAG: rows missing implied total:")
    print(missing_implied[["team", "opponent", "player"]].to_string(index=False))

# (b) Quantile monotonicity -- the sort-on-output guarantee (predict_quantiles sorts
# each row), now exercised on out-of-sample-season rows. The models themselves cross
# ~25.8% of the time out of sample (documented step-46 blemish), so this checks that
# the sort is doing its job on the saved log, not that the raw models are monotone.
print("\n(b) Quantile monotonicity (q05 <= q10 <= ... <= q95) on every row:")
qvals = result[QUANTILE_COLS].to_numpy(dtype=float)
diffs = np.diff(qvals, axis=1)
violations = (diffs < 0).any(axis=1)
n_viol = int(violations.sum())
print(f"    quantile columns checked: {', '.join(QUANTILE_COLS)}")
print(f"    rows violating monotonicity: {n_viol}/{len(result)}")
if n_viol == 0:
    print("    PASS: monotonic on every row")
else:
    print("    FLAG: non-monotonic rows:")
    print(result.loc[violations, ["team", "player"] + QUANTILE_COLS].to_string(index=False))

# (c) Ridge-vs-q50 relationship at forward-week scale. The 2025 season replay found
# mean signed gap +4.58 (Ridge above the quantile median -- the right-skew signature)
# and mean |gap| 6.54. This is a shape check on a forward week, not a pass/fail bar.
print("\n(c) Calibrated Ridge vs quantile-model median (pred_rushing_yards - q50):")
gap = result["pred_rushing_yards"] - result["q50"]


def _gap_stats(label, mask):
    g = gap[mask]
    if g.empty:
        print(f"    {label:22s} n=0")
        return
    print(f"    {label:22s} n={len(g):3d}   mean {g.mean():+6.2f}   "
          f"mean|gap| {g.abs().mean():5.2f}   median {g.median():+6.2f}   "
          f"range {g.min():+7.2f} to {g.max():+7.2f}")


# Split by the prop-slice selector now carried in the log (carries_l8 >= 8).
# The step-34 replay figures (mean +4.58, mean|gap| 6.54) are PROP-SLICE numbers,
# not all-GRADED (verified in step 54 item 0 by recomputing from the 2025 grades
# files: all-GRADED is +3.73 / 6.02). So the prop line below is the apples-to-
# apples comparator, and the all-rows line is NOT comparable to +4.58.
prop_mask = result["carries_l8"] >= PROP_SLICE_MIN_CARRIES_L8
_gap_stats("all rows", pd.Series(True, index=result.index))
_gap_stats(f"prop (carries_l8>={PROP_SLICE_MIN_CARRIES_L8})", prop_mask)
_gap_stats("rest (low-volume)", ~prop_mask)
print(f"\n    2025 replay reference (PROP SLICE, the comparable one): mean +4.58, mean|gap| 6.54")
print(f"    2025 replay reference (all GRADED, not comparable): mean +3.73, mean|gap| 6.02")
skew_verdict = "consistent with right-skew" if gap.mean() > 0 else "FLAG: Ridge below q50 on average"
print(f"    {skew_verdict}")

print("\n    largest divergences (|gap|), where the two estimators disagree most:")
widest = result.assign(gap=gap).reindex(gap.abs().sort_values(ascending=False).index)
wide_cols = ["team", "player", "carries_l8", "pred_rushing_yards", "q50", "gap"]
print(widest[wide_cols].head(10).to_string(index=False))

print(f"\n    prop-slice size: {int(prop_mask.sum())}/{len(result)} rows "
      f"(carries_l8 >= {PROP_SLICE_MIN_CARRIES_L8})")
missing_cl8 = int(result["carries_l8"].isna().sum())
print(f"    carries_l8 missing: {missing_cl8}/{len(result)}"
      + ("  (NaN rows fall outside the prop slice)" if missing_cl8 else ""))

# (d) Stale-roster visibility via as_of.
print("\n(d) Roster as_of staleness (2025-final rosters carried into 2026 wk1):")
print("    as_of value counts:")
print(result["as_of"].value_counts().sort_index().to_string().replace("\n", "\n      "))

# Deeply-stale: drawn from before 2025 wk18 (regular-season end) -- most likely
# offseason casualties. Zero-padded as_of strings compare chronologically.
deeply_stale = result[result["as_of"] < "2025 wk18"]
print(f"\n    predicted players with as_of older than 2025 wk18: {len(deeply_stale)}/{len(result)}")
if not deeply_stale.empty:
    cols = ["team", "player", "pred_rushing_yards", "as_of"]
    print(deeply_stale.sort_values("as_of")[cols].head(20).to_string(index=False))
print("\n    (as_of makes the September roster review tractable: sort by it, eyeball")
print("     the oldest. 2026 offseason departures are still not detectable from data")
print("     alone -- they remain on 2025-final rosters until refreshed.)")
