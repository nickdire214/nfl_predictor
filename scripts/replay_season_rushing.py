"""Full 2025 RB rushing season replay: run_week_rushing + grade_week_rushing
for every week, then aggregate.

Skips run / grade for a given week if its output file already exists (does not
force-overwrite -- week 8 canonical + grades already exist from Step 50).
Per-week detail is suppressed; one progress line per week, then a season-wide
aggregate, segment splits, the full-season TIE-AWARE quantile-bucket calibration
(the real verdict on the shipped quantile-reg layer), an explicit tail-dispersion
read (is the tails-heavy week-8 hint real?), and the season-scale q50-vs-Ridge
point comparison.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\replay_season_rushing.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.evaluate_rushing import (
    PROP_MIN_CARRIES_L8,
    RUSHING_EXPECTED_BUCKET_FREQ,
    RUSHING_QUANTILE_BUCKETS,
    _bucket_weights_tieaware,
    grade_week_rushing,
)
from src.models.predict_rushing import run_week_rushing

SEASON = 2025
WEEKS = range(1, 23)
TAIL_LIMIT = 0.25  # combined below-q10 + above-q90 mass above this => intervals too narrow


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _availability(df):
    n_graded = int((df["status"] == "GRADED").sum())
    n_dnp = int((df["status"] == "DID_NOT_PLAY").sum())
    denom = n_graded + n_dnp
    acc = n_graded / denom if denom else float("nan")
    return n_graded, denom, acc


def _prop_graded(df):
    return df[(df["status"] == "GRADED") & (df["actual_carries_l8"] >= PROP_MIN_CARRIES_L8)]


def _accumulate_buckets(graded_df):
    """Tie-aware fractional bucket totals over a GRADED df (recomputed from the
    stored quantile columns + actual, identical logic to the grader)."""
    totals = {b: 0.0 for b in RUSHING_QUANTILE_BUCKETS}
    for _, r in graded_df.iterrows():
        w = _bucket_weights_tieaware(
            r["actual_rushing_yards"], r["q10"], r["q25"], r["q50"], r["q75"], r["q90"]
        )
        for b, val in w.items():
            totals[b] += val
    return totals


def _bucket_table(totals, n, indent="  "):
    print(f"{indent}{'bucket':<10} {'observed':>10} {'expected':>10} {'weight':>8}")
    for bucket in RUSHING_QUANTILE_BUCKETS:
        observed = totals[bucket] / n if n else float("nan")
        expected = RUSHING_EXPECTED_BUCKET_FREQ[bucket]
        obs_str = f"{observed * 100:>9.1f}%" if n else f"{'n/a':>10}"
        print(f"{indent}{bucket:<10} {obs_str} {expected * 100:>9.1f}% {totals[bucket]:>8.1f}")


all_grades = []

print(f"{'week':<5} {'rows':>5} {'graded':>7} {'dnp':>5} {'nostat':>7} {'MAE-all':>8} {'MAE-prop':>9}  notes")

for week in WEEKS:
    pred_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_rushing_yards.parquet"
    grades_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_rushing_yards_grades.parquet"

    notes = []

    if pred_path.exists():
        notes.append("predict skipped (exists)")
    else:
        _quiet(run_week_rushing, SEASON, week)

    if grades_path.exists():
        notes.append("grade skipped (exists)")
        grades = pd.read_parquet(grades_path)
    else:
        grades = _quiet(grade_week_rushing, SEASON, week)

    all_grades.append(grades)

    n_rows = len(grades)
    n_graded = int((grades["status"] == "GRADED").sum())
    n_dnp = int((grades["status"] == "DID_NOT_PLAY").sum())
    n_nostat = int((grades["status"] == "NO_STATS").sum())

    graded = grades[grades["status"] == "GRADED"]
    mae_all = graded["abs_error"].mean()
    mae_prop = _prop_graded(grades)["abs_error"].mean()
    mae_all_str = f"{mae_all:.2f}" if pd.notna(mae_all) else "n/a"
    mae_prop_str = f"{mae_prop:.2f}" if pd.notna(mae_prop) else "n/a"

    note_str = f"  ({'; '.join(notes)})" if notes else ""
    print(f"{week:<5} {n_rows:>5} {n_graded:>7} {n_dnp:>5} {n_nostat:>7} "
          f"{mae_all_str:>8} {mae_prop_str:>9}{note_str}")


all_grades_df = pd.concat(all_grades, ignore_index=True)
graded_all = all_grades_df[all_grades_df["status"] == "GRADED"]
prop_all = _prop_graded(all_grades_df)

n_graded_total, avail_denom, overall_acc = _availability(all_grades_df)
n_dnp_total = int((all_grades_df["status"] == "DID_NOT_PLAY").sum())
n_nostat_total = int((all_grades_df["status"] == "NO_STATS").sum())

print("\n" + "=" * 64)
print("Season aggregate (2025, weeks 1-22)")
print("=" * 64)
print(f"total rows: {len(all_grades_df)}")
print(f"GRADED={n_graded_total}, DID_NOT_PLAY={n_dnp_total}, NO_STATS={n_nostat_total}")
print(f"availability accuracy: {n_graded_total}/{avail_denom} = {overall_acc:.3f}")
print(f"point MAE (calibrated-Ridge, all GRADED, n={len(graded_all)}): {graded_all['abs_error'].mean():.2f}")
print(f"point bias (all GRADED): {graded_all['error'].mean():.2f}")
print(f"point MAE (prop-relevant, carries_l8>={PROP_MIN_CARRIES_L8}, n={len(prop_all)}): {prop_all['abs_error'].mean():.2f}")
print(f"point bias (prop-relevant): {prop_all['error'].mean():.2f}")

prop_totals = _accumulate_buckets(prop_all)
all_totals = _accumulate_buckets(graded_all)

print(f"\nFull-season tie-aware quantile-bucket distribution (all GRADED, n={len(graded_all)}):")
_bucket_table(all_totals, len(graded_all))

print(f"\nProp-relevant tie-aware quantile-bucket distribution (n={len(prop_all)}):")
_bucket_table(prop_totals, len(prop_all))

print("\n" + "-" * 64)
print("Splits: availability accuracy & prop-slice point-MAE")
print("-" * 64)
reg = all_grades_df[all_grades_df["week"] <= 18]
playoffs = all_grades_df[all_grades_df["week"] >= 19]
print(f"\n  {'segment':<26} {'avail_acc':>16} {'prop_MAE':>10} {'prop_n':>7}")
for label, df in [("regular season (wk 1-18)", reg), ("playoffs (wk 19-22)", playoffs)]:
    n_g, n_d, acc = _availability(df)
    prop = _prop_graded(df)
    acc_str = f"{n_g}/{n_d}={acc:.3f}" if n_d else "n/a"
    mae_str = f"{prop['abs_error'].mean():.2f}" if len(prop) else "n/a"
    print(f"  {label:<26} {acc_str:>16} {mae_str:>10} {len(prop):>7}")

print("\n" + "-" * 64)
print("Tail-dispersion read (prop slice): are the intervals too narrow?")
print("-" * 64)
n_prop = len(prop_all)
below = prop_totals["below_q10"] / n_prop
above = prop_totals["above_q90"] / n_prop
combined = below + above
print(f"  below q10 (nominal 10%): {below * 100:.1f}%  (weight {prop_totals['below_q10']:.1f})")
print(f"  above q90 (nominal 10%): {above * 100:.1f}%  (weight {prop_totals['above_q90']:.1f})")
print(f"  combined tail mass (nominal 20%): {combined * 100:.1f}%")
if combined > TAIL_LIMIT:
    print(f"  -> FLAG: combined tail mass {combined * 100:.1f}% exceeds {TAIL_LIMIT * 100:.0f}% "
          f"-- intervals run TOO NARROW in live application; documented limitation feeding the threshold decision.")
else:
    print(f"  -> OK: combined tail mass {combined * 100:.1f}% within tolerance (<= {TAIL_LIMIT * 100:.0f}%); "
          f"the week-8 tails-heavy hint does not hold at season scale.")

print("\n" + "-" * 64)
print("q50 (quantile model) vs pred_rushing_yards (calibrated Ridge), prop slice")
print("-" * 64)
gap = prop_all["pred_rushing_yards"] - prop_all["q50"]
mean_abs_gap = gap.abs().mean()
mean_signed_gap = gap.mean()
higher = "Ridge" if mean_signed_gap > 0 else "q50"
print(f"  mean |pred - q50|: {mean_abs_gap:.2f}")
print(f"  mean (pred - q50): {mean_signed_gap:+.2f}  -> {higher} runs higher on average")
print(f"  (expectation: Ridge (mean-like) above q50 (median) is the right-skew signature)")
