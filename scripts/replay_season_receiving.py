"""Full 2025 receiving season replay: run_week_receiving + grade_week_receiving
for every week, then aggregate.

Skips run / grade for a given week if its output file already exists (does not
force-overwrite — week 8 canonical + grades already exist from Step 37).
Per-week tables are suppressed; one progress line is printed per week, followed
by a season-wide aggregate, segment/position splits, per-position quantile-bucket
calibration tables, and a Travis Hunter spotlight (the documented two-way-player
limitation, now in graded reality).

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\replay_season_receiving.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.evaluate import EXPECTED_BUCKET_FREQ, QUANTILE_BUCKETS
from src.models.evaluate_receiving import PROP_MIN_TARGETS_L8, grade_week_receiving
from src.models.predict_receiving import run_week_receiving

SEASON = 2025
WEEKS = range(1, 23)
MODELED_POSITIONS = ["WR", "TE", "RB"]


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _availability(df):
    """GRADED / (GRADED + DID_NOT_PLAY) — roster-resolution quality."""
    n_graded = int((df["status"] == "GRADED").sum())
    n_dnp = int((df["status"] == "DID_NOT_PLAY").sum())
    denom = n_graded + n_dnp
    acc = n_graded / denom if denom else float("nan")
    return n_graded, denom, acc


def _prop_graded(df):
    """GRADED rows on the prop-relevant slice (actual targets_l8 >= threshold)."""
    return df[(df["status"] == "GRADED") & (df["actual_targets_l8"] >= PROP_MIN_TARGETS_L8)]


def _bucket_table(graded_df, indent="  "):
    n = len(graded_df)
    print(f"{indent}{'bucket':<10} {'observed':>10} {'expected':>10} {'count':>7}")
    for bucket in QUANTILE_BUCKETS:
        count = int((graded_df["quantile_bucket"] == bucket).sum())
        observed = count / n if n else float("nan")
        expected = EXPECTED_BUCKET_FREQ[bucket]
        obs_str = f"{observed * 100:>9.1f}%" if n else f"{'n/a':>10}"
        print(f"{indent}{bucket:<10} {obs_str} {expected * 100:>9.1f}% {count:>7}")


all_grades = []

print(f"{'week':<5} {'rows':>5} {'graded':>7} {'dnp':>5} {'nostat':>7} {'MAE-all':>8} {'MAE-prop':>9}  notes")

for week in WEEKS:
    pred_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_receiving_yards.parquet"
    grades_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_receiving_yards_grades.parquet"

    notes = []

    if pred_path.exists():
        notes.append("predict skipped (exists)")
    else:
        _quiet(run_week_receiving, SEASON, week)

    if grades_path.exists():
        notes.append("grade skipped (exists)")
        grades = pd.read_parquet(grades_path)
    else:
        grades = _quiet(grade_week_receiving, SEASON, week)

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
print(f"MAE (all GRADED, n={len(graded_all)}): {graded_all['abs_error'].mean():.2f}")
print(f"bias (all GRADED): {graded_all['error'].mean():.2f}")
print(f"MAE (prop-relevant, targets_l8>={PROP_MIN_TARGETS_L8}, n={len(prop_all)}): {prop_all['abs_error'].mean():.2f}")
print(f"bias (prop-relevant): {prop_all['error'].mean():.2f}")

print(f"\nFull-season quantile-bucket distribution (all GRADED, n={len(graded_all)}):")
_bucket_table(graded_all)

print(f"\nProp-relevant quantile-bucket distribution (n={len(prop_all)}):")
_bucket_table(prop_all)

print("\n" + "-" * 64)
print("Splits: availability accuracy & prop-slice MAE")
print("-" * 64)

reg = all_grades_df[all_grades_df["week"] <= 18]
playoffs = all_grades_df[all_grades_df["week"] >= 19]

print("\nBy season segment:")
print(f"  {'segment':<26} {'avail_acc':>16} {'prop_MAE':>10} {'prop_n':>7}")
for label, df in [("regular season (wk 1-18)", reg), ("playoffs (wk 19-22)", playoffs)]:
    n_g, n_d, acc = _availability(df)
    prop = _prop_graded(df)
    acc_str = f"{n_g}/{n_d}={acc:.3f}" if n_d else "n/a"
    mae_str = f"{prop['abs_error'].mean():.2f}" if len(prop) else "n/a"
    print(f"  {label:<26} {acc_str:>16} {mae_str:>10} {len(prop):>7}")

print("\nBy position:")
print(f"  {'position':<26} {'avail_acc':>16} {'prop_MAE':>10} {'prop_n':>7}")
for pos in MODELED_POSITIONS:
    df = all_grades_df[all_grades_df["position"] == pos]
    n_g, n_d, acc = _availability(df)
    prop = _prop_graded(df)
    acc_str = f"{n_g}/{n_d}={acc:.3f}" if n_d else "n/a"
    mae_str = f"{prop['abs_error'].mean():.2f}" if len(prop) else "n/a"
    print(f"  {pos:<26} {acc_str:>16} {mae_str:>10} {len(prop):>7}")

print("\n" + "-" * 64)
print("Per-position quantile-bucket calibration (prop-relevant slice)")
print("-" * 64)
for pos in MODELED_POSITIONS:
    pos_prop = prop_all[prop_all["position"] == pos]
    print(f"\n{pos} (n={len(pos_prop)}):")
    _bucket_table(pos_prop)

print("\n" + "-" * 64)
print("Travis Hunter spotlight (documented two-way-player limitation)")
print("-" * 64)
hunter = all_grades_df[all_grades_df["player"] == "Travis Hunter"].sort_values("week")
if len(hunter):
    cols = ["week", "team", "position", "status", "pred_receiving_yards",
            "actual_receiving_yards", "error", "sigma", "quantile_bucket"]
    print(hunter[cols].to_string(index=False))
    graded_h = hunter[hunter["status"] == "GRADED"]
    if len(graded_h):
        print(f"\n  graded weeks: {len(graded_h)}, MAE: {graded_h['abs_error'].mean():.2f}, "
              f"sigma applied: RB-baseline (position={graded_h['position'].iloc[0]} falls back to RB)")
else:
    print("Travis Hunter did not appear in any predicted row this season.")
