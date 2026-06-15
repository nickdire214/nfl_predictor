"""Full 2025 season replay: run_week + grade_week for every week, then aggregate.

Skips run_week / grade_week for a given week if its output file already
exists (does not force-overwrite — week 8 already exists from the Step 17/18
dry runs). Per-week tables are suppressed; one progress line is printed per
week, followed by a season-wide aggregate.

Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\replay_season.py
"""

import io
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

sys.path.insert(0, str(PROJECT_ROOT))
from src.models.evaluate import EXPECTED_BUCKET_FREQ, QUANTILE_BUCKETS, grade_week
from src.models.predict import run_week

SEASON = 2025
WEEKS = range(1, 23)


def _quiet(fn, *args, **kwargs):
    with redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


def _starter_accuracy(df):
    n_graded = int((df["status"] == "GRADED").sum())
    n_wrong = int((df["status"] == "WRONG_STARTER").sum())
    n_played = n_graded + n_wrong
    acc = n_graded / n_played if n_played else float("nan")
    return n_graded, n_played, acc


all_grades = []

print(f"{'week':<6} {'rows':>6} {'graded':>7} {'wrong_starter':>14} {'MAE':>8}  notes")

for week in WEEKS:
    pred_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_qb_pass_yards.parquet"
    grades_path = PREDICTIONS_DIR / f"{SEASON}_w{week:02d}_qb_pass_yards_grades.parquet"

    notes = []

    if pred_path.exists():
        notes.append("predict skipped (exists)")
    else:
        _quiet(run_week, SEASON, week)

    if grades_path.exists():
        notes.append("grade skipped (exists)")
        grades = pd.read_parquet(grades_path)
        grades["status"] = grades["status"].replace("NO_GAME", "NO_STATS")
    else:
        grades = _quiet(grade_week, SEASON, week)

    all_grades.append(grades)

    n_rows = len(grades)
    n_graded = int((grades["status"] == "GRADED").sum())
    n_wrong = int((grades["status"] == "WRONG_STARTER").sum())
    mae = grades.loc[grades["status"] == "GRADED", "abs_error"].mean()
    mae_str = f"{mae:.2f}" if pd.notna(mae) else "n/a"

    note_str = f"  ({'; '.join(notes)})" if notes else ""
    print(f"{week:<6} {n_rows:>6} {n_graded:>7} {n_wrong:>14} {mae_str:>8}{note_str}")


all_grades_df = pd.concat(all_grades, ignore_index=True)

n_graded_total, n_played_total, overall_acc = _starter_accuracy(all_grades_df)
n_no_stats_total = int((all_grades_df["status"] == "NO_STATS").sum())
graded = all_grades_df[all_grades_df["status"] == "GRADED"]
mae = graded["abs_error"].mean()
bias = graded["error"].mean()

print("\n" + "=" * 60)
print("Season aggregate (2025, weeks 1-22)")
print("=" * 60)
print(f"total rows: {len(all_grades_df)}")
print(f"GRADED={n_graded_total}, WRONG_STARTER={n_played_total - n_graded_total}, NO_STATS={n_no_stats_total}")
print(f"starter accuracy: {n_graded_total}/{n_played_total} = {overall_acc:.3f}")
print(f"MAE: {mae:.2f}")
print(f"bias: {bias:.2f}")

print("\nQuantile-bucket distribution (observed vs expected):")
print(f"{'bucket':<10} {'observed':>10} {'expected':>10} {'count':>7}")
n_graded_for_buckets = len(graded)
for bucket in QUANTILE_BUCKETS:
    count = int((graded["quantile_bucket"] == bucket).sum())
    observed = count / n_graded_for_buckets
    expected = EXPECTED_BUCKET_FREQ[bucket]
    print(f"{bucket:<10} {observed * 100:>9.1f}% {expected * 100:>9.1f}% {count:>7}")

print("\nStarter accuracy by season segment:")
reg = all_grades_df[all_grades_df["week"] <= 18]
playoffs = all_grades_df[all_grades_df["week"] >= 19]
early = all_grades_df[all_grades_df["week"] <= 4]
later = all_grades_df[all_grades_df["week"] >= 5]

for label, df in [
    ("regular season (wk 1-18)", reg),
    ("playoffs (wk 19-22)", playoffs),
    ("weeks 1-4", early),
    ("weeks 5+", later),
]:
    n_g, n_p, acc = _starter_accuracy(df)
    acc_str = f"{acc:.3f}" if n_p else "n/a"
    print(f"  {label:<26} {n_g}/{n_p} = {acc_str}")
