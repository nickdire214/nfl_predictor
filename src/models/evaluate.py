"""Weekly grader for QB passing-yards predictions.

Compares a saved prediction log (data/predictions/{season}_w{week:02d}_qb_pass_yards.parquet)
against actual outcomes in data/features/qb_matrix.parquet, and saves an
immutable grades file alongside it.
"""

import argparse

import numpy as np
import pandas as pd

from src.ingestion._common import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

QUANTILE_BUCKETS = ["below_p10", "p10-p25", "p25-p50", "p50-p75", "p75-p90", "above_p90"]
EXPECTED_BUCKET_FREQ = {
    "below_p10": 0.10,
    "p10-p25": 0.15,
    "p25-p50": 0.25,
    "p50-p75": 0.25,
    "p75-p90": 0.15,
    "above_p90": 0.10,
}


def _quantile_bucket(actual, p10, p25, p50, p75, p90):
    if actual < p10:
        return "below_p10"
    if actual < p25:
        return "p10-p25"
    if actual < p50:
        return "p25-p50"
    if actual < p75:
        return "p50-p75"
    if actual < p90:
        return "p75-p90"
    return "above_p90"


def grade_week(season, week, force=False):
    """Grade a saved weekly prediction file against actual outcomes.

    Returns the grades DataFrame and saves it to
    data/predictions/{season}_w{week:02d}_qb_pass_yards_grades.parquet
    (immutable: raises FileExistsError unless force=True).
    """
    pred_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_qb_pass_yards.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run `python -m src.models.predict "
            f"--season {season} --week {week}` first."
        )
    preds = pd.read_parquet(pred_path)

    qb_matrix = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
    actuals = qb_matrix[(qb_matrix["season"] == season) & (qb_matrix["week"] == week)][
        ["team", "qb_id", "player_display_name", "passing_yards"]
    ].rename(columns={
        "qb_id": "actual_qb_id",
        "player_display_name": "actual_qb_name",
        "passing_yards": "actual_passing_yards",
    })

    grades = preds.merge(actuals, on="team", how="left")

    def classify(row):
        if pd.isna(row["actual_qb_id"]):
            return "NO_STATS"
        if row["qb_id"] == row["actual_qb_id"]:
            return "GRADED"
        return "WRONG_STARTER"

    grades["status"] = grades.apply(classify, axis=1)

    graded_mask = grades["status"] == "GRADED"

    grades["error"] = np.nan
    grades["abs_error"] = np.nan
    grades["quantile_bucket"] = None

    grades.loc[graded_mask, "error"] = (
        grades.loc[graded_mask, "pred_passing_yards"] - grades.loc[graded_mask, "actual_passing_yards"]
    )
    grades.loc[graded_mask, "abs_error"] = grades.loc[graded_mask, "error"].abs()

    for idx in grades[graded_mask].index:
        row = grades.loc[idx]
        grades.loc[idx, "quantile_bucket"] = _quantile_bucket(
            row["actual_passing_yards"], row["p10"], row["p25"], row["p50"], row["p75"], row["p90"]
        )

    has_lines = "line" in preds.columns and "prob_over" in preds.columns
    if has_lines:
        grades["over_under_correct"] = pd.NA
        gradeable_line_mask = graded_mask & grades["line"].notna()
        actual_over = grades.loc[gradeable_line_mask, "actual_passing_yards"] > grades.loc[gradeable_line_mask, "line"]
        pred_over = grades.loc[gradeable_line_mask, "prob_over"] > 0.5
        grades.loc[gradeable_line_mask, "over_under_correct"] = (actual_over == pred_over)

    # --- Per-row table ---
    print("Per-row results:")
    display_cols = ["team", "qb_name", "status", "pred_passing_yards", "actual_passing_yards",
                     "error", "abs_error", "quantile_bucket"]
    if has_lines:
        display_cols += ["line", "prob_over", "over_under_correct"]
    display_cols += ["actual_qb_name"]
    print(grades[display_cols].to_string(index=False))

    # --- Summary ---
    n_graded = int(graded_mask.sum())
    n_wrong_starter = int((grades["status"] == "WRONG_STARTER").sum())
    n_no_stats = int((grades["status"] == "NO_STATS").sum())
    n_played = n_graded + n_wrong_starter

    print("\nSummary:")
    print(f"  rows: {len(grades)} (GRADED={n_graded}, WRONG_STARTER={n_wrong_starter}, NO_STATS={n_no_stats})")

    if n_wrong_starter:
        print("  wrong-starter games:")
        for _, row in grades[grades["status"] == "WRONG_STARTER"].iterrows():
            print(f"    {row['team']}: predicted {row['qb_name']}, actual starter was {row['actual_qb_name']}")

    if n_played:
        starter_accuracy = n_graded / n_played
        print(f"  starter accuracy: {n_graded}/{n_played} = {starter_accuracy:.3f}")
    else:
        print("  starter accuracy: n/a (no played games)")

    if n_graded:
        mae = grades.loc[graded_mask, "abs_error"].mean()
        bias = grades.loc[graded_mask, "error"].mean()
        print(f"  MAE: {mae:.2f}")
        print(f"  bias: {bias:.2f}")

        print("\n  Quantile-bucket distribution (observed vs expected):")
        print(f"  {'bucket':<10} {'observed':>10} {'expected':>10} {'count':>7}")
        for bucket in QUANTILE_BUCKETS:
            count = int((grades.loc[graded_mask, "quantile_bucket"] == bucket).sum())
            observed = count / n_graded
            expected = EXPECTED_BUCKET_FREQ[bucket]
            print(f"  {bucket:<10} {observed * 100:>9.1f}% {expected * 100:>9.1f}% {count:>7}")
    else:
        print("  MAE/bias/quantile distribution: n/a (no graded rows)")

    if has_lines and n_graded:
        gradeable = grades.loc[graded_mask & grades["line"].notna()]
        if len(gradeable):
            n_correct = int(gradeable["over_under_correct"].sum())
            print(f"\n  over/under accuracy: {n_correct}/{len(gradeable)} = {n_correct / len(gradeable):.3f}")
        else:
            print("\n  over/under accuracy: n/a (no rows with a line)")

    # --- Save grades ---
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_qb_pass_yards_grades.parquet"
    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"{out_path} already exists. Grade logs are immutable; pass force=True to overwrite."
            )
        print(f"\nWARNING: {out_path} already exists — overwriting (force=True).")

    grades.to_parquet(out_path)
    print(f"\nSaved to: {out_path}")

    return grades


def main():
    parser = argparse.ArgumentParser(description="Grade a weekly QB passing-yards prediction file.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing grades file")
    args = parser.parse_args()

    grade_week(args.season, args.week, force=args.force)


if __name__ == "__main__":
    main()
