"""Weekly grader for receiving-yards predictions.

Compares a saved prediction log
(data/predictions/{season}_w{week:02d}_receiving_yards.parquet) against
actual outcomes in data/features/receiving_matrix.parquet, and saves an
immutable grades file alongside it. Mirrors src.models.evaluate.grade_week,
reusing its quantile-bucket and over/under logic, but adds a three-way play
classification (GRADED / DID_NOT_PLAY / NO_STATS) because the receiving
prediction population is a resolved multi-player roster per team, not a
single starter.
"""

import argparse

import numpy as np
import pandas as pd

from src.ingestion._common import PROJECT_ROOT
from src.models.evaluate import EXPECTED_BUCKET_FREQ, QUANTILE_BUCKETS, _quantile_bucket

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

PROP_MIN_TARGETS_L8 = 3


def _snap_gsis_ids(season, week):
    """Set of gsis_ids with ANY snap_counts row for (season, week).

    snap_counts is keyed by pfr_player_id; map it to gsis_id via the
    players crosswalk. Used to separate DID_NOT_PLAY (no snap row at all
    — roster staleness / inactive) from NO_STATS (took a snap but produced
    no receiving-matrix row — e.g. 0 offensive snaps, non-skill snap
    position, or an unmatched crosswalk — a genuine data gap).
    """
    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    crosswalk = players[["gsis_id", "pfr_id"]].dropna()

    snaps = pd.read_parquet(RAW_DATA_DIR / "snap_counts.parquet")
    snaps_week = snaps[(snaps["season"] == season) & (snaps["week"] == week)]
    snaps_week = snaps_week.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left")

    return set(snaps_week["gsis_id"].dropna())


def grade_week_receiving(season, week, force=False):
    """Grade a saved weekly receiving-yards prediction file against actuals.

    Returns the grades DataFrame and saves it to
    data/predictions/{season}_w{week:02d}_receiving_yards_grades.parquet
    (immutable: raises FileExistsError unless force=True).
    """
    pred_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_receiving_yards.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run `python -m src.models.predict_receiving "
            f"--season {season} --week {week}` first. (Note: this grades the canonical file, "
            f"not a labeled dry-run such as ..._receiving_yards_dryrun.parquet.)"
        )
    preds = pd.read_parquet(pred_path)

    receiving_matrix = pd.read_parquet(FEATURES_DIR / "receiving_matrix.parquet")
    actuals = receiving_matrix[
        (receiving_matrix["season"] == season) & (receiving_matrix["week"] == week)
    ][["team", "gsis_id", "player", "position", "receiving_yards", "targets_l8"]].rename(columns={
        "player": "actual_player",
        "position": "actual_position",
        "receiving_yards": "actual_receiving_yards",
        "targets_l8": "actual_targets_l8",
    })
    actuals["matrix_row"] = True

    grades = preds.merge(actuals, on=["team", "gsis_id"], how="left")
    grades["matrix_row"] = grades["matrix_row"].fillna(False)

    snap_ids = _snap_gsis_ids(season, week)

    def classify(row):
        if row["matrix_row"]:
            return "GRADED"
        if row["gsis_id"] in snap_ids:
            return "NO_STATS"
        return "DID_NOT_PLAY"

    grades["status"] = grades.apply(classify, axis=1)

    graded_mask = grades["status"] == "GRADED"

    grades["error"] = np.nan
    grades["abs_error"] = np.nan
    grades["quantile_bucket"] = None

    grades.loc[graded_mask, "error"] = (
        grades.loc[graded_mask, "pred_receiving_yards"] - grades.loc[graded_mask, "actual_receiving_yards"]
    )
    grades.loc[graded_mask, "abs_error"] = grades.loc[graded_mask, "error"].abs()

    for idx in grades[graded_mask].index:
        row = grades.loc[idx]
        grades.loc[idx, "quantile_bucket"] = _quantile_bucket(
            row["actual_receiving_yards"], row["p10"], row["p25"], row["p50"], row["p75"], row["p90"]
        )

    has_lines = "line" in preds.columns and "prob_over" in preds.columns
    if has_lines:
        grades["over_under_correct"] = pd.NA
        gradeable_line_mask = graded_mask & grades["line"].notna()
        actual_over = grades.loc[gradeable_line_mask, "actual_receiving_yards"] > grades.loc[gradeable_line_mask, "line"]
        pred_over = grades.loc[gradeable_line_mask, "prob_over"] > 0.5
        grades.loc[gradeable_line_mask, "over_under_correct"] = (actual_over == pred_over)

    # --- Per-row table (top 30 GRADED by prediction; full table is 270+) ---
    print("Per-row results (top 30 GRADED by predicted receiving yards):")
    display_cols = ["team", "player", "position", "status", "pred_receiving_yards",
                    "actual_receiving_yards", "error", "abs_error", "quantile_bucket"]
    if has_lines:
        display_cols += ["line", "prob_over", "over_under_correct"]
    graded_view = grades[graded_mask].sort_values("pred_receiving_yards", ascending=False).head(30)
    print(graded_view[display_cols].to_string(index=False))

    # --- Summary ---
    n_graded = int(graded_mask.sum())
    n_did_not_play = int((grades["status"] == "DID_NOT_PLAY").sum())
    n_no_stats = int((grades["status"] == "NO_STATS").sum())

    print("\nSummary:")
    print(f"  rows: {len(grades)} (GRADED={n_graded}, DID_NOT_PLAY={n_did_not_play}, NO_STATS={n_no_stats})")

    # Availability accuracy: the roster-resolution quality metric (analogous to
    # the QB grader's starter accuracy). NO_STATS is a data gap, not a roster
    # miss, so it is excluded from the denominator.
    avail_denom = n_graded + n_did_not_play
    if avail_denom:
        print(f"  availability accuracy: {n_graded}/{avail_denom} = {n_graded / avail_denom:.3f}")
    else:
        print("  availability accuracy: n/a (no resolved-roster rows)")

    if n_graded:
        mae = grades.loc[graded_mask, "abs_error"].mean()
        bias = grades.loc[graded_mask, "error"].mean()
        print(f"  MAE (all GRADED, n={n_graded}): {mae:.2f}")
        print(f"  bias (all GRADED): {bias:.2f}")

        prop_mask = graded_mask & (grades["actual_targets_l8"] >= PROP_MIN_TARGETS_L8)
        n_prop = int(prop_mask.sum())
        if n_prop:
            prop_mae = grades.loc[prop_mask, "abs_error"].mean()
            prop_bias = grades.loc[prop_mask, "error"].mean()
            print(f"  MAE (prop-relevant GRADED, targets_l8>={PROP_MIN_TARGETS_L8}, n={n_prop}): {prop_mae:.2f}")
            print(f"  bias (prop-relevant GRADED): {prop_bias:.2f}")
        else:
            print(f"  prop-relevant slice (targets_l8>={PROP_MIN_TARGETS_L8}): n/a (no rows)")

        print("\n  Quantile-bucket distribution (observed vs expected, all GRADED):")
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
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_receiving_yards_grades.parquet"
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
    parser = argparse.ArgumentParser(description="Grade a weekly receiving-yards prediction file.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing grades file")
    args = parser.parse_args()

    grade_week_receiving(args.season, args.week, force=args.force)


if __name__ == "__main__":
    main()
