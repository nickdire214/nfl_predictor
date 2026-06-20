"""Weekly grader for RB rushing-yards predictions (hybrid log).

Compares a saved prediction log
(data/predictions/{season}_w{week:02d}_rushing_yards.parquet) against actual
outcomes in data/features/rushing_matrix.parquet, and saves an immutable
grades file alongside it. Mirrors src.models.evaluate_receiving (three-way
play classification, availability accuracy, prop-slice MAE) with two rushing
adaptations:

  - POINT grading is on the calibrated-Ridge headline (pred_rushing_yards),
    the QB/receiving-comparable MAE.
  - QUANTILE-BUCKET calibration is graded against the QUANTILE-MODEL
    distribution columns (q10..q90), not a residual pool -- the structural
    difference from QB/receiving, whose intervals came from a pooled/conditional
    residual distribution. Bucketing is TIE-AWARE: an actual landing exactly on
    a quantile boundary splits 0.5/0.5 across the two adjacent buckets, so the
    rushing zero-mass floor (a literal point mass at a boundary, per step 47 /
    the receiving step-31 lesson) does not dump entirely into one bucket.
"""

import argparse

import numpy as np
import pandas as pd

from src.ingestion._common import PROJECT_ROOT
from src.models.evaluate_receiving import _snap_gsis_ids

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"

PROP_MIN_CARRIES_L8 = 8

# Buckets defined by the row's own q10/q25/q50/q75/q90 quantile columns.
RUSHING_QUANTILE_BUCKETS = [
    "below_q10", "q10-q25", "q25-q50", "q50-q75", "q75-q90", "above_q90",
]
RUSHING_EXPECTED_BUCKET_FREQ = {
    "below_q10": 0.10, "q10-q25": 0.15, "q25-q50": 0.25,
    "q50-q75": 0.25, "q75-q90": 0.15, "above_q90": 0.10,
}


def _bucket_weights_tieaware(actual, q10, q25, q50, q75, q90):
    """Tie-aware bucket weights for `actual` against this row's quantiles.

    Returns a dict {bucket: weight} summing to 1.0. An actual strictly inside a
    band gets weight 1 there; an actual exactly equal to a quantile boundary
    splits 0.5/0.5 across the two adjacent buckets (the zero-mass-floor lesson).
    """
    edges = [q10, q25, q50, q75, q90]
    labels = RUSHING_QUANTILE_BUCKETS
    weights = {b: 0.0 for b in labels}

    for i, e in enumerate(edges):
        if actual == e:
            weights[labels[i]] += 0.5      # band below the boundary
            weights[labels[i + 1]] += 0.5  # band above the boundary
            return weights

    if actual < edges[0]:
        weights[labels[0]] = 1.0
    elif actual < edges[1]:
        weights[labels[1]] = 1.0
    elif actual < edges[2]:
        weights[labels[2]] = 1.0
    elif actual < edges[3]:
        weights[labels[3]] = 1.0
    elif actual < edges[4]:
        weights[labels[4]] = 1.0
    else:
        weights[labels[5]] = 1.0
    return weights


def grade_week_rushing(season, week, force=False):
    """Grade a saved weekly rushing-yards prediction file against actuals.

    Returns the grades DataFrame and saves it to
    data/predictions/{season}_w{week:02d}_rushing_yards_grades.parquet
    (immutable: raises FileExistsError unless force=True).
    """
    pred_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_rushing_yards.parquet"
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {pred_path}. Run `python -m src.models.predict_rushing "
            f"--season {season} --week {week}` first. (Note: this grades the canonical file, "
            f"not a labeled dry-run such as ..._rushing_yards_dryrun.parquet.)"
        )
    preds = pd.read_parquet(pred_path)

    rushing_matrix = pd.read_parquet(FEATURES_DIR / "rushing_matrix.parquet")
    actuals = rushing_matrix[
        (rushing_matrix["season"] == season) & (rushing_matrix["week"] == week)
    ][["team", "gsis_id", "player", "rushing_yards", "carries_l8"]].rename(columns={
        "player": "actual_player",
        "rushing_yards": "actual_rushing_yards",
        "carries_l8": "actual_carries_l8",
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

    # --- Point grading on the calibrated-Ridge headline ---
    grades["error"] = np.nan
    grades["abs_error"] = np.nan
    grades.loc[graded_mask, "error"] = (
        grades.loc[graded_mask, "pred_rushing_yards"] - grades.loc[graded_mask, "actual_rushing_yards"]
    )
    grades.loc[graded_mask, "abs_error"] = grades.loc[graded_mask, "error"].abs()

    # --- Tie-aware quantile-bucket calibration vs the quantile-model distribution ---
    grades["quantile_bucket"] = None
    bucket_totals = {b: 0.0 for b in RUSHING_QUANTILE_BUCKETS}
    n_tie = 0
    for idx in grades[graded_mask].index:
        row = grades.loc[idx]
        weights = _bucket_weights_tieaware(
            row["actual_rushing_yards"], row["q10"], row["q25"], row["q50"], row["q75"], row["q90"]
        )
        for b, w in weights.items():
            bucket_totals[b] += w
        nonzero = [b for b, w in weights.items() if w > 0]
        if len(nonzero) == 1:
            grades.loc[idx, "quantile_bucket"] = nonzero[0]
        else:
            grades.loc[idx, "quantile_bucket"] = "|".join(nonzero)  # boundary tie
            n_tie += 1

    has_lines = "line" in preds.columns and "prob_over" in preds.columns
    if has_lines:
        grades["over_under_correct"] = pd.NA
        gradeable_line_mask = graded_mask & grades["line"].notna()
        actual_over = grades.loc[gradeable_line_mask, "actual_rushing_yards"] > grades.loc[gradeable_line_mask, "line"]
        pred_over = grades.loc[gradeable_line_mask, "prob_over"] > 0.5
        grades.loc[gradeable_line_mask, "over_under_correct"] = (actual_over == pred_over)

    # --- Summary ---
    n_graded = int(graded_mask.sum())
    n_did_not_play = int((grades["status"] == "DID_NOT_PLAY").sum())
    n_no_stats = int((grades["status"] == "NO_STATS").sum())

    print("Summary:")
    print(f"  rows: {len(grades)} (GRADED={n_graded}, DID_NOT_PLAY={n_did_not_play}, NO_STATS={n_no_stats})")

    avail_denom = n_graded + n_did_not_play
    if avail_denom:
        print(f"  availability accuracy: {n_graded}/{avail_denom} = {n_graded / avail_denom:.3f}")
    else:
        print("  availability accuracy: n/a (no resolved-roster rows)")

    if n_graded:
        mae = grades.loc[graded_mask, "abs_error"].mean()
        bias = grades.loc[graded_mask, "error"].mean()
        print(f"  point MAE (calibrated-Ridge, all GRADED, n={n_graded}): {mae:.2f}")
        print(f"  point bias (all GRADED): {bias:.2f}")

        prop_mask = graded_mask & (grades["actual_carries_l8"] >= PROP_MIN_CARRIES_L8)
        n_prop = int(prop_mask.sum())
        if n_prop:
            prop_mae = grades.loc[prop_mask, "abs_error"].mean()
            prop_bias = grades.loc[prop_mask, "error"].mean()
            print(f"  point MAE (prop-relevant GRADED, carries_l8>={PROP_MIN_CARRIES_L8}, n={n_prop}): {prop_mae:.2f}")
            print(f"  point bias (prop-relevant GRADED): {prop_bias:.2f}")
        else:
            print(f"  prop-relevant slice (carries_l8>={PROP_MIN_CARRIES_L8}): n/a (no rows)")

        print(f"\n  Quantile-bucket calibration (tie-aware, vs quantile-model distribution; "
              f"{n_tie} boundary tie(s) split):")
        print(f"  {'bucket':<10} {'observed':>10} {'expected':>10} {'weight':>8}")
        for bucket in RUSHING_QUANTILE_BUCKETS:
            observed = bucket_totals[bucket] / n_graded
            expected = RUSHING_EXPECTED_BUCKET_FREQ[bucket]
            print(f"  {bucket:<10} {observed * 100:>9.1f}% {expected * 100:>9.1f}% {bucket_totals[bucket]:>8.1f}")
    else:
        print("  point MAE/bias/quantile calibration: n/a (no graded rows)")

    if has_lines and n_graded:
        gradeable = grades.loc[graded_mask & grades["line"].notna()]
        if len(gradeable):
            n_correct = int(gradeable["over_under_correct"].sum())
            print(f"\n  over/under accuracy: {n_correct}/{len(gradeable)} = {n_correct / len(gradeable):.3f}")
        else:
            print("\n  over/under accuracy: n/a (no rows with a line)")

    # --- Save grades ---
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_rushing_yards_grades.parquet"
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
    parser = argparse.ArgumentParser(description="Grade a weekly RB rushing-yards prediction file.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing grades file")
    args = parser.parse_args()

    grade_week_rushing(args.season, args.week, force=args.force)


if __name__ == "__main__":
    main()
