"""Probability layer and weekly prediction runner for the QB passing yards model.

Builds an empirical predictive distribution for a calibrated point
prediction by adding a pooled, mean-centered out-of-sample residual
distribution (see scripts/build_residual_pool.py) to the prediction.

Also implements the weekly runner (run_week / resolve_starters) that
trains the standard pipeline on history, replays a RollingCalibrator
over the current season's completed weeks, builds prediction-time
features, and writes an immutable per-week prediction log.
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.engineer import build_prediction_features
from src.ingestion._common import PROJECT_ROOT
from src.models.calibrate import RollingCalibrator
from src.models.train import FEATURE_COLS, TARGET, preprocess_qb_matrix

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
MODELS_DIR = PROJECT_ROOT / "models"
RESIDUAL_POOL_PATH = MODELS_DIR / "qb_residual_pool.npy"
N0 = 150
QUANTILE_LEVELS = (0.1, 0.25, 0.5, 0.75, 0.9)

# How many of a team's most recent games vote on its default starting QB.
#
# REVERTED TO 1 (step 67 follow-up). N=4 was tried and rejected on evidence:
# it changed only 2 of 32 teams on the 2026 wk1 board (one of them a
# regression to a QB who is out of the league), and it lagged every measured
# mid-season starter change by an extra game with worse post-switch accuracy.
# The 8-team week-1 mismatch is NOT a resolution-rule defect -- it is an
# offseason-information gap that no rule derived from 2025 data can close --
# and it is handled by starters_override.csv. N=1 is exactly the
# most-recent-prior-start rule; the modal machinery below degenerates to it.
STARTER_WINDOW_GAMES = 1

# starters_override.csv player_name value meaning "this team's starter is
# genuinely undecided — drop the team rather than guess" (step 69A).
# Matched case-insensitively.
SKIP_SENTINEL = "SKIP"


def load_residual_pool(path=RESIDUAL_POOL_PATH):
    """Load the pooled, mean-centered out-of-sample residuals."""
    return np.load(Path(path))


def prob_over(calibrated_pred, line, residual_pool):
    """Probability that the realized outcome is strictly above `line`.

    Computed as the fraction of (calibrated_pred + residual_pool) that
    is strictly greater than `line`. `calibrated_pred` may be a scalar
    or an array; `line` is broadcast against it.
    """
    calibrated_pred = np.atleast_1d(calibrated_pred).astype(float)
    line = np.atleast_1d(line).astype(float)

    samples = calibrated_pred[:, None] + residual_pool[None, :]
    probs = np.mean(samples > line[:, None], axis=1)

    if probs.shape[0] == 1:
        return probs[0]
    return probs


def predict_quantiles(calibrated_pred, residual_pool, levels=(0.1, 0.25, 0.5, 0.75, 0.9)):
    """Predicted outcome quantiles: calibrated_pred + residual-pool quantiles.

    `calibrated_pred` may be a scalar or an array. Returns an array of
    shape (len(levels),) for a scalar prediction, or (n, len(levels))
    for an array of n predictions.
    """
    pool_quantiles = np.quantile(residual_pool, levels)

    calibrated_pred = np.asarray(calibrated_pred, dtype=float)
    if calibrated_pred.ndim == 0:
        return calibrated_pred + pool_quantiles

    return calibrated_pred[:, None] + pool_quantiles[None, :]


def resolve_starters(season, week):
    """Resolve each team's starting QB for (season, week).

    Default: each team's MOST FREQUENT starter across its last
    STARTER_WINDOW_GAMES prior games, ties broken by most recent start.
    History comes from qb_matrix.parquet strictly before (season, week);
    cross-season carryover is allowed (week-1 defaults come from the prior
    season's closing games).

    STARTER_WINDOW_GAMES is 1, so in practice this is the most-recent-prior-
    start rule and the modal machinery degenerates to it. A 4-game window was
    tried in step 67 and reverted on evidence (see DECISIONS): it did not fix
    the week-1 mismatch, because the 2026 cases are sustained multi-game
    handovers rather than one-off rest starts, and it lagged genuine
    mid-season changes by an extra game.

    The week-1 mismatch is an OFFSEASON-INFORMATION gap, not a rule defect:
    who starts in September is not derivable from 2025 data by any rule.
    Correct it with starters_override.csv, which for week 1 is expected to be
    populated rather than empty.

    The summary's `window_starts` column reports how contested the window was
    (at N=1 it is always "1/1", which is itself the signal that the default
    rests on a single game).

    Overrides: if data/predictions/starters_override.csv exists, rows
    matching (season, week) override the default starter for that team.
    Override `player_name` values are resolved against players.parquet
    (position == "QB") by exact display_name match; an unknown or
    ambiguous name raises a ValueError listing the candidates, and a name
    with NO qb_matrix history before (season, week) also raises rather than
    producing all-NaN rolling features (step 69B).

    player_name = "SKIP" (case-insensitive) drops the team from the board:
    it is omitted from the returned `starters` dict but appears in `summary`
    with source="skip" and a null qb_id, so a short board is visible rather
    than silent. SKIP is the intended handling for a genuinely undecided
    starting job — the choice between two plausible QBs turns on front-office
    decisions (contract money, a veteran's decline, commitment to a young
    starter) that are not derivable from passing statistics.

    Returns (starters, summary): `starters` is {team: gsis_id};
    `summary` is a printable DataFrame with one row per team showing
    qb_id, qb_name, and whether it came from "default" or "override".
    """
    qb_matrix = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
    history = qb_matrix[
        (qb_matrix["season"] < season) | ((qb_matrix["season"] == season) & (qb_matrix["week"] < week))
    ].dropna(subset=["qb_id"])

    starters = {}
    summary_rows = []

    for team in sorted(history["team"].unique()):
        team_hist = history[history["team"] == team].sort_values(["season", "week"])
        window = team_hist.tail(STARTER_WINDOW_GAMES)

        # modal starter in the window; ties -> the one who started most recently
        counts = window.groupby("qb_id").size()
        top = counts.max()
        tied = set(counts[counts == top].index)
        # window is chronological, so the last tied appearance is the most recent
        chosen = window[window["qb_id"].isin(tied)].iloc[-1]["qb_id"]

        # as_of: that QB's most recent start inside the window
        last_start = window[window["qb_id"] == chosen].iloc[-1]

        starters[team] = chosen
        summary_rows.append({
            "team": team,
            "qb_id": chosen,
            "qb_name": last_start["player_display_name"],
            "source": "default",
            "as_of": f"{int(last_start['season'])} wk{int(last_start['week'])}",
            "window_starts": f"{int(counts[chosen])}/{len(window)}",
        })

    override_path = PREDICTIONS_DIR / "starters_override.csv"
    if override_path.exists():
        overrides = pd.read_csv(override_path)
        overrides = overrides[(overrides["season"] == season) & (overrides["week"] == week)]

        if not overrides.empty:
            players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
            qbs = players[players["position"] == "QB"]

            for _, row in overrides.iterrows():
                team, name = row["team"], row["player_name"]

                # SKIP sentinel (step 69A): the job is genuinely undecided, so
                # drop the team rather than guess. The team is omitted from the
                # returned dict entirely but stays VISIBLE in the summary with
                # source="skip", so a short board is obvious rather than silent.
                if str(name).strip().upper() == SKIP_SENTINEL:
                    starters.pop(team, None)
                    summary_rows.append({
                        "team": team,
                        "qb_id": None,
                        "qb_name": None,
                        "source": "skip",
                        "as_of": f"{season} wk{week}",
                        "window_starts": "skip",
                    })
                    continue

                matches = qbs[qbs["display_name"] == name]

                if matches.empty:
                    raise ValueError(
                        f"starters_override.csv: unknown player_name '{name}' for team {team} "
                        f"(no QB in players.parquet matches this display_name)"
                    )

                gsis_ids = matches["gsis_id"].unique()
                if len(gsis_ids) > 1:
                    candidates = matches[["gsis_id", "display_name", "rookie_season", "last_season"]]
                    raise ValueError(
                        f"starters_override.csv: ambiguous player_name '{name}' for team {team} "
                        f"— candidates:\n{candidates.to_string(index=False)}"
                    )

                gsis_id = gsis_ids[0]

                # No-history guard (step 69B), mirroring _apply_roster_override:
                # a QB with no prior qb_matrix rows would produce all-NaN rolling
                # features and be dropped silently in preprocessing. Raise instead.
                if history[history["qb_id"] == gsis_id].empty:
                    info = matches.iloc[0]
                    raise ValueError(
                        f"starters_override.csv: '{name}' ({team}) has NO qb_matrix history "
                        f"before {season} wk{week}, so every rolling feature would be NaN and "
                        f"he cannot be predicted. "
                        f"(rookie_season={info['rookie_season']}, last_season={info['last_season']}, "
                        f"latest_team={info['latest_team']}.) "
                        f"If this really is the starter and he has no NFL history, use "
                        f"player_name={SKIP_SENTINEL} to drop {team} from the board instead."
                    )

                starters[team] = gsis_id
                summary_rows.append({
                    "team": team,
                    "qb_id": gsis_id,
                    "qb_name": name,
                    "source": "override",
                    "as_of": f"{season} wk{week}",
                    "window_starts": "override",
                })

    summary = pd.DataFrame(summary_rows)
    summary = (
        summary.sort_values("source")
        .drop_duplicates(subset="team", keep="last")
        .sort_values("team")
        .reset_index(drop=True)
    )

    return starters, summary


def run_week(season, week, lines=None, force=False, line_overrides=None, label=None):
    """Generate and save QB passing-yards predictions for (season, week).

    1. Trains the standard Ridge pipeline on all rows strictly before
       (season, week) — this is the pipeline used to predict the target week.
    2. Separately, trains a pipeline on rows strictly before `season` and
       replays the current season's completed weeks (1..week-1) through a
       fresh RollingCalibrator(n0=N0), one week at a time, accumulating
       prediction errors — same walk-forward pattern as
       scripts/build_residual_pool.py. The resulting calibrator offset is
       applied to the target week's predictions.
    3. Builds target-week features via build_prediction_features using
       resolve_starters, predicts, calibrates, and attaches residual-pool
       quantiles (and prob_over if `lines` is given).
    4. Saves the result to
       data/predictions/{season}_w{week:02d}_qb_pass_yards.parquet.
       Raises FileExistsError if that file already exists, unless
       force=True (a warning is printed when forcing).

    `lines`: optional dict {team: line} of sportsbook passing-yards lines.

    `line_overrides`: optional dict {team: (spread_line, total_line)},
    passed through to build_prediction_features.

    `label`: optional string; if set, the output filename becomes
    data/predictions/{season}_w{week:02d}_qb_pass_yards_{label}.parquet,
    for non-canonical/test runs that mustn't collide with real weekly logs.
    """
    qb_matrix = pd.read_parquet(FEATURES_DIR / "qb_matrix.parquet")
    df_processed = preprocess_qb_matrix(qb_matrix)

    # --- Calibrator replay over the current season's completed weeks ---
    calibrator = RollingCalibrator(n0=N0)
    history_train = df_processed[df_processed["season"] < season]
    season_data = df_processed[df_processed["season"] == season]

    if not history_train.empty:
        history_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        history_pipeline.fit(history_train[FEATURE_COLS], history_train[TARGET])

        for w in range(1, week):
            week_data = season_data[season_data["week"] == w]
            if week_data.empty:
                continue
            raw_pred = history_pipeline.predict(week_data[FEATURE_COLS])
            calibrator.update(raw_pred, week_data[TARGET].values)

    offset = calibrator.offset

    # --- Final pipeline: trained on all rows strictly before (season, week) ---
    final_train = df_processed[
        (df_processed["season"] < season) | ((df_processed["season"] == season) & (df_processed["week"] < week))
    ]
    final_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])
    final_pipeline.fit(final_train[FEATURE_COLS], final_train[TARGET])

    # --- Target-week features ---
    starters, starter_summary = resolve_starters(season, week)
    print("Starter resolution:")
    print(starter_summary.to_string(index=False))

    # SKIPped teams are read off the summary (source == "skip") rather than from a
    # third return value, so resolve_starters' 2-tuple contract is unchanged for
    # existing callers. They must be passed explicitly: omitting a team from
    # `starters` alone is NOT enough to keep it off the board (step 70).
    skip_teams = starter_summary.loc[starter_summary["source"] == "skip", "team"].tolist()

    features = build_prediction_features(
        season, week, starters=starters, line_overrides=line_overrides, skip_teams=skip_teams
    )
    features = preprocess_qb_matrix(features)

    raw_pred = final_pipeline.predict(features[FEATURE_COLS])
    calibrated_pred = calibrator.calibrate(raw_pred)

    residual_pool = load_residual_pool()
    quantiles = predict_quantiles(calibrated_pred, residual_pool, levels=QUANTILE_LEVELS)

    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    name_lookup = players.drop_duplicates("gsis_id").set_index("gsis_id")["display_name"]

    result = features[["season", "week", "team", "opponent", "qb_id", "team_implied_total"]].reset_index(drop=True).copy()
    result["qb_name"] = result["qb_id"].map(name_lookup).fillna(result["qb_id"])
    result["pred_passing_yards"] = calibrated_pred
    for level, col in zip(QUANTILE_LEVELS, quantiles.T):
        result[f"p{int(round(level * 100))}"] = col
    result["calibrator_offset"] = offset

    if lines:
        result["line"] = result["team"].map(lines)
        result["prob_over"] = np.nan
        has_line = result["line"].notna()
        if has_line.any():
            result.loc[has_line, "prob_over"] = prob_over(
                calibrated_pred[has_line.values], result.loc[has_line, "line"].values, residual_pool
            )

    result["generated_at_utc"] = datetime.now(timezone.utc)
    result = result.sort_values("pred_passing_yards", ascending=False).reset_index(drop=True)

    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_{label}" if label else ""
    out_path = PREDICTIONS_DIR / f"{season}_w{week:02d}_qb_pass_yards{suffix}.parquet"
    if out_path.exists():
        if not force:
            raise FileExistsError(
                f"{out_path} already exists. Prediction logs are immutable; pass force=True to overwrite."
            )
        print(f"WARNING: {out_path} already exists — overwriting (force=True).")

    result.to_parquet(out_path)
    print(f"\nSaved to: {out_path}")

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate weekly QB passing-yards predictions.")
    parser.add_argument("--season", type=int, required=True)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument("--force", action="store_true", help="overwrite an existing prediction log")
    parser.add_argument("--label", default=None, help="suffix for non-canonical/test runs (avoids overwriting real weekly logs)")
    args = parser.parse_args()

    result = run_week(args.season, args.week, force=args.force, label=args.label)
    print("\nPredictions:")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
