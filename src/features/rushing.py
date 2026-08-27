"""RB rushing base table module.

Builds the player-game spine for RB rushing production, mirroring
receiving.py's base-table build: snap_counts rows whose CANONICAL position
(from players.parquet, the step-35 fix) is RB and who took an offensive
snap, mapped to gsis_id via the players crosswalk, joined with player_stats
rushing columns and the same schedule/team-game context the other tables
use. No rolling features yet.

carry_share design: player_stats has no pre-computed rushing-usage share
(only target_share / air_yards_share, which are receiving). We derive
carry_share = a player's carries / his team's total RB carries that game,
where the denominator is summed over the canonical-RB spine (i.e. share of
the backfield carries, NOT of all team rushing attempts, so QB scrambles
and WR end-arounds are excluded). A team-week with zero RB carries yields
carry_share 0.0.
"""

import numpy as np
import pandas as pd

from src.features.engineer import _build_team_game_view, _verify_spread_convention
from src.features.receiving import (
    ROSTER_WINDOW_GAMES,
    SNAP_LABEL_SYNONYMS,
    _apply_latest_team,
    _apply_roster_override,
    _audit_snap_position_labels,
    _build_crosswalk,
    _build_window_roster,
    _canonical_position_map,
    _latest_team_is_authoritative,
)
from src.features.rolling import add_rolling_features
from src.ingestion._common import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

# Counting stats: a played-but-no-stat-row RB gets 0 (a dressed back who never
# touched the ball is a real 0-carry, 0-yard outcome).
RUSHING_COUNT_COLS = [
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs", "rushing_fumbles_lost",
]
# rushing_epa is not a counting stat (~16% null even where a stat row exists);
# it stays NaN rather than being zero-filled.
RUSHING_STAT_COLS = RUSHING_COUNT_COLS + ["rushing_epa"]

# Stats fed to the shared roller (grouped by gsis_id). rushing_epa is excluded
# (NaN-heavy, like racr in receiving); rushing_fumbles_lost is excluded as too
# sparse/noisy to roll usefully (a handful of league-wide events per week).
RUSHING_ROLLING_COLS = [
    "carries", "rushing_yards", "rushing_tds", "rushing_first_downs",
    "carry_share", "offense_pct",
]


def _build_rb_spine(snap_counts, players, crosswalk, verbose=True):
    """Canonical-RB player-game spine, gated on CANONICAL position (step 54).

    Candidacy order is inverted relative to the original build: every
    offensive-snap row is joined to the crosswalk and mapped to its canonical
    position from players.parquet FIRST, and the RB gate is applied to that
    canonical value. The per-week snap label gates only rows with no canonical
    position available (no crosswalk match), normalized through
    SNAP_LABEL_SYNONYMS.

    The old order gated on `position == "RB"` before canonical position was
    ever consulted, so the `HB` synonym excluded CIN's entire 2025 backfield
    before the step-35 canonical fix could correct it -- see DECISIONS step 53.
    """
    offensive = snap_counts[snap_counts["offense_snaps"] > 0].copy()

    if verbose:
        _audit_snap_position_labels(offensive, verbose=verbose)

    merged = offensive.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left")

    canon = _canonical_position_map(players)
    mapped = merged["gsis_id"].map(canon)
    snap_label = merged["position"]
    snap_label_norm = snap_label.replace(SNAP_LABEL_SYNONYMS)

    # Canonical position where available, else the normalized snap label.
    effective = mapped.fillna(snap_label_norm)
    candidates = effective == "RB"

    # Join-failure rate over RB CANDIDATES only, so the denominator stays
    # comparable to the pre-step-54 build (not diluted by non-RB rows).
    cand = merged[candidates]
    unmatched = cand["gsis_id"].isna()
    rate = unmatched.mean() if len(cand) else 0.0
    if verbose:
        print(f"RB candidate snap rows: {len(cand)}, unmatched to gsis_id: "
              f"{int(unmatched.sum())} ({rate:.4%})")
        if rate > 0.02:
            print("  FLAG: join-failure rate exceeds ~2%. Unmatched examples:")
            print(
                cand[unmatched][["season", "week", "team", "player", "pfr_player_id"]]
                .drop_duplicates(subset=["player", "pfr_player_id"])
                .head(10)
                .to_string(index=False)
            )

    keep = candidates & merged["gsis_id"].notna()

    if verbose:
        # Delta vs the old snap-label gate (position == "RB", then drop unmatched).
        old_keep = (snap_label == "RB") & merged["gsis_id"].notna()
        added = keep & ~old_keep
        removed = old_keep & ~keep
        print(f"Candidacy inversion: +{int(added.sum())} rows admitted that the snap-label "
              f"gate excluded, -{int(removed.sum())} rows excluded that it admitted "
              f"(net {int(keep.sum()) - int(old_keep.sum()):+d})")
        if added.any():
            print("  admitted by snap label:")
            print(merged[added]["position"].value_counts().rename("rows").to_string())
        if removed.any():
            print("  excluded (snap-labeled RB, non-RB canonical):")
            print(merged[removed].assign(canon=effective[removed])[["player", "canon"]]
                  .drop_duplicates().head(10).to_string(index=False))

    spine = merged[keep].drop(columns=["pfr_id"]).copy()
    spine["position"] = "RB"

    if verbose:
        print(f"canonical-RB spine rows: {len(spine)}")

    return spine


def build_rushing_base(verbose=True):
    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    snap_counts = pd.read_parquet(RAW_DATA_DIR / "snap_counts.parquet")
    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")

    crosswalk = _build_crosswalk(players, verbose=False)
    spine = _build_rb_spine(snap_counts, players, crosswalk, verbose=verbose)

    spine = spine[
        ["season", "week", "team", "opponent", "gsis_id", "player", "position",
         "offense_snaps", "offense_pct"]
    ]

    merged = spine.merge(
        player_stats[["season", "week", "team", "player_id"] + RUSHING_STAT_COLS],
        left_on=["season", "week", "team", "gsis_id"],
        right_on=["season", "week", "team", "player_id"],
        how="left",
    )

    played_no_stat = merged["player_id"].isna()
    if verbose:
        print(f"Spine rows with no player_stats row (dressed, no touch): {int(played_no_stat.sum())}")

    for col in RUSHING_COUNT_COLS:
        merged[col] = merged[col].fillna(0)

    merged = merged.drop(columns=["player_id"])

    # carry_share: share of the team's RB backfield carries that game.
    team_rb_carries = merged.groupby(["season", "week", "team"])["carries"].transform("sum")
    merged["carry_share"] = np.where(team_rb_carries > 0, merged["carries"] / team_rb_carries, 0.0)

    flip = _verify_spread_convention(schedules, verbose=False)
    spread_sign = -1 if flip else 1

    team_games = _build_team_game_view(schedules).drop(columns=["qb_id", "opponent"])
    team_games["team_implied_total"] = team_games.apply(
        lambda r: (r["total_line"] + spread_sign * r["spread_line"]) / 2 if r["is_home"] == 1
        else (r["total_line"] - spread_sign * r["spread_line"]) / 2,
        axis=1,
    )

    merged = merged.merge(team_games, on=["season", "week", "team"], how="left")

    if verbose:
        print(f"\nFinal shape: {merged.shape}")

        print("\nRows per season:")
        print(merged.groupby("season").size().to_string())

        print("\nTop 10 RBs by total rushing yards 2021-2025:")
        totals = merged.groupby(["gsis_id", "player"]).agg(
            games=("season", "size"),
            total_carries=("carries", "sum"),
            total_rushing_yards=("rushing_yards", "sum"),
        ).reset_index()
        top10 = totals.sort_values("total_rushing_yards", ascending=False).head(10)
        print(top10[["player", "games", "total_carries", "total_rushing_yards"]].to_string(index=False))

        print("\nSpot check: Saquon Barkley 2025, week by week:")
        saquon = merged[(merged["player"] == "Saquon Barkley") & (merged["season"] == 2025)].sort_values("week")
        print(saquon[["week", "offense_snaps", "carries", "carry_share", "rushing_yards"]].to_string(index=False))

    return merged


def build_rushing_matrix(verbose=True):
    df = build_rushing_base(verbose=verbose)
    return add_rolling_features(df, group_col="gsis_id", stat_cols=RUSHING_ROLLING_COLS)


def build_rushing_prediction_features(
    season, week, line_overrides=None,
    roster_window=ROSTER_WINDOW_GAMES, apply_latest_team=None, verbose=False,
):
    """Build feature rows for an upcoming week's RB rushing props.

    Mirrors build_receiving_prediction_features: target-week rows are
    constructed with all stat/label columns set to NaN, concatenated with the
    historical rushing base restricted to games strictly before (season, week),
    and run through the shared roller (add_rolling_features grouped by gsis_id
    over RUSHING_ROLLING_COLS). Only the target-week rows are returned --
    identical machinery to the training path, no duplicated rolling logic.

    Player population (step 61): every canonical RB with an offensive snap for
    that team across its last `roster_window` prior games (default
    ROSTER_WINDOW_GAMES=4, cross-season allowed), then -- when latest_team is
    authoritative for `season` -- reassigned to players.parquet's latest_team
    and filtered by status. Both changes are PREDICTION-TIME ONLY and default
    off for any season the crosswalk does not describe, so historical replay
    is untouched. Pass roster_window=1, apply_latest_team=False to reproduce
    the pre-step-61 board.

    carry_share / offense_pct rolling come from history via the roller; the
    target week's own carries and snaps are unknown and stay NaN, feeding into
    _l3/_l8/_std only through the shift (never the current game).

    Carries an `as_of` metadata column ("{season} wk{week:02d}") recording the
    game each player's roster row was drawn from (the receiving step-41
    addition), for roster-staleness review. Metadata only, not a model feature;
    NaN on the historical rows.

    line_overrides: optional dict {team: (spread_line, total_line)} for when
    schedules hasn't populated lines for future games yet.
    """
    line_overrides = line_overrides or {}

    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    flip = _verify_spread_convention(schedules, verbose=False)
    spread_sign = -1 if flip else 1

    historical = build_rushing_base(verbose=False)
    historical = historical[
        (historical["season"] < season)
        | ((historical["season"] == season) & (historical["week"] < week))
    ]

    target_sched = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    target_games = _build_team_game_view(target_sched).drop(columns=["qb_id"])

    for team, (spread_line, total_line) in line_overrides.items():
        target_games.loc[target_games["team"] == team, "spread_line"] = spread_line
        target_games.loc[target_games["team"] == team, "total_line"] = total_line

    target_games["team_implied_total"] = np.where(
        target_games["is_home"] == 1,
        (target_games["total_line"] + spread_sign * target_games["spread_line"]) / 2,
        (target_games["total_line"] - spread_sign * target_games["spread_line"]) / 2,
    )

    target_teams = target_games["team"].unique()
    roster_df = _build_window_roster(historical, target_teams, window=roster_window)

    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    if apply_latest_team is None:
        apply_latest_team = _latest_team_is_authoritative(players, season)
    if apply_latest_team:
        roster_df = _apply_latest_team(roster_df, players, target_teams, verbose=verbose)

    roster_df = _apply_roster_override(
        roster_df, historical, players, season, week, "rushing", verbose=verbose
    )

    target_rows = roster_df.merge(target_games, on="team", how="left")

    for col in ["offense_snaps", "offense_pct", "carry_share"] + RUSHING_STAT_COLS:
        target_rows[col] = float("nan")

    combined = pd.concat([historical, target_rows], ignore_index=True, sort=False)
    combined = add_rolling_features(combined, group_col="gsis_id", stat_cols=RUSHING_ROLLING_COLS)

    return combined[(combined["season"] == season) & (combined["week"] == week)]


def main():
    base_df = build_rushing_base()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    base_path = FEATURES_DIR / "rushing_base.parquet"
    base_df.to_parquet(base_path)
    print(f"\nSaved to: {base_path}")

    matrix_df = add_rolling_features(base_df, group_col="gsis_id", stat_cols=RUSHING_ROLLING_COLS)
    matrix_path = FEATURES_DIR / "rushing_matrix.parquet"
    matrix_df.to_parquet(matrix_path)
    print(f"\nRushing matrix shape: {matrix_df.shape}, columns: {len(matrix_df.columns)}")
    print(f"Saved to: {matrix_path}")


if __name__ == "__main__":
    main()
