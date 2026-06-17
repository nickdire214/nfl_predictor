"""Receiving base table module.

Builds the player-game spine for WR/TE/RB receiving production:
snap_counts (2021-2025) filtered to skill-position players who took an
offensive snap, mapped to gsis_id via the players crosswalk, joined with
player_stats receiving columns and the same schedule/team-game context
used by the QB base table. No rolling features yet.
"""

import numpy as np
import pandas as pd

from src.features.engineer import _build_team_game_view, _verify_spread_convention
from src.features.rolling import add_rolling_features
from src.ingestion._common import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"

RECEIVING_COUNT_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_tds", "receiving_air_yards",
]
RECEIVING_SHARE_COLS = ["target_share", "air_yards_share", "wopr", "racr"]

# Zero-filled for played-but-uninvolved rows (racr excluded: stays NaN).
RECEIVING_ZERO_FILL_SHARE_COLS = ["target_share", "air_yards_share", "wopr"]

RECEIVING_ROLLING_COLS = [
    "targets", "receptions", "receiving_yards", "receiving_air_yards",
    "offense_pct", "target_share", "air_yards_share", "wopr",
]

SPINE_POSITIONS = ["WR", "TE", "RB"]


def _canonical_position_map(players):
    """gsis_id -> canonical position from players.parquet.

    players.parquet has one row per gsis_id with a stable position label,
    unlike snap_counts whose per-week position can flip on upstream
    mislabels (e.g. Darren Waller tagged WR in 2025 wk13, TE otherwise).
    """
    valid = players.dropna(subset=["gsis_id"])
    return valid.drop_duplicates("gsis_id").set_index("gsis_id")["position"]


def _build_crosswalk(players, verbose=True):
    crosswalk = players[["gsis_id", "pfr_id"]].dropna()

    if verbose:
        active = players[
            players["position"].isin(SPINE_POSITIONS) & (players["last_season"] >= 2021)
        ]
        missing = active["pfr_id"].isna().sum()
        print(
            f"WR/TE/RB players active 2021+: {len(active)}, "
            f"missing pfr_id: {missing}"
        )

    return crosswalk


def _build_player_game_spine(snap_counts, crosswalk, verbose=True):
    spine = snap_counts[
        snap_counts["position"].isin(SPINE_POSITIONS) & (snap_counts["offense_snaps"] > 0)
    ].copy()

    merged = spine.merge(
        crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left"
    )

    unmatched = merged[merged["gsis_id"].isna()]
    rate = len(unmatched) / len(merged)
    if verbose:
        print(f"Snap rows: {len(merged)}, unmatched to gsis_id: {len(unmatched)} ({rate:.4%})")
        if rate > 0.02:
            print("Unmatched examples:")
            print(
                unmatched[["season", "week", "team", "player", "pfr_player_id", "position"]]
                .drop_duplicates(subset=["player", "pfr_player_id"])
                .head(10)
                .to_string(index=False)
            )

    matched = merged.dropna(subset=["gsis_id"]).drop(columns=["pfr_id"])
    return matched


def build_receiving_base(verbose=True):
    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    snap_counts = pd.read_parquet(RAW_DATA_DIR / "snap_counts.parquet")
    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")

    crosswalk = _build_crosswalk(players, verbose=verbose)
    spine = _build_player_game_spine(snap_counts, crosswalk, verbose=verbose)

    spine = spine[
        ["season", "week", "team", "opponent", "gsis_id", "player", "position",
         "offense_snaps", "offense_pct"]
    ].copy()

    # Canonical position from players.parquet. The snap-table position decides
    # spine inclusion (offensive snap as WR/TE/RB) but its per-week label flips
    # on upstream mislabels, so the position value used downstream comes from
    # the crosswalk; snap label is kept only as a fallback for gsis_ids absent
    # from players.parquet.
    canon = _canonical_position_map(players)
    snap_position = spine["position"]
    mapped = spine["gsis_id"].map(canon)
    fallback_mask = mapped.isna()
    spine["position"] = mapped.fillna(snap_position)

    if verbose:
        changed = ((spine["position"] != snap_position) & ~fallback_mask).sum()
        print(
            f"Canonical position: {fallback_mask.sum()} rows fell back to snap label "
            f"({fallback_mask.sum() / len(spine):.4%}); "
            f"{changed} (player, week) rows reclassified vs snap label"
        )
        print("Per-position rows (snap-sourced -> canonical):")
        before = snap_position.value_counts()
        after = spine["position"].value_counts()
        for pos in sorted(set(before.index) | set(after.index)):
            print(f"  {pos:<4} {before.get(pos, 0):>6} -> {after.get(pos, 0):>6}")

    stat_cols = RECEIVING_COUNT_COLS + RECEIVING_SHARE_COLS
    merged = spine.merge(
        player_stats[["season", "week", "team", "player_id"] + stat_cols],
        left_on=["season", "week", "team", "gsis_id"],
        right_on=["season", "week", "team", "player_id"],
        how="left",
    )

    played_uninvolved = merged["player_id"].isna()
    if verbose:
        print(f"Snap rows with no player_stats row (played but uninvolved): {played_uninvolved.sum()}")

    for col in RECEIVING_COUNT_COLS:
        merged[col] = merged[col].fillna(0)

    for col in RECEIVING_ZERO_FILL_SHARE_COLS:
        merged.loc[played_uninvolved, col] = merged.loc[played_uninvolved, col].fillna(0)

    merged = merged.drop(columns=["player_id"])

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

        print("\nRows per position:")
        print(merged.groupby("position").size().to_string())

        print("\nTop 10 players by total receiving yards 2021-2025:")
        totals = merged.groupby(["gsis_id", "player"]).agg(
            games=("season", "size"),
            total_receiving_yards=("receiving_yards", "sum"),
        ).reset_index()
        top10 = totals.sort_values("total_receiving_yards", ascending=False).head(10)
        print(top10[["player", "games", "total_receiving_yards"]].to_string(index=False))

        print("\nSpot check: Ja'Marr Chase 2025, week by week:")
        chase = merged[(merged["player"] == "Ja'Marr Chase") & (merged["season"] == 2025)]
        chase = chase.sort_values("week")
        print(chase[["week", "offense_snaps", "offense_pct", "targets", "receiving_yards"]].to_string(index=False))

    return merged


def build_receiving_matrix(verbose=True):
    df = build_receiving_base(verbose=verbose)
    return add_rolling_features(df, group_col="gsis_id", stat_cols=RECEIVING_ROLLING_COLS)


def build_receiving_prediction_features(season, week, line_overrides=None):
    """Build feature rows for an upcoming week's receiving props.

    Mirrors src.features.engineer.build_prediction_features: target-week
    rows are constructed from schedules with all stat/label columns set to
    NaN, concatenated with the historical receiving base restricted to
    games strictly before (season, week), and run through the shared
    roller (add_rolling_features, grouped by gsis_id). Only the
    target-week rows are returned.

    Player population for the target week: for each team scheduled in
    (season, week), the full roster of WR/TE/RB pass-catchers from that
    team's most recent prior game in the receiving spine (cross-season
    allowed) -- the same "most recent prior" logic
    src.models.predict.resolve_starters uses for the QB starter, applied
    to a team's whole recent roster rather than a single player.

    Snap-derived features (offense_pct rolling) come from history via the
    roller; the target week's own snaps are unknown and stay NaN, feeding
    into _l3/_l8/_std only through the shift (never the current row).

    Position is sourced from build_receiving_base, which now resolves it
    canonically from players.parquet, so this path and the training/matrix
    path label position identically (no per-week snap-label flips).

    Adds an `as_of` metadata column ("{season} wk{week:02d}") recording the
    game each player's roster row was drawn from -- the receiving analogue of
    the QB starter as_of, for roster-staleness review. It is metadata only
    (not a model feature) and is NaN on the historical rows.

    line_overrides: optional dict {team: (spread_line, total_line)} for
    when schedules hasn't populated lines for future games yet.
    """
    line_overrides = line_overrides or {}

    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    flip = _verify_spread_convention(schedules, verbose=False)
    spread_sign = -1 if flip else 1

    historical = build_receiving_base(verbose=False)
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

    rosters = []
    for team in target_games["team"].unique():
        team_hist = historical[historical["team"] == team]
        if team_hist.empty:
            continue
        last = team_hist[["season", "week"]].drop_duplicates().sort_values(["season", "week"]).iloc[-1]
        last_season, last_week = int(last["season"]), int(last["week"])
        roster = team_hist[
            (team_hist["season"] == last_season) & (team_hist["week"] == last_week)
        ][["gsis_id", "player", "position"]].drop_duplicates().copy()
        roster["team"] = team
        # as_of: the game each player's roster row was drawn from (metadata only,
        # not a model feature). Zero-padded week so the string sorts chronologically.
        roster["as_of"] = f"{last_season} wk{last_week:02d}"
        rosters.append(roster)

    if rosters:
        roster_df = pd.concat(rosters, ignore_index=True)
    else:
        roster_df = pd.DataFrame(columns=["gsis_id", "player", "position", "team", "as_of"])

    target_rows = roster_df.merge(target_games, on="team", how="left")

    for col in ["offense_snaps", "offense_pct"] + RECEIVING_COUNT_COLS + RECEIVING_SHARE_COLS:
        target_rows[col] = float("nan")

    combined = pd.concat([historical, target_rows], ignore_index=True, sort=False)
    combined = add_rolling_features(combined, group_col="gsis_id", stat_cols=RECEIVING_ROLLING_COLS)

    return combined[(combined["season"] == season) & (combined["week"] == week)]


def main():
    base_df = build_receiving_base()

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    base_path = FEATURES_DIR / "receiving_base.parquet"
    base_df.to_parquet(base_path)
    print(f"\nSaved to: {base_path}")

    matrix_df = add_rolling_features(base_df, group_col="gsis_id", stat_cols=RECEIVING_ROLLING_COLS)
    matrix_path = FEATURES_DIR / "receiving_matrix.parquet"
    matrix_df.to_parquet(matrix_path)
    print(f"\nReceiving matrix shape: {matrix_df.shape}, columns: {len(matrix_df.columns)}")
    print(f"Saved to: {matrix_path}")


if __name__ == "__main__":
    main()
