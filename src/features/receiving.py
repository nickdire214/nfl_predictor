"""Receiving base table module.

Builds the player-game spine for WR/TE/RB receiving production:
snap_counts (2021-2025) filtered to skill-position players who took an
offensive snap, mapped to gsis_id via the players crosswalk, joined with
player_stats receiving columns and the same schedule/team-game context
used by the QB base table. No rolling features yet.
"""

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
    ]

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
