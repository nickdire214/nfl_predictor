"""Feature engineering module.

Transforms raw ingested data into model-ready feature sets, persisted
to data/features. This module currently builds the QB base table:
joins and game-context features only, no rolling windows yet.
"""

import numpy as np
import pandas as pd

from src.ingestion._common import PROJECT_ROOT

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


def _build_team_game_view(schedules):
    cols = [
        "game_id", "season", "week", "game_type", "gameday",
        "div_game", "roof", "surface", "temp", "wind",
        "spread_line", "total_line",
    ]

    home = schedules[cols + ["home_team", "away_team", "home_rest", "away_rest", "home_qb_id"]].copy()
    home = home.rename(columns={
        "home_team": "team",
        "away_team": "opponent",
        "home_rest": "team_rest",
        "away_rest": "opponent_rest",
        "home_qb_id": "qb_id",
    })
    home["is_home"] = 1

    away = schedules[cols + ["away_team", "home_team", "away_rest", "home_rest", "away_qb_id"]].copy()
    away = away.rename(columns={
        "away_team": "team",
        "home_team": "opponent",
        "away_rest": "team_rest",
        "home_rest": "opponent_rest",
        "away_qb_id": "qb_id",
    })
    away["is_home"] = 0

    return pd.concat([home, away], ignore_index=True)


def _verify_spread_convention(schedules, verbose=True):
    lopsided = schedules[schedules["spread_line"].abs() >= 10].head(4)

    if verbose:
        print("Spread sign convention check (abs(spread_line) >= 10):")
        for _, row in lopsided.iterrows():
            home_implied = (row["total_line"] + row["spread_line"]) / 2
            away_implied = (row["total_line"] - row["spread_line"]) / 2
            favorite = "home" if row["home_moneyline"] < row["away_moneyline"] else "away"
            print(
                f"  {row['away_team']} @ {row['home_team']}: spread_line={row['spread_line']}, "
                f"total_line={row['total_line']}, home_implied={home_implied}, away_implied={away_implied}, "
                f"moneyline favorite={favorite}"
            )

    flip = False
    for _, row in lopsided.iterrows():
        home_implied = (row["total_line"] + row["spread_line"]) / 2
        away_implied = (row["total_line"] - row["spread_line"]) / 2
        favorite = "home" if row["home_moneyline"] < row["away_moneyline"] else "away"
        favorite_implied = home_implied if favorite == "home" else away_implied
        other_implied = away_implied if favorite == "home" else home_implied
        if favorite_implied <= other_implied:
            flip = True

    if verbose:
        if flip:
            print("  -> Sign convention appears BACKWARDS. Flipping spread_line sign for implied totals.")
        else:
            print("  -> Sign convention confirmed: positive spread_line = home favored. No flip needed.")

    return flip


def build_qb_base_table(verbose=True):
    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")

    flip = _verify_spread_convention(schedules, verbose=verbose)
    spread_sign = -1 if flip else 1

    team_games = _build_team_game_view(schedules)

    team_games["team_implied_total"] = team_games.apply(
        lambda r: (r["total_line"] + spread_sign * r["spread_line"]) / 2 if r["is_home"] == 1
        else (r["total_line"] - spread_sign * r["spread_line"]) / 2,
        axis=1,
    )

    label_cols = [
        "passing_yards", "attempts", "completions", "passing_tds",
        "passing_interceptions", "passing_epa", "passing_cpoe", "sacks_suffered",
    ]
    qb_stats = player_stats[player_stats["position"] == "QB"]

    # player_stats.game_id is null for all of 2021 and 2024 (and a few stragglers
    # in 2022/2025), so join on (season, week, team, player_id) instead, which is
    # complete and unique in both datasets.
    merged = team_games.merge(
        qb_stats[["season", "week", "team", "player_id", "player_display_name"] + label_cols],
        left_on=["season", "week", "team", "qb_id"],
        right_on=["season", "week", "team", "player_id"],
        how="left",
    )

    if verbose:
        # Diagnostics
        print(f"\nFinal shape: {merged.shape}")

        print("\nRows per season:")
        print(merged.groupby("season").size().to_string())

        print("\nStarts per QB (top 10 by total starts), broken down by season:")
        starts = merged.dropna(subset=["qb_id"])
        name_lookup = qb_stats.drop_duplicates("player_id").set_index("player_id")["player_display_name"]
        starts = starts.copy()
        starts["qb_name"] = starts["qb_id"].map(name_lookup).fillna(starts["qb_id"])

        top10_ids = starts["qb_id"].value_counts().head(10).index
        pivot = starts[starts["qb_id"].isin(top10_ids)].pivot_table(
            index="qb_name", columns="season", values="game_id", aggfunc="count", fill_value=0
        )
        pivot["total"] = pivot.sum(axis=1)
        pivot = pivot.sort_values("total", ascending=False)
        print(pivot.to_string())

        # Excluded player_stats QB rows: backups/relievers not matching the starter qb_id
        matched = merged.dropna(subset=["player_id"])
        matched_keys = set(zip(matched["season"], matched["week"], matched["team"], matched["player_id"]))
        qb_stats_keys = set(zip(qb_stats["season"], qb_stats["week"], qb_stats["team"], qb_stats["player_id"]))
        excluded = qb_stats_keys - matched_keys
        print(f"\nExcluded player_stats QB rows (non-starter appearances): {len(excluded)}")

    merged = merged.drop(columns=["player_id"])
    return merged


STAT_COLS = [
    "passing_yards", "attempts", "completions", "passing_tds",
    "passing_interceptions", "passing_epa", "passing_cpoe", "sacks_suffered",
]


def _compute_rolling_features(df):
    df = df.sort_values(["qb_id", "season", "week"]).reset_index(drop=True)

    for col in STAT_COLS:
        # l3/l8 windows cross season boundaries: shift within qb_id only.
        shifted = df.groupby("qb_id")[col].shift(1)

        df[f"{col}_l3"] = (
            shifted.groupby(df["qb_id"]).transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )
        df[f"{col}_l8"] = (
            shifted.groupby(df["qb_id"]).transform(lambda s: s.rolling(window=8, min_periods=1).mean())
        )

        # Season-to-date resets each season: shift within (qb_id, season), so
        # the first game of a season has no carryover from the prior season.
        shifted_season = df.groupby(["qb_id", "season"])[col].shift(1)
        df[f"{col}_std"] = (
            shifted_season.groupby([df["qb_id"], df["season"]]).transform(lambda s: s.expanding().mean())
        )

    df["games_into_season"] = df.groupby(["qb_id", "season"]).cumcount()

    return df


def add_rolling_features(df):
    df = df.dropna(subset=["passing_yards"]).copy()
    return _compute_rolling_features(df)


def build_prediction_features(season, week, starters=None, line_overrides=None):
    """Build feature rows for an upcoming week's games.

    Uses the same base-table and rolling-feature machinery as the
    training path (build_qb_base_table / _compute_rolling_features):
    target-week rows are constructed from schedules with all stat/label
    columns set to NaN, concatenated with the historical base table
    restricted to games strictly before (season, week), and run through
    _compute_rolling_features. Only the target-week rows are returned.

    starters: optional dict {team: gsis_id} overriding the QB for that
    team. Teams not in starters fall back to schedules' home_qb_id /
    away_qb_id columns.

    line_overrides: optional dict {team: (spread_line, total_line)} for
    when schedules hasn't populated lines for future games yet. Live use
    may need to source these lines from the Odds API instead.
    """
    starters = starters or {}
    line_overrides = line_overrides or {}

    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    flip = _verify_spread_convention(schedules, verbose=False)
    spread_sign = -1 if flip else 1

    historical = build_qb_base_table(verbose=False)
    historical = historical.dropna(subset=["passing_yards"]).copy()
    historical = historical[
        (historical["season"] < season)
        | ((historical["season"] == season) & (historical["week"] < week))
    ]

    target_sched = schedules[(schedules["season"] == season) & (schedules["week"] == week)]
    target_games = _build_team_game_view(target_sched)

    for team, qb_id in starters.items():
        target_games.loc[target_games["team"] == team, "qb_id"] = qb_id

    for team, (spread_line, total_line) in line_overrides.items():
        target_games.loc[target_games["team"] == team, "spread_line"] = spread_line
        target_games.loc[target_games["team"] == team, "total_line"] = total_line

    target_games["team_implied_total"] = np.where(
        target_games["is_home"] == 1,
        (target_games["total_line"] + spread_sign * target_games["spread_line"]) / 2,
        (target_games["total_line"] - spread_sign * target_games["spread_line"]) / 2,
    )

    target_games["player_display_name"] = None
    for col in STAT_COLS:
        target_games[col] = float("nan")

    combined = pd.concat([historical, target_games], ignore_index=True, sort=False)
    combined = _compute_rolling_features(combined)

    return combined[(combined["season"] == season) & (combined["week"] == week)]


def main():
    df = build_qb_base_table()
    df = add_rolling_features(df)

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / "qb_matrix.parquet"
    df.to_parquet(path)
    print(f"\nFinal shape: {df.shape}")
    print(f"Column count: {len(df.columns)}")
    print(f"Saved to: {path}")


if __name__ == "__main__":
    main()
