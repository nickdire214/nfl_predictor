"""Feature engineering module.

Transforms raw ingested data into model-ready feature sets, persisted
to data/features. This module currently builds the QB base table:
joins and game-context features only, no rolling windows yet.
"""

import numpy as np
import pandas as pd

from src.features.rolling import add_rolling_features as _add_rolling_features
from src.ingestion._common import PROJECT_ROOT, current_season

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


def _reconcile_team_games(merged, schedules, verbose=True):
    """Assert every PLAYED team-game produced a qb_matrix row.

    `build_qb_base_table` takes `qb_id` from schedules' designated starter
    (home_qb_id / away_qb_id), merges it against player_stats, and the caller
    later drops rows with a null `passing_yards`. When the designation does not
    match anyone who threw for that team, the entire team-game disappears with
    no error -- which is how 2026 wk2 ATL vanished (designated Tua Tagovailoa,
    actual passers Cooper Rush 17 att and Jack Strand 15 att).

    The step-93 diagnostic found 50 such team-games across 2021/2022/2024/2025/
    2026, from TWO causes:

      A. (5) The `position == "QB"` filter in build_qb_base_table discards a
         real starter listed at another position. Taysom Hill is `TE` in
         player_stats in every season, so New Orleans' 2021 starts under him
         were invisible to the merge even though his stats row was present.
         This one is ours, not nflverse's.

      B. (45) nflverse's designated starter never played that week -- the
         designation lags mid-season quarterback changes, sometimes by a month.
         2024 alone loses 33 team-games this way, including five consecutive
         Jayden Daniels starts designated to Marcus Mariota.

    SCOPE, and why it is not a blanket raise. Those 45 are permanent historical
    facts; nflverse will not retroactively correct 2024. A guard that raised on
    any mismatch would therefore fail every build from now until the source data
    changes, which it will not. So the guard raises only for the CURRENT season
    -- where a miss is actionable, because it means this week's board is about
    to be built on a hole -- and warns for prior seasons, where the miss is the
    documented step-93 residue and nothing can be done about it here.

    The real fix for both causes is to derive the starter from the team's
    leading passer by attempts rather than trusting the designation, which the
    step-93 diagnostic showed recovers 2,912 of 2,912 played team-games. That is
    a modelling change (it redefines "starter" and shifts every downstream
    rolling feature), so it is deliberately a separate step.

    Raises ValueError if any missing team-game is in the current season.
    """
    played = schedules[schedules["home_score"].notna()]
    seasons = set(merged["season"].unique())
    played = played[played["season"].isin(seasons)]

    expected = pd.concat([
        played[["season", "week", "game_type", "home_team",
                "home_qb_id", "home_qb_name"]].rename(columns={
                    "home_team": "team", "home_qb_id": "qb_id",
                    "home_qb_name": "qb_name"}),
        played[["season", "week", "game_type", "away_team",
                "away_qb_id", "away_qb_name"]].rename(columns={
                    "away_team": "team", "away_qb_id": "qb_id",
                    "away_qb_name": "qb_name"}),
    ], ignore_index=True)

    have = set(zip(*(merged.dropna(subset=["passing_yards"])[c]
                     for c in ("season", "week", "team"))))
    expected["_ok"] = [k in have for k in zip(expected["season"],
                                              expected["week"],
                                              expected["team"])]
    missing = expected[~expected["_ok"]].sort_values(["season", "week", "team"])

    if missing.empty:
        if verbose:
            print(f"\nTeam-game reconciliation: OK — all {len(expected):,} played "
                  f"team-games produced a row.")
        return

    # Only read player_stats when there is actually something to explain.
    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    passers = player_stats[player_stats["attempts"] > 0]

    cur = current_season()
    current_missing = missing[missing["season"] == cur]

    print(f"\n{'=' * 78}")
    print(f"TEAM-GAME RECONCILIATION: {len(missing)} played team-game(s) produced NO row")
    print(f"{'=' * 78}")
    print(f"  expected (played team-games): {len(expected):,}")
    print(f"  actual (rows with passing_yards): {len(have):,}")
    print(f"  current season (current_season() = {cur}): {len(current_missing)} missing")
    print(f"  prior seasons: {len(missing) - len(current_missing)} missing\n")

    for _, r in missing.iterrows():
        act = passers[(passers["season"] == r["season"])
                      & (passers["week"] == r["week"])
                      & (passers["team"] == r["team"])].sort_values(
                          "attempts", ascending=False)
        who = "; ".join(
            f"{a['player_display_name']} {int(a['attempts'])}att/"
            f"{int(a['passing_yards'])}yd" for _, a in act.iterrows()
        ) or "NOBODY THREW A PASS"
        marker = "  <-- CURRENT SEASON" if r["season"] == cur else ""
        print(f"  {int(r['season'])} wk{int(r['week']):02d} {r['team']:4s} "
              f"[{r['game_type']}]  designated={r['qb_name']} ({r['qb_id']})"
              f"{marker}")
        print(f"      actually threw: {who}")

    if not current_missing.empty:
        rows = ", ".join(f"{int(r['season'])} wk{int(r['week'])} {r['team']}"
                         for _, r in current_missing.iterrows())
        raise ValueError(
            f"qb_matrix reconciliation FAILED for the current season "
            f"({cur}): {len(current_missing)} played team-game(s) produced no "
            f"row -- {rows}. The designated starter in schedules.parquet did "
            f"not throw for that team, so the merge dropped the team-game "
            f"silently. This blocks the build because a current-season hole "
            f"means this week's board would be built on incomplete data. "
            f"See the table above for who actually threw. Prior-season misses "
            f"are the documented step-93 residue and only warn."
        )

    print(f"\n{'=' * 78}")
    print(f"WARNING: {len(missing)} historical team-game(s) missing — build CONTINUES")
    print(f"{'=' * 78}")
    print("  These are the DOCUMENTED step-93 historical residue, not a new fault:")
    print("    - 5 from the position=='QB' filter discarding Taysom Hill (TE) starts")
    print("    - the rest from stale nflverse starter designations that will not")
    print("      be retroactively corrected.")
    print("  They are expected on every build until the leading-passer derivation")
    print("  replaces the designated-starter join. Do not treat this as new.")
    print(f"  Current season ({cur}) is CLEAN — that is what the raise protects.\n")


def _derive_starters(player_stats):
    """The team's LEADING PASSER by attempts, per (season, week, team).

    Replaces schedules' designated starter (home_qb_id / away_qb_id) as the
    definition of "the quarterback who started this team-game" (step 95).

    WHY. The designation is wrong often enough to matter, and it fails
    silently: the step-93 diagnostic found 50 of 2,912 played team-games
    producing no qb_matrix row at all, from two causes --

      A. The old `position == "QB"` filter discarded real starters listed at
         another position. Taysom Hill is TE in player_stats in every season,
         so New Orleans' 2021 starts under him were invisible even though he
         was both the designated AND the actual starter. This filter is gone.

      B. nflverse's designation lags mid-season quarterback changes, sometimes
         by a month. 2024 alone lost 33 team-games this way.

    Whoever threw the most passes for a team in a game is that team's starting
    quarterback for our purposes. That is observable from the box score, needs
    no external designation, and cannot go stale. It recovers 2,912 of 2,912
    played team-games.

    Ties are broken deterministically -- attempts, then passing yards, then
    player_id -- so a rebuild is reproducible rather than depending on row
    order. `position` is NOT filtered on: the leading passer is the starter
    whatever the roster sheet calls him.

    Returns a frame of (season, week, team, qb_id_derived).
    """
    passers = player_stats[player_stats["attempts"] > 0].copy()
    passers = passers.sort_values(
        ["season", "week", "team", "attempts", "passing_yards", "player_id"],
        ascending=[True, True, True, False, False, True],
    )
    lead = passers.groupby(["season", "week", "team"], as_index=False).head(1)
    return lead[["season", "week", "team", "player_id"]].rename(
        columns={"player_id": "qb_id_derived"})


def _report_designation_agreement(team_games, verbose=True):
    """Cross-check the derived starter against schedules' designation.

    The designation is kept as a DIAGNOSTIC only -- it no longer decides who
    the starter is. This reports how often the two disagree, by season, so the
    upstream data's reliability stays visible instead of being silently
    discarded.
    """
    if not verbose:
        return
    both = team_games[team_games["qb_id_derived"].notna()
                      & team_games["qb_id_designated"].notna()]
    if both.empty:
        return
    both = both.copy()
    both["agree"] = both["qb_id_derived"] == both["qb_id_designated"]

    print("\nDerived starter vs schedules designation (played games only):")
    print(f"  {'season':>8s}{'games':>8s}{'agree':>8s}{'differ':>8s}{'differ %':>10s}")
    for season, g in both.groupby("season"):
        n, agree = len(g), int(g["agree"].sum())
        print(f"  {int(season):>8d}{n:>8d}{agree:>8d}{n - agree:>8d}"
              f"{(n - agree) / n * 100:>9.1f}%")
    n, agree = len(both), int(both["agree"].sum())
    print(f"  {'ALL':>8s}{n:>8d}{agree:>8d}{n - agree:>8d}"
          f"{(n - agree) / n * 100:>9.1f}%")
    print("  (disagreement = nflverse's designated starter was not the leading passer;")
    print("   the derived value wins -- the designation is diagnostic only.)")


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

    # STARTER DEFINITION (step 95): the team's leading passer by attempts, not
    # schedules' designated starter. See _derive_starters for why. The
    # designation is retained alongside purely as a cross-check.
    team_games = team_games.rename(columns={"qb_id": "qb_id_designated"})
    team_games = team_games.merge(
        _derive_starters(player_stats), on=["season", "week", "team"], how="left")
    _report_designation_agreement(team_games, verbose=verbose)

    # Played games take the derived starter; games with no passer yet (future
    # weeks in the schedule) fall back to the designation so the schedule spine
    # is still usable downstream. Those rows carry no passing_yards and are
    # dropped by the callers' dropna anyway.
    team_games["qb_id"] = team_games["qb_id_derived"].fillna(
        team_games["qb_id_designated"])

    # NO position filter: the leading passer is the starter whatever the roster
    # sheet calls him. Filtering to position == "QB" is what hid Taysom Hill's
    # five New Orleans starts (step-93 cause A).
    qb_stats = player_stats

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

    merged = merged.drop(columns=["player_id", "qb_id_derived", "qb_id_designated"])

    # Guard: every played team-game must have produced a row. Raises for the
    # current season, warns for history. See _reconcile_team_games.
    _reconcile_team_games(merged, schedules, verbose=verbose)

    return merged


STAT_COLS = [
    "passing_yards", "attempts", "completions", "passing_tds",
    "passing_interceptions", "passing_epa", "passing_cpoe", "sacks_suffered",
]

DEFENSE_COLS = [
    "def_pass_yards_allowed_l8", "def_pass_yards_allowed_std",
    "def_pass_epa_allowed_l8", "def_pass_epa_allowed_std",
]


def _build_defense_table(player_stats):
    """Build per-(season, week, defense_team) opponent pass-defense features.

    For every scheduled team-game (2021-2026, from schedules), "allowed"
    stats are the opposing offense's summed passing_yards/passing_epa for
    that game. Rolling features use the same shift-then-roll pattern as
    the QB stat columns: def_*_l8 is an 8-game rolling mean (crosses
    seasons), def_*_std is a season-to-date expanding mean (resets each
    season). Both are computed from games strictly before (season, week)
    via the shift, so joining this table on (season, week, defense_team)
    is leakage-safe for any season/week, including future ones.
    """
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    team_games = _build_team_game_view(schedules)[["season", "week", "team", "opponent"]]

    offense = player_stats.groupby(["season", "week", "team"], as_index=False).agg(
        def_pass_yards_allowed=("passing_yards", "sum"),
        def_pass_epa_allowed=("passing_epa", "sum"),
    )

    defense = team_games.rename(columns={"team": "defense_team", "opponent": "offense_team"})
    defense = defense.merge(
        offense.rename(columns={"team": "offense_team"}),
        on=["season", "week", "offense_team"],
        how="left",
    )

    defense = defense.sort_values(["defense_team", "season", "week"]).reset_index(drop=True)

    for col in ["def_pass_yards_allowed", "def_pass_epa_allowed"]:
        shifted = defense.groupby("defense_team")[col].shift(1)
        defense[f"{col}_l8"] = (
            shifted.groupby(defense["defense_team"]).transform(lambda s: s.rolling(window=8, min_periods=1).mean())
        )

        shifted_season = defense.groupby(["defense_team", "season"])[col].shift(1)
        defense[f"{col}_std"] = (
            shifted_season.groupby([defense["defense_team"], defense["season"]]).transform(lambda s: s.expanding().mean())
        )

    return defense[["season", "week", "defense_team"] + DEFENSE_COLS]


def _compute_rolling_features(df):
    df = _add_rolling_features(df, group_col="qb_id", stat_cols=STAT_COLS)

    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    defense_table = _build_defense_table(player_stats)
    df = df.merge(
        defense_table.rename(columns={"defense_team": "opponent"}),
        on=["season", "week", "opponent"],
        how="left",
    )

    return df


def add_rolling_features(df):
    df = df.dropna(subset=["passing_yards"]).copy()
    return _compute_rolling_features(df)


def build_prediction_features(season, week, starters=None, line_overrides=None,
                              skip_teams=None):
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

    skip_teams: optional iterable of teams to EXCLUDE from the frame
    entirely (the SKIP sentinel in starters_override.csv). These are
    dropped before the schedule fallback above, so a skipped team cannot
    be silently resurrected from schedules' own qb_id — which is exactly
    what happened on completed weeks before step 70.

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

    # Drop SKIPped teams BEFORE the schedule fallback (step 70). _build_team_game_view
    # seeds qb_id from schedules' home_qb_id/away_qb_id, which is populated for any
    # completed week -- so without this a SKIPped team would silently fall back to the
    # schedule's QB and be predicted anyway. It only looked correct on future weeks,
    # where those columns are null and the row happened to die in preprocessing.
    skip_teams = set(skip_teams or ())
    if skip_teams:
        target_games = target_games[~target_games["team"].isin(skip_teams)].copy()

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
