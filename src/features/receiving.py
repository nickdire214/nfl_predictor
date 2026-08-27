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

# Every snap_counts.position label observed among offensive-snap rows in the
# 2021-2025 data. The audit below flags anything outside this set: an
# unrecognized label is how the step-53 `HB` defect hid (CIN's entire 2025
# backfield was labeled HB, which the old exact-match candidacy gate silently
# excluded from both spines, with no counter firing).
KNOWN_SNAP_POSITIONS = {
    # skill / offense
    "QB", "RB", "HB", "FB", "WR", "TE",
    # offensive line
    "T", "G", "C", "OL",
    # defensive (appear on offensive snaps in trick/goal-line packages)
    "DE", "DT", "DL", "NT", "LB", "CB", "DB", "S", "FS", "SS",
    # special teams
    "K", "P", "LS",
}

# Snap-label synonyms for canonical positions. Applied ONLY when a row has no
# canonical position available (no crosswalk match), so the snap label can
# still gate candidacy sensibly. Canonical position from players.parquet is
# always preferred when present.
SNAP_LABEL_SYNONYMS = {"HB": "RB"}


def _audit_snap_position_labels(offensive_snaps, verbose=True):
    """Report the snap-label vocabulary and FLAG any unrecognized label.

    `offensive_snaps` is the offense_snaps > 0 subset of snap_counts. This is
    the counter whose absence let `HB` hide: candidacy used to be an exact
    string match, so a synonym removed a whole position group silently.
    """
    counts = offensive_snaps["position"].value_counts(dropna=False)
    if verbose:
        print(f"Snap position labels among {len(offensive_snaps)} offensive-snap rows:")
        print("  " + ", ".join(f"{pos}={n}" for pos, n in counts.items()))

    unknown = [p for p in counts.index if pd.isna(p) or p not in KNOWN_SNAP_POSITIONS]
    if unknown:
        print("  FLAG: unrecognized snap position label(s) — these are excluded from")
        print("        candidacy unless canonical position admits them. If any is a")
        print("        skill-position synonym, add it to SNAP_LABEL_SYNONYMS:")
        for pos in unknown:
            print(f"          {pos!r}: {counts[pos]} rows")
    elif verbose:
        print("  OK: all labels recognized (no unknown synonyms)")

    return unknown


# --- Prediction-time roster resolution (step 61) -----------------------------
# Both defects fixed here are PREDICTION-TIME ONLY. Historical rows and the
# training path never see either of these: build_*_base and the matrices are
# untouched, so a 2021 Packers game stays a Packers game forever.

# Defect B: how many of a team's most recent games contribute to its predicted
# roster. N=1 (the original behaviour) drops any starter who missed that one
# game -- season-ending injuries made Malik Nabers, Breece Hall, Brock Bowers
# et al. invisible on the entire 2026 wk1 board despite correct team labels.
ROSTER_WINDOW_GAMES = 4

# Defect A: players.parquet `status` values that count as "actually playing".
# CUT / RES / NWT / RSN are excluded because the step-60 diagnostic showed they
# retain a latest_team 100% of the time (CUT 59/59, RES 96/96), so latest_team
# alone cannot distinguish a rostered player from a released one -- status is
# the only discriminator.
ALLOWED_ROSTER_STATUS = {"ACT", "PUP"}


def _build_window_roster(historical, teams, window=ROSTER_WINDOW_GAMES):
    """Roster = anyone with an offensive snap for `team` in its last `window` games.

    `historical` is a spine already restricted to games strictly before the
    target week. For each team we take its last `window` distinct team-games
    (cross-season allowed, same as the original most-recent-game logic) and
    admit every player appearing in any of them.

    `as_of` remains the PLAYER'S OWN most recent game inside that window, not
    the team's -- so it still reads as a staleness signal per player rather
    than collapsing to one value per team.
    """
    rosters = []
    for team in teams:
        team_hist = historical[historical["team"] == team]
        if team_hist.empty:
            continue

        games = (
            team_hist[["season", "week"]]
            .drop_duplicates()
            .sort_values(["season", "week"])
            .tail(window)
        )
        keys = {(int(s), int(w)) for s, w in zip(games["season"], games["week"])}

        in_window = team_hist[
            [(int(s), int(w)) in keys
             for s, w in zip(team_hist["season"], team_hist["week"])]
        ]
        if in_window.empty:
            continue

        # each player's own latest game in the window supplies as_of
        newest = (
            in_window.sort_values(["season", "week"])
            .drop_duplicates(subset="gsis_id", keep="last")
        )
        roster = newest[["gsis_id", "player", "position", "season", "week"]].copy()
        roster["team"] = team
        roster["as_of"] = [
            f"{int(s)} wk{int(w):02d}" for s, w in zip(roster["season"], roster["week"])
        ]
        rosters.append(roster.drop(columns=["season", "week"]))

    if not rosters:
        return pd.DataFrame(columns=["gsis_id", "player", "position", "team", "as_of"])
    return pd.concat(rosters, ignore_index=True)


def _latest_team_is_authoritative(players, season):
    """Is players.parquet's latest_team valid as the team authority for `season`?

    latest_team describes the CURRENT roster snapshot, so it is only meaningful
    when the season being predicted is the season the crosswalk describes.
    Predicting 2025 from a 2026 crosswalk must NOT relabel anyone -- that would
    push 2026 teams onto historical replays and break replay equivalence.

    Self-adjusting: once the pull advances to 2027, 2027 becomes authoritative.
    """
    return int(players["last_season"].max()) == int(season)


def _apply_latest_team(roster_df, players, valid_teams, verbose=False):
    """Reassign predicted-roster players to players.parquet `latest_team`.

    PREDICTION-TIME ONLY -- callers gate on _latest_team_is_authoritative.

    Three filters, in order:
      1. status must be in ALLOWED_ROSTER_STATUS, else the player is dropped
         (a released/reserve player is not playing this week at all)
      2. team becomes latest_team
      3. latest_team must be one of the teams playing this week, else dropped

    A player appearing in two teams' windows (mid-season trade) collapses to
    one row on his latest_team, keeping his freshest as_of.
    """
    if roster_df.empty:
        return roster_df

    pmap = players.drop_duplicates("gsis_id").set_index("gsis_id")
    out = roster_df.copy()
    out["_status"] = out["gsis_id"].map(pmap["status"])
    out["_latest_team"] = out["gsis_id"].map(pmap["latest_team"])

    bad_status = ~out["_status"].isin(ALLOWED_ROSTER_STATUS)
    dropped_status = out[bad_status]
    out = out[~bad_status]

    out["team"] = out["_latest_team"]

    off_slate = ~out["team"].isin(set(valid_teams))
    dropped_team = out[off_slate]
    out = out[~off_slate]

    # trade collapse: one row per player, freshest as_of wins
    out = out.sort_values("as_of").drop_duplicates(subset="gsis_id", keep="last")

    if verbose:
        print(f"  latest_team override: {len(roster_df)} -> {len(out)} rows")
        print(f"    dropped, status not in {sorted(ALLOWED_ROSTER_STATUS)}: {len(dropped_status)}")
        if len(dropped_status):
            print(dropped_status.groupby("_status").size().rename("rows").to_string())
        print(f"    dropped, latest_team not on this week's slate: {len(dropped_team)}")

    return out.drop(columns=["_status", "_latest_team"]).reset_index(drop=True)


ROSTER_OVERRIDE_PATH = PROJECT_ROOT / "data" / "predictions" / "roster_override.csv"
ROSTER_OVERRIDE_MARKETS = {"receiving", "rushing"}


def _apply_roster_override(roster_df, historical, players, season, week, market, verbose=False):
    """Manually add players to a prediction board (step 67B).

    Reads data/predictions/roster_override.csv (columns: season, week, market,
    team, player_name; market in {receiving, rushing}) and injects the matching
    rows into the target-week roster. This is the mechanism RUNBOOK §5.2b
    assumed but did not have: a player whose last appearance falls outside
    ROSTER_WINDOW_GAMES is invisible to the board, and a posted book line is the
    signal that he is back.

    Name resolution mirrors starters_override.csv -- exact `display_name` match
    against players.parquet, with unknown and ambiguous names raising a
    ValueError that lists candidates. It deliberately does NOT filter by
    position first (starters_override filters to QB): a canonically-CB two-way
    player like Travis Hunter must remain addressable, and the history guard
    below is a stricter check than a position filter would be.

    Extra guard: the resolved player must have rows in the historical spine
    strictly before (season, week). Without them every rolling feature would be
    NaN and preprocess_* would silently drop him, so this raises instead.

    The override wins on team assignment: if the player is already on the board
    (on some other team), his existing row is replaced.
    """
    if not ROSTER_OVERRIDE_PATH.exists():
        return roster_df

    ov = pd.read_csv(ROSTER_OVERRIDE_PATH)
    missing_cols = {"season", "week", "market", "team", "player_name"} - set(ov.columns)
    if missing_cols:
        raise ValueError(
            f"roster_override.csv: missing required column(s) {sorted(missing_cols)}; "
            f"expected season, week, market, team, player_name"
        )

    bad_market = set(ov["market"].unique()) - ROSTER_OVERRIDE_MARKETS
    if bad_market:
        raise ValueError(
            f"roster_override.csv: unknown market value(s) {sorted(bad_market)}; "
            f"expected one of {sorted(ROSTER_OVERRIDE_MARKETS)}"
        )

    ov = ov[(ov["season"] == season) & (ov["week"] == week) & (ov["market"] == market)]
    if ov.empty:
        return roster_df

    added = []
    for _, row in ov.iterrows():
        team, name = row["team"], row["player_name"]
        matches = players[players["display_name"] == name]

        if matches.empty:
            raise ValueError(
                f"roster_override.csv: unknown player_name '{name}' for team {team} "
                f"({market}) — no player in players.parquet matches this display_name"
            )

        gsis_ids = matches["gsis_id"].dropna().unique()
        if len(gsis_ids) > 1:
            cands = matches[["gsis_id", "display_name", "position", "rookie_season", "last_season"]]
            raise ValueError(
                f"roster_override.csv: ambiguous player_name '{name}' for team {team} "
                f"({market}) — candidates:\n{cands.to_string(index=False)}"
            )

        gsis_id = gsis_ids[0]
        hist = historical[historical["gsis_id"] == gsis_id]
        if hist.empty:
            info = matches.iloc[0]
            raise ValueError(
                f"roster_override.csv: '{name}' ({team}, {market}) has NO history in the "
                f"{market} spine before {season} wk{week}, so every rolling feature would "
                f"be NaN and he cannot be predicted. "
                f"(position={info['position']}, rookie_season={info['rookie_season']}, "
                f"last_season={info['last_season']}.) Remove the row, or wait until he has "
                f"played a game."
            )

        last = hist.sort_values(["season", "week"]).iloc[-1]
        added.append({
            "gsis_id": gsis_id,
            "player": last["player"],
            "position": last["position"],
            "team": team,
            "as_of": f"{int(last['season'])} wk{int(last['week']):02d}",
        })

    add_df = pd.DataFrame(added)
    # override wins on team assignment: drop any existing row for these players
    kept = roster_df[~roster_df["gsis_id"].isin(add_df["gsis_id"])]
    replaced = len(roster_df) - len(kept)
    out = pd.concat([kept, add_df], ignore_index=True)

    if verbose:
        print(f"  roster_override ({market}): +{len(add_df)} rows "
              f"({replaced} existing row(s) replaced)")
        for r in added:
            print(f"    {r['player']} -> {r['team']} ({r['position']}, as_of {r['as_of']})")

    return out


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


def _build_player_game_spine(snap_counts, players, crosswalk, verbose=True):
    """WR/TE/RB player-game spine, UNION-gated on canonical + snap position.

    Every offensive-snap row is joined to the crosswalk and mapped to its
    canonical position from players.parquet, then admitted if EITHER that
    canonical position OR the synonym-normalized snap label is in
    SPINE_POSITIONS. Canonical position always supplies the position VALUE
    (snap label is the value fallback only for crosswalk misses).

    Two things are being preserved at once:

    - Step 54 (canonical arm): the old build gated on the raw snap label
      BEFORE canonical position was consulted, so an unrecognized synonym
      (`HB`) excluded a player before the step-35 canonical fix could correct
      him -- CIN's entire 2025 backfield vanished from the spine. See step 53.
    - Step 24 (snap arm): players who take WR/TE/RB offensive snaps but are
      canonically FB or CB (fullbacks, Travis Hunter) are deliberately kept in
      the matrix as RB-baseline-encoded rows. A canonical-only gate would drop
      them, so the union keeps step 54 purely additive. See step 55.
    """
    offensive = snap_counts[snap_counts["offense_snaps"] > 0].copy()

    if verbose:
        _audit_snap_position_labels(offensive, verbose=verbose)

    merged = offensive.merge(
        crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="left"
    )

    canon = _canonical_position_map(players)
    mapped = merged["gsis_id"].map(canon)
    snap_label = merged["position"]
    snap_label_norm = snap_label.replace(SNAP_LABEL_SYNONYMS)

    # Canonical position where available, else the normalized snap label.
    effective = mapped.fillna(snap_label_norm)

    # UNION gate (step 55): admit if EITHER the canonical position OR the
    # synonym-normalized snap label is a spine position. Canonical still
    # supplies the position VALUE.
    #
    # The canonical arm is the step-54 HB fix (it admits players the snap label
    # mislabels). The snap arm preserves step 24's deliberate inclusion of
    # players who take WR/TE/RB offensive snaps but are canonically FB or CB
    # (Travis Hunter, fullbacks) -- a canonical-only gate would have dropped
    # them, retiring a documented population by side effect. They stay in the
    # matrix carrying their canonical FB/CB position, RB-baseline-encoded by
    # the model and excluded from per-position sigma fitting, exactly as
    # step 24 specified. Union makes step 54 purely additive.
    canonical_admits = mapped.isin(SPINE_POSITIONS)
    snap_admits = snap_label_norm.isin(SPINE_POSITIONS)
    candidates = canonical_admits | snap_admits

    # Join-failure rate is reported over CANDIDATES only (skill-position rows),
    # so the denominator stays comparable to the pre-step-54 build rather than
    # being diluted by linemen, who are now in the input population.
    cand = merged[candidates]
    unmatched = cand[cand["gsis_id"].isna()]
    rate = len(unmatched) / len(cand) if len(cand) else 0.0
    if verbose:
        print(f"Snap rows (WR/TE/RB candidates): {len(cand)}, "
              f"unmatched to gsis_id: {len(unmatched)} ({rate:.4%})")
        if rate > 0.02:
            print("Unmatched examples:")
            print(
                unmatched[["season", "week", "team", "player", "pfr_player_id", "position"]]
                .drop_duplicates(subset=["player", "pfr_player_id"])
                .head(10)
                .to_string(index=False)
            )

    keep = candidates & merged["gsis_id"].notna()

    if verbose:
        # Delta vs the PRE-step-54 snap-label gate (raw snap label in
        # SPINE_POSITIONS, then drop unmatched). The union gate must be a
        # strict superset of it: `removed` is expected to be 0.
        old_keep = snap_label.isin(SPINE_POSITIONS) & merged["gsis_id"].notna()
        added = keep & ~old_keep
        removed = old_keep & ~keep
        print(f"Union candidacy vs pre-step-54 snap-label gate: "
              f"+{int(added.sum())} admitted, -{int(removed.sum())} excluded "
              f"(net {int(keep.sum()) - int(old_keep.sum()):+d})")
        if removed.any():
            print("  FLAG: the union gate should be purely additive — these rows regressed:")
            print(merged[removed].assign(canonical=effective[removed])
                  .groupby(["position", "canonical"]).size()
                  .rename("rows").reset_index().to_string(index=False))
        else:
            print("  OK: purely additive — no row the old gate admitted was dropped")
        if added.any():
            print("  admitted by the canonical arm (snap label -> canonical):")
            print(merged[added].assign(canonical=effective[added])
                  .groupby(["position", "canonical"]).size()
                  .rename("rows").reset_index().to_string(index=False))

        # Rows the canonical arm alone would NOT have admitted -- i.e. step 24's
        # FB/CB population, kept alive by the snap arm.
        snap_only = keep & ~canonical_admits
        print(f"  retained by the snap arm only (canonical outside SPINE_POSITIONS): "
              f"{int(snap_only.sum())} rows")
        if snap_only.any():
            print(merged[snap_only].assign(canonical=effective[snap_only])
                  .groupby(["position", "canonical"]).size()
                  .rename("rows").reset_index().to_string(index=False))

    spine = merged[keep].drop(columns=["pfr_id"]).copy()
    spine["position"] = effective[keep]

    if verbose:
        fallback = mapped[keep].isna()
        changed = (effective[keep] != snap_label[keep]) & ~fallback
        print(
            f"Canonical position: {int(fallback.sum())} rows fell back to snap label "
            f"({fallback.mean():.4%}); "
            f"{int(changed.sum())} (player, week) rows reclassified vs snap label"
        )
        print("Per-position rows (snap-sourced -> canonical):")
        before = snap_label[keep].value_counts()
        after = spine["position"].value_counts()
        for pos in sorted(set(before.index) | set(after.index)):
            print(f"  {pos:<4} {before.get(pos, 0):>6} -> {after.get(pos, 0):>6}")

    return spine


def build_receiving_base(verbose=True):
    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    snap_counts = pd.read_parquet(RAW_DATA_DIR / "snap_counts.parquet")
    player_stats = pd.read_parquet(RAW_DATA_DIR / "player_stats.parquet")
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")

    crosswalk = _build_crosswalk(players, verbose=verbose)
    # Candidacy + canonical position are both resolved inside the spine builder
    # (step 54): position is already canonical on return.
    spine = _build_player_game_spine(snap_counts, players, crosswalk, verbose=verbose)

    spine = spine[
        ["season", "week", "team", "opponent", "gsis_id", "player", "position",
         "offense_snaps", "offense_pct"]
    ].copy()

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


def build_receiving_prediction_features(
    season, week, line_overrides=None,
    roster_window=ROSTER_WINDOW_GAMES, apply_latest_team=None, verbose=False,
):
    """Build feature rows for an upcoming week's receiving props.

    Mirrors src.features.engineer.build_prediction_features: target-week
    rows are constructed from schedules with all stat/label columns set to
    NaN, concatenated with the historical receiving base restricted to
    games strictly before (season, week), and run through the shared
    roller (add_rolling_features, grouped by gsis_id). Only the
    target-week rows are returned.

    Player population for the target week (step 61): every WR/TE/RB with an
    offensive snap for that team across its last `roster_window` prior games
    (default ROSTER_WINDOW_GAMES=4, cross-season allowed), then -- when
    latest_team is authoritative for `season` -- reassigned to
    players.parquet's latest_team and filtered by status. A one-game window
    silently dropped every starter who missed that specific game.

    Both roster changes are PREDICTION-TIME ONLY. `apply_latest_team` defaults
    to _latest_team_is_authoritative(players, season), which is False whenever
    the target season is not the season the crosswalk describes -- so a 2025
    replay is untouched and replay equivalence is structurally preserved.
    Pass roster_window=1, apply_latest_team=False to reproduce the pre-step-61
    board exactly.

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

    target_teams = target_games["team"].unique()
    roster_df = _build_window_roster(historical, target_teams, window=roster_window)

    players = pd.read_parquet(RAW_DATA_DIR / "players.parquet")
    if apply_latest_team is None:
        apply_latest_team = _latest_team_is_authoritative(players, season)
    if apply_latest_team:
        roster_df = _apply_latest_team(roster_df, players, target_teams, verbose=verbose)

    roster_df = _apply_roster_override(
        roster_df, historical, players, season, week, "receiving", verbose=verbose
    )

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
