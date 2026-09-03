"""Play-by-play ingestion and team-week aggregation.

Unlike the other ingestion modules, this one does NOT persist its source.
Raw pbp is ~48.8k rows x 372 columns per season (~195 MB in pandas, ~19.5 MB
as parquet); we need a few dozen team-week numbers out of it, so pbp is
loaded, aggregated, and dropped. Only the aggregates are written, to
data/features/ rather than data/raw/ -- they are derived, not raw.

Three outputs:

  defense_team_week.parquet
      One row per (season, week, team): what that team's DEFENSE did.
      pbp-derived volume (dropbacks faced, sacks, QB hits) plus
      PFR-derived pressure/blitz/hurry counts. 2021-2025.

  offense_drives_team_week.parquet
      One row per (season, week, team): that team's OFFENSIVE drive
      structure -- drives, plays, time of possession, field position,
      scoring rate. 2021-2025.

  defense_team_week_ftn.parquet
      One row per (season, week, team) from FTN charting: blitzers,
      pass rushers, box count. 2022-2025 ONLY -- FTN does not exist for
      2021, which is why this is a separate file rather than four NaN
      columns bolted onto defense_team_week.

WHY PFR FOR PRESSURE: play-by-play carries no pressure, blitz, hurry, or
pass-rusher information at all (step 77 scanned all 372 column names; the
only defensive-disruption fields are the completed events -- sack, qb_hit,
tackled_for_loss, interception). PFR advanced stats is QB-week grain with a
`team`/`opponent` pair, so summing over `opponent` yields pressure GENERATED
by that defense. See module docstring notes in aggregate_pfr_defense for the
verification that route passed.

NOTHING IN THE MODEL PIPELINE CONSUMES THESE FILES. They are a data layer,
built and validated ahead of a pre-registered A/B that has not been run.
"""

import argparse

import numpy as np
import pandas as pd

import nflreadpy as nfl

from src.ingestion._common import PROJECT_ROOT

FEATURES_DIR = PROJECT_ROOT / "data" / "features"

SEASONS = list(range(2021, 2026))
FTN_SEASONS = list(range(2022, 2026))          # FTN charting starts in 2022

PFR_COUNT_COLS = ["times_pressured", "times_blitzed", "times_hurried",
                  "times_sacked", "times_hit"]

# Drive results (fixed_drive_result) that put points on the board. Used for
# drives_scoring; "Safety" is excluded because it scores for the DEFENSE.
SCORING_DRIVE_RESULTS = {"Touchdown", "Field goal"}


def save_feature(df, filename):
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / filename
    df.to_parquet(path)
    return path


# --------------------------------------------------------------------- pbp

def _load_pbp(season):
    """Load one season of pbp as pandas. Not persisted anywhere."""
    return nfl.load_pbp([season]).to_pandas()


def aggregate_pbp_defense(pbp):
    """Per (season, week, team): volume faced by that team's defense.

    Keyed on `defteam`, so every row describes what the DEFENSE saw.
    Rows with a null defteam (timeouts, end-of-quarter markers) are dropped.
    """
    d = pbp[pbp["defteam"].notna()].copy()
    out = (d.groupby(["season", "week", "defteam"])
           .agg(dropbacks_faced=("qb_dropback", "sum"),
                plays_faced=("play_id", "size"),
                sacks=("sack", "sum"),
                qb_hits=("qb_hit", "sum"),
                scrimmage_plays_faced=("is_scrimmage", "sum"))
           .reset_index()
           .rename(columns={"defteam": "team"}))
    for c in ("dropbacks_faced", "sacks", "qb_hits", "scrimmage_plays_faced"):
        out[c] = out[c].astype(float)
    return out


def _parse_top(s):
    """'5:31' -> 331.0 seconds. Returns NaN on anything unparseable."""
    if not isinstance(s, str) or ":" not in s:
        return np.nan
    try:
        mins, secs = s.split(":")
        return float(int(mins) * 60 + int(secs))
    except (ValueError, TypeError):
        return np.nan


def _parse_start_yardline(s, posteam):
    """'ARI 22' -> yards from the offense's own goal line (0-100).

    'ARI 22' when ARI has the ball means their own 22 -> 22 yards of field
    behind them. 'NO 45' when ARI has the ball means the opponent's 45 ->
    100 - 45 = 55. Higher = better starting field position.

    Deliberately NOT derived from a first() on `yardline_100`: the kickoff
    row belongs to the drive, so first() returns 35 (the kicking team's own
    35) on every drive that follows a score (step 77 finding).
    """
    if not isinstance(s, str):
        return np.nan
    parts = s.split()
    if len(parts) != 2:
        # Midfield is sometimes recorded as '50'.
        if s.strip() == "50":
            return 50.0
        return np.nan
    side, yard = parts
    try:
        yard = float(yard)
    except (ValueError, TypeError):
        return np.nan
    if not isinstance(posteam, str):
        return np.nan
    return yard if side == posteam else 100.0 - yard


def aggregate_pbp_offense_drives(pbp):
    """Per (season, week, team): that team's offensive drive structure.

    Uses fixed_drive / fixed_drive_result (0.00% null) rather than the
    drive_* family (1.17% null), per step 77.
    """
    d = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()].copy()

    # --- one row per drive first, then per team-week -----------------------
    drive = (d.groupby(["season", "week", "game_id", "posteam", "fixed_drive"])
             .agg(result=("fixed_drive_result", "last"),
                  official_plays=("drive_play_count", "last"),
                  top_str=("drive_time_of_possession", "last"),
                  first_downs=("drive_first_downs", "last"),
                  start_yl_str=("drive_start_yard_line", "last"),
                  scrimmage_plays=("is_scrimmage", "sum"))
             .reset_index())

    drive["top_seconds"] = drive["top_str"].map(_parse_top)
    drive["start_field_pos"] = [
        _parse_start_yardline(s, t)
        for s, t in zip(drive["start_yl_str"], drive["posteam"])
    ]
    drive["is_scoring"] = drive["result"].isin(SCORING_DRIVE_RESULTS).astype(float)

    out = (drive.groupby(["season", "week", "posteam"])
           .agg(drives=("fixed_drive", "size"),
                scrimmage_plays=("scrimmage_plays", "sum"),
                official_plays=("official_plays", "sum"),
                first_downs=("first_downs", "sum"),
                drives_scoring=("is_scoring", "sum"),
                top_seconds=("top_seconds", "sum"),
                avg_start_field_pos=("start_field_pos", "mean"))
           .reset_index()
           .rename(columns={"posteam": "team"}))

    out["plays_per_drive"] = out["scrimmage_plays"] / out["drives"]
    out["first_downs_per_drive"] = out["first_downs"] / out["drives"]
    out["score_rate"] = out["drives_scoring"] / out["drives"]
    out["seconds_per_drive"] = out["top_seconds"] / out["drives"]
    return out


# --------------------------------------------------------------------- PFR

def aggregate_pfr_defense(seasons):
    """Per (season, week, team): pressure GENERATED, summed over opponents.

    PFR advstats(stat_type='pass') is QB-week grain carrying both `team`
    (the passer's team) and `opponent`. Grouping by `opponent` and summing
    therefore gives what that defense did.

    Route verified over 2021-2025 (step 78):
      - team abbreviations are nflverse's: 32/32 match schedules.parquet in
        both the `team` and `opponent` columns, zero either-way difference,
        and game_id is the same 2021_01_ARI_TEN format (1424/1424 match).
      - every one of the 2,848 scheduled team-games has exactly one
        aggregated opponent row; none are missing.
      - 20.7% of team-weeks faced more than one charted passer, so the
        aggregation MUST sum (a defense that faced a starter and a backup
        generated pressure against both).
      - times_sacked cross-checks against pbp-derived sacks on 2,848
        team-weeks: 98.46% exact, r=0.9974, mean |diff| 0.016.

    KNOWN UNDERCOUNT: PFR charts only passers it lists, and occasionally
    omits a relief QB. All 44 cross-check disagreements run one direction
    (PFR low by 1 or 2) and every one inspected is a backup's snaps missing
    -- e.g. 2021 wk8 DET took 6 sacks, 5 on Goff and 1 on David Blough, and
    PFR has only the Goff row. This affects 1.5% of team-weeks by ~1 event.
    The pbp-derived sack/hit counts do NOT have this gap, which is why the
    defense table keeps both and uses pbp volume as the rate denominator.
    """
    frames = []
    for season in seasons:
        d = nfl.load_pfr_advstats([season], stat_type="pass",
                                  summary_level="week").to_pandas()
        frames.append(d)
    pfr = pd.concat(frames, ignore_index=True)

    agg = {c: (c, "sum") for c in PFR_COUNT_COLS}
    agg["pfr_qbs_faced"] = ("pfr_player_name", "size")
    out = (pfr.groupby(["season", "week", "opponent"])
           .agg(**agg)
           .reset_index()
           .rename(columns={"opponent": "team",
                            "times_pressured": "pfr_pressures",
                            "times_blitzed": "pfr_blitzes",
                            "times_hurried": "pfr_hurries",
                            "times_sacked": "pfr_sacks",
                            "times_hit": "pfr_qb_hits"}))
    return out


# --------------------------------------------------------------------- FTN

def aggregate_ftn_defense(seasons):
    """Per (season, week, team): FTN charting means over dropbacks faced.

    FTN is play-level and joins to pbp on
    (nflverse_game_id, nflverse_play_id) -> (game_id, play_id), which
    matched 47,316/47,316 FTN rows with zero orphans in 2025 (step 77).
    Restricted to dropbacks, where n_blitzers / n_pass_rushers /
    n_defense_box are 0.00% null.

    2022+ ONLY. load_ftn_charting([2021]) raises
    'Season must be between 2022 and 2025'.

    ZERO SENTINEL: FTN writes an UNCHARTED play as 0, not null. A dropback
    cannot have zero pass rushers, so `n_pass_rushers == 0` marks the play as
    uncharted and it is excluded from every mean here -- including the blitz
    numbers, whose 0 would otherwise read as a genuine "no blitz". (0 is a
    legitimate value for n_blitzers on a charted play: 70.6% of 2025
    dropbacks were not blitzed.) `n_defense_box == 0` is excluded from the
    box mean on the same logic.

    In 2025 this affects 58 of 20,886 dropbacks (0.28%), but it is not
    uniformly scattered: five team-weeks are uncharted end to end (2024 wk7
    ARI/ATL/LAC/SEA and 2025 wk3 LAC, 29-43 dropbacks each). Left unhandled
    those five would have reported a fabricated 0.0 blitz rate and 0.0 box
    count. They now come through as NaN with ftn_charted_dropbacks = 0,
    which is the honest representation -- and is why the table also carries
    ftn_dropbacks (plays seen) beside ftn_charted_dropbacks (plays usable).
    """
    rows = []
    for season in seasons:
        pbp = _load_pbp(season)
        ftn = nfl.load_ftn_charting([season]).to_pandas()
        m = pbp[["game_id", "play_id", "season", "week", "defteam", "qb_dropback"]].merge(
            ftn[["nflverse_game_id", "nflverse_play_id", "n_blitzers",
                 "n_pass_rushers", "n_defense_box"]],
            left_on=["game_id", "play_id"],
            right_on=["nflverse_game_id", "nflverse_play_id"],
            how="inner",
        )
        db = m[(m["qb_dropback"] == 1) & m["defteam"].notna()].copy()

        charted = db["n_pass_rushers"] > 0
        db["blitzers_v"] = db["n_blitzers"].where(charted)
        db["rushers_v"] = db["n_pass_rushers"].where(charted)
        db["box_v"] = db["n_defense_box"].where(charted & (db["n_defense_box"] > 0))
        db["is_charted"] = charted.astype(float)
        db["is_blitz"] = db["blitzers_v"].ge(1).where(charted).astype(float)

        agg = (db.groupby(["season", "week", "defteam"])
               .agg(ftn_dropbacks=("n_blitzers", "size"),
                    ftn_charted_dropbacks=("is_charted", "sum"),
                    mean_blitzers=("blitzers_v", "mean"),
                    blitz_rate=("is_blitz", "mean"),
                    mean_pass_rushers=("rushers_v", "mean"),
                    mean_defense_box=("box_v", "mean"))
               .reset_index()
               .rename(columns={"defteam": "team"}))
        rows.append(agg)
        n_unch = int((~charted).sum())
        print(f"  FTN {season}: {len(agg)} team-weeks from {len(db):,} dropbacks "
              f"({n_unch} uncharted zero-sentinel plays excluded)")
    return pd.concat(rows, ignore_index=True)


# ----------------------------------------------------------------- drivers

def build_team_week_tables(seasons=SEASONS, ftn_seasons=FTN_SEASONS):
    """Build and save all three aggregate tables. Returns them as a dict."""
    def_frames, off_frames = [], []
    for season in seasons:
        pbp = _load_pbp(season)
        pbp["is_scrimmage"] = pbp["play_type"].isin(["pass", "run"]).astype(float)
        def_frames.append(aggregate_pbp_defense(pbp))
        off_frames.append(aggregate_pbp_offense_drives(pbp))
        print(f"  pbp {season}: {len(pbp):,} plays -> "
              f"{len(def_frames[-1])} defense / {len(off_frames[-1])} offense team-weeks")
        del pbp

    defense = pd.concat(def_frames, ignore_index=True)
    offense = pd.concat(off_frames, ignore_index=True)

    print("\n  PFR pressure aggregation...")
    pfr = aggregate_pfr_defense(seasons)
    defense = defense.merge(pfr, on=["season", "week", "team"], how="left")

    # Rates use the pbp dropback denominator (counts every QB, unlike PFR).
    denom = defense["dropbacks_faced"].replace(0, np.nan)
    defense["sack_rate"] = defense["sacks"] / denom
    defense["qb_hit_rate"] = defense["qb_hits"] / denom
    defense["pressure_rate"] = defense["pfr_pressures"] / denom
    defense["blitz_rate"] = defense["pfr_blitzes"] / denom
    defense["hurry_rate"] = defense["pfr_hurries"] / denom

    print("\n  FTN aggregation (2022+)...")
    ftn = aggregate_ftn_defense(ftn_seasons)

    defense = defense.sort_values(["season", "week", "team"]).reset_index(drop=True)
    offense = offense.sort_values(["season", "week", "team"]).reset_index(drop=True)
    ftn = ftn.sort_values(["season", "week", "team"]).reset_index(drop=True)

    save_feature(defense, "defense_team_week.parquet")
    save_feature(offense, "offense_drives_team_week.parquet")
    save_feature(ftn, "defense_team_week_ftn.parquet")

    return {"defense": defense, "offense": offense, "ftn": ftn}


def main():
    parser = argparse.ArgumentParser(
        description="Build pbp/PFR/FTN team-week aggregates. No model consumes these yet.")
    parser.add_argument("--seasons", type=int, nargs="+", default=SEASONS)
    args = parser.parse_args()

    ftn_seasons = [s for s in args.seasons if s >= 2022]
    tables = build_team_week_tables(args.seasons, ftn_seasons)

    for name, df in tables.items():
        print(f"\n{'=' * 70}\n{name}: {df.shape}\n{'=' * 70}")
        print(f"  seasons: {sorted(df['season'].unique())}")
        print("  per-season team-weeks:")
        print(df.groupby("season").size().to_string())
        print("  null rates:")
        for c in df.columns:
            n = df[c].isna().mean() * 100
            if n > 0:
                print(f"    {c:26s} {n:6.2f}%")


if __name__ == "__main__":
    main()
