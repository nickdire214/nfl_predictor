"""Ingestion module for player and team statistics.

Pulls player stats, snap counts, schedules, and the player ID
crosswalk from nflreadpy and persists them as parquet files in
data/raw/. Pull-and-persist only — no transformation beyond Polars
to pandas conversion.
"""

import nflreadpy as nfl

from src.ingestion._common import RAW_DATA_DIR, current_season, save_parquet

# Upper bound derived from the date, not hardcoded (step 87): a literal end year
# is a bug with a one-year fuse, and it fired -- 2026 was silently missing from
# player_stats and snap_counts on the first live grading run. Schedules run one
# season past the stats because the upcoming slate is published before it is played.
# Requesting a season nflverse has not published yet (SCHEDULE_SEASONS always
# asks for one) returns EMPTY SILENTLY -- no error, no warning, and the saved
# file simply lacks that season. Never read a successful run as proof that a
# season arrived; check `seasons present` in the printed output.
SEASONS = list(range(2021, current_season() + 1))
SCHEDULE_SEASONS = list(range(2021, current_season() + 2))


def ingest_player_stats(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    save_parquet(df, "player_stats.parquet")
    return df


def ingest_snap_counts(seasons):
    df = nfl.load_snap_counts(seasons).to_pandas()
    save_parquet(df, "snap_counts.parquet")
    return df


def ingest_schedules(seasons):
    df = nfl.load_schedules(seasons).to_pandas()
    save_parquet(df, "schedules.parquet")
    return df


def ingest_players():
    df = nfl.load_players().to_pandas()
    save_parquet(df, "players.parquet")
    return df


def main():
    seasonal_datasets = {
        "player_stats": (ingest_player_stats, SEASONS),
        "snap_counts": (ingest_snap_counts, SEASONS),
        "schedules": (ingest_schedules, SCHEDULE_SEASONS),
    }

    for name, (ingest_fn, seasons) in seasonal_datasets.items():
        df = ingest_fn(seasons)
        path = RAW_DATA_DIR / f"{name}.parquet"

        if df.empty:
            print(f"WARNING: {name} returned an empty dataframe")

        seasons_present = sorted(df["season"].unique()) if "season" in df.columns else "N/A"
        size_mb = path.stat().st_size / (1024 * 1024)

        print(f"{name}:")
        print(f"  shape: {df.shape}")
        print(f"  seasons present: {seasons_present}")
        print(f"  saved to: {path} ({size_mb:.2f} MB)")
        print()

    df = ingest_players()
    path = RAW_DATA_DIR / "players.parquet"

    if df.empty:
        print("WARNING: players returned an empty dataframe")

    size_mb = path.stat().st_size / (1024 * 1024)

    print("players:")
    print(f"  shape: {df.shape}")
    print(f"  saved to: {path} ({size_mb:.2f} MB)")
    print()


if __name__ == "__main__":
    main()
