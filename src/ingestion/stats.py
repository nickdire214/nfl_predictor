"""Ingestion module for player and team statistics.

Pulls player stats, snap counts, and schedules from nflreadpy and
persists them as parquet files in data/raw/. Pull-and-persist only —
no transformation beyond Polars to pandas conversion.
"""

from pathlib import Path

import nflreadpy as nfl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SEASONS = list(range(2021, 2026))


def ingest_player_stats(seasons):
    df = nfl.load_player_stats(seasons).to_pandas()
    return _save(df, "player_stats.parquet")


def ingest_snap_counts(seasons):
    df = nfl.load_snap_counts(seasons).to_pandas()
    return _save(df, "snap_counts.parquet")


def ingest_schedules(seasons):
    df = nfl.load_schedules(seasons).to_pandas()
    return _save(df, "schedules.parquet")


def _save(df, filename):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / filename
    df.to_parquet(path)
    return df


def main():
    datasets = {
        "player_stats": ingest_player_stats,
        "snap_counts": ingest_snap_counts,
        "schedules": ingest_schedules,
    }

    for name, ingest_fn in datasets.items():
        df = ingest_fn(SEASONS)
        path = RAW_DATA_DIR / f"{name}.parquet"

        if df.empty:
            print(f"WARNING: {name} returned an empty dataframe")

        seasons_present = sorted(df["season"].unique()) if "season" in df.columns else "N/A"
        size_bytes = path.stat().st_size
        size_mb = size_bytes / (1024 * 1024)

        print(f"{name}:")
        print(f"  shape: {df.shape}")
        print(f"  seasons present: {seasons_present}")
        print(f"  saved to: {path} ({size_mb:.2f} MB)")
        print()


if __name__ == "__main__":
    main()
