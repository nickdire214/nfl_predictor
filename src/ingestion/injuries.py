"""Ingestion module for player injury reports and statuses.

Pulls weekly injury report data from nflreadpy and persists it as a
parquet file in data/raw/. Pull-and-persist only — no transformation
beyond Polars to pandas conversion.
"""

import nflreadpy as nfl

from src.ingestion._common import RAW_DATA_DIR, save_parquet

SEASONS = list(range(2021, 2026))


def ingest_injuries(seasons):
    df = nfl.load_injuries(seasons).to_pandas()
    save_parquet(df, "injuries.parquet")
    return df


def main():
    df = ingest_injuries(SEASONS)
    path = RAW_DATA_DIR / "injuries.parquet"

    if df.empty:
        print("WARNING: injuries returned an empty dataframe")

    seasons_present = sorted(df["season"].unique()) if "season" in df.columns else "N/A"
    size_mb = path.stat().st_size / (1024 * 1024)

    print("injuries:")
    print(f"  shape: {df.shape}")
    print(f"  seasons present: {seasons_present}")
    print(f"  saved to: {path} ({size_mb:.2f} MB)")
    print()


if __name__ == "__main__":
    main()
