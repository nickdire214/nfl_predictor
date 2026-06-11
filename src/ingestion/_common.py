"""Shared helpers for ingestion modules."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def save_parquet(df, filename):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / filename
    df.to_parquet(path)
    return path
