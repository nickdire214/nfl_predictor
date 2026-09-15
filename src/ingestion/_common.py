"""Shared helpers for ingestion modules."""

from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def save_parquet(df, filename):
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DATA_DIR / filename
    df.to_parquet(path)
    return path


def current_season(today=None):
    """The NFL season year that the given date falls in.

    The league year turns over in March, and a season is named for the
    calendar year it STARTS in: season 2026 runs Sep 2026 through Feb 2027.
    So anything from March onward belongs to the current calendar year, and
    Jan/Feb belong to the season that started the previous year.

    Added 2026-09-15 (step 87). The ingestion modules previously hardcoded
    `range(2021, 2026)`, which silently excluded 2026 from player_stats and
    snap_counts on the first live grading run -- the graders found no week-1
    actuals and every row came back ungraded. A hardcoded upper bound is a
    bug with a one-year fuse; deriving it from the date defuses it.
    """
    today = today or date.today()
    return today.year if today.month >= 3 else today.year - 1
