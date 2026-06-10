# Decisions Log

| Date | Component | Change | Reason | Evidence |
| --- | --- | --- | --- | --- |
| 2026-06-10 | Ingestion | Use nflreadpy instead of deprecated nfl_data_py; convert Polars→pandas at ingestion boundary | nfl_data_py officially deprecated, no further updates | nflverse README |
| 2026-06-10 | Ingestion | Verified Odds API key works pre-season; quota at 425 remaining / 75 used after this check | Confirm API access and quota headroom before building ingestion against it | scripts/check_odds_api.py output (2 calls, both HTTP 200) |
| 2026-06-10 | Ingestion | Use seasons 2021-2025 (17-game era) for player_stats, snap_counts, schedules; store raw pulls as parquet in data/raw/ | 2020 excluded as a COVID-affected, anomalous (16-game) season; parquet is compact and preserves dtypes for downstream pandas use | src/ingestion/stats.py output: player_stats (94845, 115) 3.20MB, snap_counts (132616, 16) 0.99MB, schedules (1424, 46) 0.13MB |
