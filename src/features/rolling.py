"""Generic leakage-safe rolling feature machinery.

Shared by the QB and receiving feature paths. For each stat column:
  {stat}_l3 / {stat}_l8: shift(1) then rolling mean over 3/8 prior games
  within the group, crossing season boundaries.
  {stat}_std: shift(1) within (group, season) then expanding mean —
  season-to-date, resets at season boundaries.
Also adds games_into_season (0-indexed count within (group, season)).
"""


def add_rolling_features(df, group_col, stat_cols, season_col="season", week_col="week"):
    df = df.sort_values([group_col, season_col, week_col]).reset_index(drop=True)

    for col in stat_cols:
        shifted = df.groupby(group_col)[col].shift(1)

        df[f"{col}_l3"] = (
            shifted.groupby(df[group_col]).transform(lambda s: s.rolling(window=3, min_periods=1).mean())
        )
        df[f"{col}_l8"] = (
            shifted.groupby(df[group_col]).transform(lambda s: s.rolling(window=8, min_periods=1).mean())
        )

        shifted_season = df.groupby([group_col, season_col])[col].shift(1)
        df[f"{col}_std"] = (
            shifted_season.groupby([df[group_col], df[season_col]]).transform(lambda s: s.expanding().mean())
        )

    df["games_into_season"] = df.groupby([group_col, season_col]).cumcount()

    return df
