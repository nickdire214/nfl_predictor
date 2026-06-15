"""Odds ingestion module — Part 1: events, schedule matching, game lines.

Fetches NFL events and game-level odds (spreads/totals) from The Odds API,
matches events to nflverse schedules, and persists raw bulk snapshots as
parquet in data/raw/odds/. The API key is read from .env and never printed.
"""

import os
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from dotenv import load_dotenv

from src.ingestion._common import RAW_DATA_DIR

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT_KEY = "americanfootball_nfl"

ODDS_DATA_DIR = RAW_DATA_DIR / "odds"

ET_ZONE = ZoneInfo("America/New_York")

# The Odds API full team names -> nflverse team abbreviations (all 32 teams).
TEAM_NAME_MAP = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV",
    "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


def _assert_team_map_valid(schedules):
    schedule_teams = set(schedules["home_team"]) | set(schedules["away_team"])
    mapped_abbrs = set(TEAM_NAME_MAP.values())

    missing = mapped_abbrs - schedule_teams
    assert not missing, f"TEAM_NAME_MAP abbreviations not found in schedules: {missing}"
    assert len(TEAM_NAME_MAP) == 32, f"TEAM_NAME_MAP has {len(TEAM_NAME_MAP)} entries, expected 32"


def fetch_events():
    """Fetch upcoming/live NFL events (free endpoint).

    Returns a dataframe with event_id, commence_time (UTC), home/away
    full names and mapped nflverse abbreviations, and gameday_et (the
    commence_time's date in America/New_York).
    """
    resp = requests.get(f"{BASE_URL}/sports/{SPORT_KEY}/events", params={"apiKey": API_KEY})
    resp.raise_for_status()

    print(f"fetch_events: HTTP {resp.status_code}")
    print(f"  x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"  x-requests-used: {resp.headers.get('x-requests-used')}")

    events = resp.json()
    rows = []
    for e in events:
        commence_time = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00"))
        gameday_et = commence_time.astimezone(ET_ZONE).date()

        rows.append({
            "event_id": e["id"],
            "commence_time": commence_time,
            "home_team_full": e["home_team"],
            "away_team_full": e["away_team"],
            "home_team": TEAM_NAME_MAP.get(e["home_team"]),
            "away_team": TEAM_NAME_MAP.get(e["away_team"]),
            "gameday_et": gameday_et,
        })

    return pd.DataFrame(rows)


def match_events_to_schedule(events_df):
    """Match fetched events to nflverse schedules on (teams, ET gameday).

    Returns the matched dataframe (events_df columns + season/week/game_id).
    Also prints both directions of mismatch: events that matched no
    schedule row, and (within the date span covered by events) schedule
    games with no matching event.
    """
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    schedules = schedules.copy()
    schedules["gameday_dt"] = pd.to_datetime(schedules["gameday"]).dt.date

    if events_df.empty:
        print("\nNo events to match.")
        return events_df.assign(season=pd.Series(dtype="float"), week=pd.Series(dtype="float"),
                                 game_id=pd.Series(dtype="object"))

    matched = events_df.merge(
        schedules[["season", "week", "game_id", "home_team", "away_team", "gameday_dt"]],
        left_on=["home_team", "away_team", "gameday_et"],
        right_on=["home_team", "away_team", "gameday_dt"],
        how="left",
    ).drop(columns=["gameday_dt"])

    unmatched_events = matched[matched["game_id"].isna()]
    print(f"\nEvents matched: {len(matched) - len(unmatched_events)}/{len(matched)}")

    if not unmatched_events.empty:
        print("\nEvents that matched no schedule row:")
        for _, row in unmatched_events.iterrows():
            print(f"  {row['gameday_et']}: {row['away_team_full']} @ {row['home_team_full']} "
                  f"({row['away_team']} @ {row['home_team']})")
    else:
        print("All events matched a schedule row.")

    start, end = events_df["gameday_et"].min(), events_df["gameday_et"].max()
    in_span = schedules[(schedules["gameday_dt"] >= start) & (schedules["gameday_dt"] <= end)]
    matched_game_ids = set(matched["game_id"].dropna())
    no_event = in_span[~in_span["game_id"].isin(matched_game_ids)]

    print(f"\nSchedule games in {start}..{end} with no matching event: {len(no_event)}/{len(in_span)}")
    if not no_event.empty:
        for _, row in no_event.iterrows():
            print(f"  {row['gameday']}: {row['away_team']} @ {row['home_team']} ({row['game_id']})")

    return matched


def fetch_game_lines(snapshot_label):
    """Fetch a bulk game-lines snapshot (spreads + totals, US books).

    One row per (event_id, bookmaker). Saves to
    data/raw/odds/game_lines_{snapshot_label}.parquet and prints the
    quota headers after the call.
    """
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/odds",
        params={
            "apiKey": API_KEY,
            "regions": "us",
            "markets": "spreads,totals",
            "oddsFormat": "american",
        },
    )
    resp.raise_for_status()

    print(f"\nfetch_game_lines: HTTP {resp.status_code}")
    print(f"  x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"  x-requests-used: {resp.headers.get('x-requests-used')}")
    print(f"  x-requests-last: {resp.headers.get('x-requests-last')}")

    pulled_at = datetime.now(timezone.utc)
    rows = []

    for event in resp.json():
        home_full, away_full = event["home_team"], event["away_team"]

        for bookmaker in event.get("bookmakers", []):
            row = {
                "event_id": event["id"],
                "home_team": TEAM_NAME_MAP.get(home_full),
                "away_team": TEAM_NAME_MAP.get(away_full),
                "bookmaker": bookmaker["key"],
                "spread_home": None,
                "spread_home_odds": None,
                "spread_away_odds": None,
                "total": None,
                "over_odds": None,
                "under_odds": None,
                "pulled_at_utc": pulled_at,
            }

            for market in bookmaker.get("markets", []):
                if market["key"] == "spreads":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == home_full:
                            row["spread_home"] = outcome["point"]
                            row["spread_home_odds"] = outcome["price"]
                        elif outcome["name"] == away_full:
                            row["spread_away_odds"] = outcome["price"]
                elif market["key"] == "totals":
                    for outcome in market["outcomes"]:
                        if outcome["name"] == "Over":
                            row["total"] = outcome["point"]
                            row["over_odds"] = outcome["price"]
                        elif outcome["name"] == "Under":
                            row["under_odds"] = outcome["price"]

            rows.append(row)

    lines_df = pd.DataFrame(rows)

    ODDS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = ODDS_DATA_DIR / f"game_lines_{snapshot_label}.parquet"
    lines_df.to_parquet(out_path)
    print(f"  saved to: {out_path}")

    return lines_df


def consensus_game_lines(lines_df):
    """Collapse a game-lines snapshot to one consensus row per event.

    Returns a dataframe with event_id, the median spread_home and total
    across bookmakers, and n_books (number of bookmakers contributing).
    """
    return (
        lines_df.groupby("event_id")
        .agg(
            spread_home=("spread_home", "median"),
            total=("total", "median"),
            n_books=("bookmaker", "nunique"),
        )
        .reset_index()
    )


def lines_to_overrides(consensus_df, matched_events):
    """Convert consensus game lines to the line_overrides dict.

    Returns {team: (spread_line, total_line)} for both teams of each
    matched game, in nflverse's spread_line convention (positive =
    home favored). The Odds API's spread_home is the home team's
    handicap (home favorite = negative), so spread_line = -spread_home.

    Prints a 3-game sanity check after conversion: for each game, shows
    the converted spread_line and both teams' implied totals, and
    confirms the favorite (per the original spread_home) has the larger
    implied total.
    """
    merged = matched_events.merge(consensus_df, on="event_id", how="inner")
    merged["spread_line"] = -merged["spread_home"]
    merged["total_line"] = merged["total"]

    overrides = {}
    for _, row in merged.iterrows():
        overrides[row["home_team"]] = (row["spread_line"], row["total_line"])
        overrides[row["away_team"]] = (row["spread_line"], row["total_line"])

    print("\nSpread sign-conversion sanity check (3 games):")
    for _, row in merged.head(3).iterrows():
        home_implied = (row["total_line"] + row["spread_line"]) / 2
        away_implied = (row["total_line"] - row["spread_line"]) / 2
        favorite = "home" if row["spread_home"] < 0 else "away"
        favorite_implied = home_implied if favorite == "home" else away_implied
        other_implied = away_implied if favorite == "home" else home_implied
        check = "OK" if favorite_implied > other_implied else "MISMATCH"
        print(
            f"  {row['away_team']} @ {row['home_team']}: spread_home={row['spread_home']}, "
            f"-> spread_line={row['spread_line']}, total_line={row['total_line']}, "
            f"home_implied={home_implied}, away_implied={away_implied}, "
            f"favorite={favorite} ({check}: favorite has larger implied total)"
        )

    return overrides


def main():
    schedules = pd.read_parquet(RAW_DATA_DIR / "schedules.parquet")
    _assert_team_map_valid(schedules)

    events_df = fetch_events()
    print(f"\nEvents fetched: {len(events_df)}")

    matched = match_events_to_schedule(events_df)

    snapshot_label = f"2026wk1_test_{date.today().isoformat()}"
    lines_df = fetch_game_lines(snapshot_label)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"events fetched: {len(events_df)}")
    if len(matched):
        n_matched = matched["game_id"].notna().sum()
        match_rate = n_matched / len(matched)
        print(f"match rate: {n_matched}/{len(matched)} ({match_rate:.1%})")
    else:
        print("match rate: n/a (no events)")

    n_lines = len(lines_df)
    n_events_with_lines = lines_df["event_id"].nunique() if n_lines else 0
    n_books = lines_df["bookmaker"].nunique() if n_lines else 0
    print(f"lines parsed: {n_lines} rows, {n_events_with_lines} unique events, {n_books} unique bookmakers")

    if "season" in matched.columns:
        week1 = matched[(matched["season"] == 2026) & (matched["week"] == 1)]
        week1_lines = lines_df.merge(week1[["event_id", "season", "week"]], on="event_id", how="inner")
    else:
        week1_lines = lines_df.iloc[0:0]

    print("\nSample week-1 lines (up to 5 rows):")
    if week1_lines.empty:
        print("  (none — no week-1 lines posted/matched yet)")
    else:
        print(week1_lines[["event_id", "bookmaker", "home_team", "away_team", "spread_home", "total"]]
              .head(5).to_string(index=False))


if __name__ == "__main__":
    main()
