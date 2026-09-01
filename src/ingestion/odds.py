"""Odds ingestion module — Part 1: events, schedule matching, game lines.

Fetches NFL events and game-level odds (spreads/totals) from The Odds API,
matches events to nflverse schedules, and persists raw bulk snapshots as
parquet in data/raw/odds/. The API key is read from .env and never printed.
"""

import json
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


def fetch_event_props(event_id, markets, regions="us", save_raw=True):
    """Fetch player-prop odds for ONE event (per-event odds endpoint).

    Thin reconnaissance wrapper on
    /v4/sports/{sport}/events/{event_id}/odds — deliberately NOT a parser:
    it returns the decoded JSON exactly as the API sent it, so the payload
    shape can be inspected before any normalization is designed.

    Player props are only available on the per-event endpoint (they are not
    returned by the bulk /odds route used by fetch_game_lines), and they are
    priced per event per market, so cost is measured empirically from the
    quota headers rather than assumed.

    `markets`: comma-separated market keys, e.g.
        "player_pass_yds,player_reception_yds,player_rush_yds"
    `save_raw`: write the untouched payload to
        data/raw/odds/props_raw_{event_id}_{YYYY-MM-DD}.json

    Returns (payload, headers) where headers carries the quota fields.
    """
    resp = requests.get(
        f"{BASE_URL}/sports/{SPORT_KEY}/events/{event_id}/odds",
        params={
            "apiKey": API_KEY,
            "regions": regions,
            "markets": markets,
            "oddsFormat": "american",
        },
    )
    resp.raise_for_status()

    print(f"\nfetch_event_props({event_id}): HTTP {resp.status_code}")
    print(f"  markets requested: {markets}")
    print(f"  x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"  x-requests-used: {resp.headers.get('x-requests-used')}")
    print(f"  x-requests-last: {resp.headers.get('x-requests-last')}")

    payload = resp.json()

    if save_raw:
        ODDS_DATA_DIR.mkdir(parents=True, exist_ok=True)
        out_path = ODDS_DATA_DIR / f"props_raw_{event_id}_{date.today().isoformat()}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"  saved raw JSON to: {out_path}")

    return payload, dict(resp.headers)


def parse_event_props(raw_json):
    """Flatten a per-event player-prop payload into a long-form frame.

    One row per (bookmaker, market, player, side). Columns:
        event_id, home_team, away_team (nflverse abbreviations),
        bookmaker, market, book_name, side, line, price, last_update

    `book_name` comes from the outcome's `description` field and `side` from
    its `name` field -- the endpoint carries the PLAYER in `description` and
    Over/Under in `name`, the opposite of the game-lines endpoint where `name`
    is the team or Over/Under. No normalization happens here: book_name is
    passed through verbatim for the resolver to handle.

    NOTE: the payload carries NO per-player team attribution (see step 58) --
    team is only known at event level, hence both home_team and away_team on
    every row. Event-scoped resolution is the resolver's problem, not this
    function's.

    A missing `bookmakers` key, a bookmaker with no `markets`, a market with
    no `outcomes`, or an outcome missing fields all degrade to fewer rows /
    None values rather than raising.
    """
    event_id = raw_json.get("id")
    home_team = TEAM_NAME_MAP.get(raw_json.get("home_team"))
    away_team = TEAM_NAME_MAP.get(raw_json.get("away_team"))

    rows = []
    for bookmaker in raw_json.get("bookmakers") or []:
        for market in bookmaker.get("markets") or []:
            for outcome in market.get("outcomes") or []:
                rows.append({
                    "event_id": event_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "bookmaker": bookmaker.get("key"),
                    "market": market.get("key"),
                    "book_name": outcome.get("description"),
                    "side": outcome.get("name"),
                    "line": outcome.get("point"),
                    "price": outcome.get("price"),
                    "last_update": market.get("last_update"),
                })

    columns = ["event_id", "home_team", "away_team", "bookmaker", "market",
               "book_name", "side", "line", "price", "last_update"]
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows)[columns]


def consensus_prop_lines(props_df):
    """Collapse a parsed prop frame to one row per (market, book_name).

    Mirrors consensus_game_lines: median across bookmakers, plus n_books so
    thin coverage is auditable. Over and Under prices stay SEPARATE (a prop's
    two sides are priced independently and are not interchangeable).

    Returns columns: event_id, home_team, away_team, market, book_name,
        line (median across all rows for the pair), n_books,
        over_price / under_price (median per side), last_update (max).

    The line is medianed across every (bookmaker, side) row for the pair --
    Over and Under of the same prop share one line, so this is a median over
    books, not a blend of two different quantities.
    """
    columns = ["event_id", "home_team", "away_team", "market", "book_name",
               "line", "n_books", "over_price", "under_price", "last_update"]
    if props_df.empty:
        return pd.DataFrame(columns=columns)

    def _side_price(group, side):
        vals = group.loc[group["side"] == side, "price"]
        return vals.median() if len(vals) else None

    records = []
    for (market, book_name), g in props_df.groupby(["market", "book_name"], sort=True):
        records.append({
            "event_id": g["event_id"].iloc[0],
            "home_team": g["home_team"].iloc[0],
            "away_team": g["away_team"].iloc[0],
            "market": market,
            "book_name": book_name,
            "line": g["line"].median(),
            "n_books": g["bookmaker"].nunique(),
            "over_price": _side_price(g, "Over"),
            "under_price": _side_price(g, "Under"),
            "last_update": g["last_update"].max(),
        })

    return pd.DataFrame(records)[columns]


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


def lines_to_overrides(consensus_df, matched_events, season, week):
    """Convert consensus game lines to the line_overrides dict for ONE week.

    Returns {team: (spread_line, total_line)} for both teams of each
    matched game, in nflverse's spread_line convention (positive =
    home favored). The Odds API's spread_home is the home team's
    handicap (home favorite = negative), so spread_line = -spread_home.

    `season` and `week` are REQUIRED and the frame is filtered to them here.

    Why the function filters rather than trusting a pre-filtered frame: the
    returned dict is keyed by TEAM, so a frame spanning several weeks does not
    error and does not produce NaN -- each team simply ends up with whichever of
    its games iterated last, silently giving every row of the prediction a wrong
    team_implied_total. A saved game-lines snapshot spans the whole season (272
    events), so the unfiltered frame is the natural thing to pass and the
    corruption is invisible. Requiring `season`/`week` -- and requiring the
    columns needed to honour them -- makes the mistake impossible to make
    quietly.

    Raises ValueError if:
      - `matched_events` lacks `season`/`week` columns (week identity cannot be
        verified, so filtering cannot be honoured -- refuse rather than guess);
      - no rows survive the (season, week) filter;
      - a team appears in more than one surviving game (the corruption symptom).

    Events that matched no schedule row (NaN season/week, from
    match_events_to_schedule) carry no week identity and are dropped, with a
    count printed.

    Prints a 3-game sanity check after conversion: for each game, shows
    the converted spread_line and both teams' implied totals, and
    confirms the favorite (per the original spread_home) has the larger
    implied total.
    """
    missing_cols = {"season", "week"} - set(matched_events.columns)
    if missing_cols:
        raise ValueError(
            f"lines_to_overrides requires season/week columns on matched_events to "
            f"filter to a single week; missing {sorted(missing_cols)}. Pass the output "
            f"of match_events_to_schedule(), or join the snapshot's events to "
            f"schedules.parquet first. Columns present: {sorted(matched_events.columns)}"
        )

    scoped = matched_events.copy()
    n_all = len(scoped)
    unidentified = scoped["season"].isna() | scoped["week"].isna()
    if unidentified.any():
        print(f"\nlines_to_overrides: dropping {int(unidentified.sum())} event(s) with no "
              f"schedule match (NaN season/week) — no week identity to filter on.")
        scoped = scoped[~unidentified]

    scoped = scoped[(scoped["season"] == season) & (scoped["week"] == week)]
    if scoped.empty:
        raise ValueError(
            f"lines_to_overrides: no events for {season} week {week} in matched_events "
            f"({n_all} rows in, spanning "
            f"{sorted(matched_events.dropna(subset=['season', 'week'])[['season', 'week']].drop_duplicates().itertuples(index=False, name=None))})."
        )
    if len(scoped) < n_all:
        print(f"lines_to_overrides: filtered {n_all} -> {len(scoped)} events for "
              f"{season} week {week}.")

    merged = scoped.merge(consensus_df, on="event_id", how="inner")
    if merged.empty:
        raise ValueError(
            f"lines_to_overrides: {len(scoped)} events for {season} week {week} but none "
            f"have consensus lines (event_id join was empty)."
        )

    teams = pd.concat([merged["home_team"], merged["away_team"]])
    dupes = sorted(teams[teams.duplicated()].unique())
    if dupes:
        raise ValueError(
            f"lines_to_overrides: team(s) {dupes} appear in more than one game for "
            f"{season} week {week}. The override dict is keyed by team, so one of the "
            f"games would silently win. Deduplicate the events before calling."
        )

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
