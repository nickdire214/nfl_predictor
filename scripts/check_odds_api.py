"""Diagnostic script: verify The Odds API access and quota.

Quick standalone check, not pipeline code. Run with the venv's python:
    venv\\Scripts\\python.exe scripts\\check_odds_api.py

Never prints the API key itself.
"""

import sys

import requests
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"


def section(title):
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


if not API_KEY:
    print("ERROR: ODDS_API_KEY not found in .env. Set it and try again.")
    sys.exit(1)


try:
    section("Sports list")
    resp = requests.get(f"{BASE_URL}/sports/", params={"apiKey": API_KEY})
    print(f"HTTP status: {resp.status_code}")
    print(f"x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"x-requests-used: {resp.headers.get('x-requests-used')}")

    sports = resp.json()
    nfl_related = [s for s in sports if "americanfootball" in s.get("key", "")]
    print(f"\nAmerican football sports ({len(nfl_related)}):")
    for s in nfl_related:
        print(f"  {s['key']}: active={s['active']}")
except Exception as e:
    print(f"ERROR fetching sports list: {e}")


try:
    section("NFL events")
    resp = requests.get(
        f"{BASE_URL}/sports/americanfootball_nfl/events",
        params={"apiKey": API_KEY},
    )
    print(f"HTTP status: {resp.status_code}")
    print(f"x-requests-remaining: {resp.headers.get('x-requests-remaining')}")
    print(f"x-requests-used: {resp.headers.get('x-requests-used')}")

    events = resp.json()
    print(f"\nNumber of events: {len(events)}")
    if not events:
        print("No NFL events (expected in offseason)")
    else:
        for e in events:
            print(f"  {e['commence_time']}: {e['away_team']} @ {e['home_team']}")
except Exception as e:
    print(f"ERROR fetching NFL events: {e}")
