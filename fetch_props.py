"""Pull current NFL player-prop lines from The Odds API into props_lines/.

Needs a free API key from https://the-odds-api.com (500 credits/month) in the
ODDS_API_KEY environment variable. Budget guard: one snapshot per day max,
four markets, DraftKings only — a weekly pull uses well under half the free
tier. The script prints the credit count the API reports after each run.

    ODDS_API_KEY=... python fetch_props.py
"""

import csv
import os
import sys
from datetime import date
from pathlib import Path

import requests

MARKETS = {
    "player_pass_yds": "passing_yards",
    "player_rush_yds": "rushing_yards",
    "player_reception_yds": "receiving_yards",
    "player_receptions": "receptions",
}
BOOKMAKER = "draftkings"
BASE = "https://api.the-odds-api.com/v4"
OUT_DIR = Path("props_lines")
MAX_EVENTS = 16  # one NFL week; hard cap on API spend


def main() -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY (free key: https://the-odds-api.com). "
                 "No lines pulled.")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{date.today().isoformat()}.csv"
    if out_path.exists():
        sys.exit(f"{out_path} already exists — one snapshot per day. "
                 "Delete it to re-pull.")

    events = requests.get(
        f"{BASE}/sports/americanfootball_nfl/events",
        params={"apiKey": key}, timeout=20)
    events.raise_for_status()
    rows, remaining = [], "?"
    for ev in events.json()[:MAX_EVENTS]:
        r = requests.get(
            f"{BASE}/sports/americanfootball_nfl/events/{ev['id']}/odds",
            params={"apiKey": key, "regions": "us",
                    "bookmakers": BOOKMAKER,
                    "markets": ",".join(MARKETS),
                    "oddsFormat": "american"},
            timeout=20)
        if r.status_code != 200:
            print(f"skip {ev.get('away_team')} @ {ev.get('home_team')}: "
                  f"{r.status_code}", file=sys.stderr)
            continue
        remaining = r.headers.get("x-requests-remaining", remaining)
        for bk in r.json().get("bookmakers", []):
            for market in bk.get("markets", []):
                stat = MARKETS.get(market["key"])
                if not stat:
                    continue
                for oc in market.get("outcomes", []):
                    rows.append({
                        "pulled": date.today().isoformat(),
                        "commence": ev.get("commence_time", ""),
                        "home_team": ev.get("home_team", ""),
                        "away_team": ev.get("away_team", ""),
                        "stat": stat,
                        "player": oc.get("description", ""),
                        "side": oc.get("name", ""),   # Over / Under
                        "line": oc.get("point"),
                        "odds": oc.get("price"),
                        "book": bk["key"],
                    })

    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                           ["pulled"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_path} ({len(rows)} outcomes). "
          f"API credits remaining this month: {remaining}")


if __name__ == "__main__":
    main()
