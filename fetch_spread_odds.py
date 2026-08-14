"""Snapshot current DraftKings spread prices (line + juice, both sides) for
upcoming NFL games into spread_odds.csv — feeds the dashboard's payout math.
Cost: 1 credit per run (current-odds bulk endpoint, one market, one book).

    ODDS_API_KEY=... python fetch_spread_odds.py
"""

import csv
import os
import sys

import requests

from export_web import TEAMS

NAME_TO_ABBR = {name: abbr for abbr, (name, _) in TEAMS.items()}


def main() -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("No ODDS_API_KEY set — spread payout math will fall back to -110.")
    r = requests.get(
        "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds",
        params={"apiKey": key, "regions": "us", "bookmakers": "draftkings",
                "markets": "spreads", "oddsFormat": "american"},
        timeout=30)
    r.raise_for_status()
    rows = []
    for ev in r.json():
        h, a = NAME_TO_ABBR.get(ev["home_team"]), NAME_TO_ABBR.get(ev["away_team"])
        if not h or not a:
            continue
        for bk in ev.get("bookmakers", []):
            for mkt in bk.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    side = NAME_TO_ABBR.get(oc["name"])
                    if side in (h, a):
                        rows.append({"home": h, "away": a,
                                     "side": "home" if side == h else "away",
                                     "point": oc.get("point"),
                                     "price": oc.get("price")})
    with open("spread_odds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["home", "away", "side", "point", "price"])
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote spread_odds.csv ({len(rows)} sides), credits remaining "
          f"{r.headers.get('x-requests-remaining')}")


if __name__ == "__main__":
    main()
