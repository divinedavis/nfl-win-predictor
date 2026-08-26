"""Snapshot the US sportsbook market for upcoming NFL games.

Cost: 2 credits per run (2 markets x 1 region). The Odds API bills
markets x regions, NOT per bookmaker — asking for `regions=us` returns every
US book it carries for exactly what one book used to cost, so the consensus
is free. Verified against the x-requests-last header.

Writes three files:
  spread_odds.csv        best available price per side (dashboard payout math)
  market_consensus.csv   median no-vig home win probability per game
  spread_odds_history.csv every snapshot, appended, never overwritten

The history file is the seed for an open->close line-movement feature. No free
source of historical opening lines exists (nflverse carries closing numbers
only, and ESPN's open/close blocks come back null for past seasons), so the
only way to get that data is to start recording it. It is not a feature yet
and will not be until a full season has accumulated.

    ODDS_API_KEY=... python fetch_spread_odds.py
"""

import csv
import os
import statistics
import sys
from datetime import datetime, timezone

import requests

from export_web import TEAMS

NAME_TO_ABBR = {name: abbr for abbr, (name, _) in TEAMS.items()}
API = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
HISTORY_CSV = "spread_odds_history.csv"
CONSENSUS_CSV = "market_consensus.csv"
LOW_CREDIT_COST = 2      # markets x regions for this call
LOW_CREDIT_WARN = 100    # below this, a props pull can no longer complete
CREDIT_RESERVE = 10      # never spend the last few; leave room to act
CREDIT_FILE = ".odds_credits"   # last known x-requests-remaining


def american_to_prob(price: float) -> float:
    """American odds -> implied probability (vig included)."""
    price = float(price)
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def schedule_index() -> dict:
    """(season, away, home) -> nflverse game_id, so snapshots join to the
    feature table on the same key it uses. Deriving the week from the kickoff
    date instead would misfile every international and flexed game."""
    import nflreadpy as nfl

    sched = nfl.load_schedules().to_pandas()
    sched = sched[sched["season"] >= datetime.now(timezone.utc).year - 1]
    return {(int(r.season), str(r.away_team), str(r.home_team)): str(r.game_id)
            for r in sched.itertuples(index=False)}


def last_known_credits() -> int | None:
    try:
        with open(CREDIT_FILE) as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


def main() -> None:
    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("No ODDS_API_KEY set — spread payout math will fall back to -110.")

    # Four gameday runs a week on top of the daily one is 22 credits a week for
    # this call alone. Stop before zero rather than after it: the model reads
    # nflverse moneylines when the consensus file is stale, so skipping costs
    # freshness, while running dry costs every other job on the same key.
    known = last_known_credits()
    if known is not None and known - LOW_CREDIT_COST < CREDIT_RESERVE:
        print(f"!!!!!!!!!! ALERT: {known} Odds API credits left, reserve is "
              f"{CREDIT_RESERVE} — SKIPPING the market snapshot. Top up the plan "
              f"or lower CREDIT_RESERVE. !!!!!!!!!!", file=sys.stderr)
        sys.exit(0)

    r = requests.get(API, params={
        "apiKey": key, "regions": "us", "markets": "spreads,h2h",
        "oddsFormat": "american"}, timeout=30)
    r.raise_for_status()
    events = r.json()
    sched = schedule_index()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")

    best: dict = {}       # (home, away, side) -> best price row for the bettor
    history: list = []
    consensus: list = []

    for ev in events:
        h = NAME_TO_ABBR.get(ev["home_team"])
        a = NAME_TO_ABBR.get(ev["away_team"])
        if not h or not a:
            continue
        kickoff = datetime.fromisoformat(ev["commence_time"].replace("Z", "+00:00"))
        season = kickoff.year - 1 if kickoff.month < 3 else kickoff.year
        game_id = sched.get((season, a, h))
        if not game_id:
            continue  # preseason, or a matchup nflverse has not published
        home_probs, spreads = [], []

        for bk in ev.get("bookmakers", []):
            book = bk["key"]
            prices: dict = {}
            for mkt in bk.get("markets", []):
                for oc in mkt.get("outcomes", []):
                    side = NAME_TO_ABBR.get(oc["name"])
                    if side not in (h, a):
                        continue
                    which = "home" if side == h else "away"
                    prices[(mkt["key"], which)] = (oc.get("point"), oc.get("price"))
                    history.append({
                        "fetched_at": stamp, "game_id": game_id,
                        "commence_time": ev["commence_time"], "book": book,
                        "market": mkt["key"], "side": which,
                        "point": oc.get("point"), "price": oc.get("price"),
                    })
                    if mkt["key"] == "spreads":
                        row = {"home": h, "away": a, "side": which,
                               "point": oc.get("point"), "price": oc.get("price"),
                               "book": book}
                        cur = best.get((h, a, which))
                        # Best price = the most favourable number for the
                        # bettor: more points first, then better juice.
                        if cur is None or (
                            (row["point"] or 0), (row["price"] or -10000)
                        ) > ((cur["point"] or 0), (cur["price"] or -10000)):
                            best[(h, a, which)] = row

            hm, am = prices.get(("h2h", "home")), prices.get(("h2h", "away"))
            if hm and am and hm[1] is not None and am[1] is not None:
                ph, pa = american_to_prob(hm[1]), american_to_prob(am[1])
                if ph + pa > 0:
                    home_probs.append(ph / (ph + pa))   # this book, vig removed
            sp = prices.get(("spreads", "home"))
            if sp and sp[0] is not None:
                spreads.append(float(sp[0]))

        if home_probs:
            consensus.append({
                "game_id": game_id, "home": h, "away": a,
                "commence_time": ev["commence_time"],
                # Median, not mean: one stale or off-market book cannot drag it.
                "mkt_prob": round(statistics.median(home_probs), 5),
                "books": len(home_probs),
                # spread_line convention matches nflverse: positive = home favoured.
                "consensus_spread": (round(-statistics.median(spreads), 2)
                                     if spreads else ""),
                "fetched_at": stamp,
            })

    with open("spread_odds.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["home", "away", "side", "point", "price", "book"])
        w.writeheader()
        w.writerows(best.values())

    with open(CONSENSUS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "home", "away", "commence_time",
                                          "mkt_prob", "books", "consensus_spread",
                                          "fetched_at"])
        w.writeheader()
        w.writerows(consensus)

    fresh = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["fetched_at", "game_id", "commence_time",
                                          "book", "market", "side", "point", "price"])
        if fresh:
            w.writeheader()
        w.writerows(history)

    # The plan is a hard ceiling: once credits run out this call starts
    # failing and the consensus silently falls back to the schedule's line.
    # Say so while there is still time to act on it.
    remaining = r.headers.get("x-requests-remaining")
    try:
        if remaining is not None:
            with open(CREDIT_FILE, "w") as f:
                f.write(str(int(remaining)))
        if remaining is not None and int(remaining) < LOW_CREDIT_WARN:
            print(f"!!!!!!!!!! ALERT: only {remaining} Odds API credits left "
                  f"({LOW_CREDIT_COST} per run here, ~640 per props pull) "
                  f"!!!!!!!!!!", file=sys.stderr)
    except ValueError:
        pass

    books = len({row["book"] for row in history})
    print(f"{len(events)} events, {books} books | spread_odds.csv {len(best)} sides "
          f"| market_consensus.csv {len(consensus)} games "
          f"| +{len(history)} history rows | credits used {r.headers.get('x-requests-last')}, "
          f"remaining {r.headers.get('x-requests-remaining')}")


if __name__ == "__main__":
    main()
