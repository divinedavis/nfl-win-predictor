"""Outside opinions on this week's FBS games -> cfb_sources.csv.

  fpi     ESPN's per-game FPI win probability, from the same public predictor
          endpoint the NFL page uses, one call per unplayed game in the next
          two weeks (about 150 calls, throttled).
  kalshi  the exchange's mid-price for the home team, matched to ESPN teams
          by name: Kalshi tickers carry its own short codes, never ESPN ids.

Both are free and unmetered. Either may be missing for any game, and the
page simply shows the sources it has.

    python -m cfb.fetch_sources
"""

import csv
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from cfb.data import LAST_SEASON, Teams, load_schedule

OUT = "cfb_sources.csv"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/college-football"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
DELAY = 0.12
TIMEOUT = 25
MAX_SPREAD = 0.15      # a wider bid-ask than this is an empty book, not a price
WEEKS_AHEAD = 2


def _norm(s: str) -> str:
    s = s.lower().replace("st.", "state").replace("&", "and")
    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def fpi_for(game_id: int) -> float | None:
    url = f"{CORE}/events/{game_id}/competitions/{game_id}/predictor"
    try:
        r = requests.get(url, timeout=TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        stats = r.json()["homeTeam"]["statistics"]
        for s in stats:
            if s.get("name") == "gameProjection":
                return float(s["value"]) / 100.0
    except (KeyError, ValueError, TypeError):
        return None
    return None


def kalshi_events() -> list:
    out, cursor = [], None
    for _ in range(20):
        params = {"series_ticker": "KXNCAAFGAME", "status": "open",
                  "limit": 200, "with_nested_markets": "true"}
        if cursor:
            params["cursor"] = cursor
        try:
            r = requests.get(f"{KALSHI}/events", params=params, timeout=TIMEOUT)
        except requests.RequestException:
            break
        if r.status_code != 200:
            break
        d = r.json()
        out.extend(d.get("events", []))
        cursor = d.get("cursor")
        if not cursor:
            break
        time.sleep(DELAY)
    return out


def main() -> None:
    teams = Teams()
    sched = load_schedule()
    now = datetime.now(timezone.utc)
    up = sched[(sched.season == LAST_SEASON) & sched.home_score.isna()]
    up = up[(up.game_date >= now - pd.Timedelta(days=1))
            & (up.game_date <= now + pd.Timedelta(days=7 * WEEKS_AHEAD))]

    # Name index for Kalshi: every alias of every team on the slate.
    alias: dict = {}
    for tid in set(up.home_id) | set(up.away_id):
        for n in teams.names_for_match(tid):
            alias.setdefault(_norm(n), int(tid))
    kal: dict = {}
    for ev in kalshi_events():
        sides = {}
        for m in ev.get("markets", []):
            name = _norm(str(m.get("yes_sub_title") or ""))
            tid = alias.get(name)
            if tid is None:
                continue
            try:
                bid, ask = float(m["yes_bid_dollars"]), float(m["yes_ask_dollars"])
            except (KeyError, TypeError, ValueError):
                continue
            if ask - bid > MAX_SPREAD or ask <= 0:
                continue
            sides[tid] = (bid + ask) / 2
        if len(sides) == 2:
            kal[frozenset(sides)] = sides

    rows = []
    for r in up.itertuples(index=False):
        p_fpi = fpi_for(int(r.game_id))
        time.sleep(DELAY)
        k = kal.get(frozenset({int(r.home_id), int(r.away_id)}))
        p_kal = k.get(int(r.home_id)) if k else None
        rows.append({"game_id": int(r.game_id), "fpi_home_prob": p_fpi,
                     "kalshi_home_prob": p_kal,
                     "fetched_at": now.strftime("%Y-%m-%dT%H:%M:%SZ")})
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["game_id", "fpi_home_prob",
                                          "kalshi_home_prob", "fetched_at"])
        w.writeheader()
        w.writerows(rows)
    n_f = sum(1 for x in rows if x["fpi_home_prob"] is not None)
    n_k = sum(1 for x in rows if x["kalshi_home_prob"] is not None)
    print(f"Wrote {OUT}: {len(rows)} games, FPI {n_f}, Kalshi {n_k}")


if __name__ == "__main__":
    main()
