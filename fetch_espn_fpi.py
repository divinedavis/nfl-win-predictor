"""ESPN FPI game projections -> espn_fpi.csv.

FPI is a genuinely independent opinion: a different model, built by different
people from different inputs, with no connection to this project's Elo or to
the betting market. That makes it worth testing as a feature in a way that one
more sportsbook copying the same originator is not.

Unlike every other outside source checked here, it is also **historically
available** -- the predictor endpoint answers for games back to 2015, which is
exactly the backtest window. Kalshi retains two weeks, ESPN's own open/close
odds blocks are null for past seasons, and historical prop lines are paywalled.
This is the one free outside number that can actually be validated before it is
trusted.

The endpoint is public and needs no key. It is also somebody else's server:
requests are throttled, the cache is resumable, and a backfill is meant to be
run once and then topped up weekly.

    python fetch_espn_fpi.py                  # current season, fill gaps
    python fetch_espn_fpi.py --seasons 2015-2026   # full backfill (~10 min)
"""

import argparse
import csv
import os
import sys
import time

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
CORE = "https://sports.core.api.espn.com/v2/sports/football/leagues/nfl"
OUT_CSV = "espn_fpi.csv"

# Politeness: this is a public endpoint with no published rate limit, so the
# budget is self-imposed. A full backfill is ~3,300 calls; at this delay it
# takes about ten minutes and never bursts.
DELAY = 0.12
RETRIES = 3
BACKOFF = 2.0
TIMEOUT = 25

ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}
FIELDS = ["game_id", "season", "week", "away", "home", "espn_event_id",
          "fpi_home_prob", "fpi_raw_home", "fpi_raw_away", "matchup_quality",
          "fetched_at"]


def canon(t: str) -> str:
    return ESPN_TO_NFLVERSE.get(t.upper(), t.upper())


def get(url, **params):
    """GET with bounded retries. Backs off on 429/5xx, gives up quietly."""
    for attempt in range(RETRIES):
        try:
            r = requests.get(url, params=params or None, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(BACKOFF * (attempt + 1))
                continue
            return None
        except requests.RequestException:
            time.sleep(BACKOFF * (attempt + 1))
    return None


def load_cache() -> dict:
    if not os.path.exists(OUT_CSV):
        return {}
    with open(OUT_CSV, newline="") as f:
        return {r["game_id"]: r for r in csv.DictReader(f)}


def projection(event_id: str) -> tuple:
    """(home win prob, raw home, raw away, matchup quality).

    ESPN publishes the two sides as percentages that do not quite sum to 100 --
    the remainder is its tie probability -- so they are renormalised into a
    two-way probability to match every other number in this project.
    """
    d = get(f"{CORE}/events/{event_id}/competitions/{event_id}/predictor")
    if not d:
        return None
    def stat(side, name):
        for s in d.get(side, {}).get("statistics", []):
            if s.get("name") == name:
                return float(s.get("value"))
        return None
    h, a = stat("homeTeam", "gameProjection"), stat("awayTeam", "gameProjection")
    if h is None or a is None or (h + a) <= 0:
        return None
    return h / (h + a), h, a, stat("homeTeam", "matchupQuality")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="", help="e.g. 2015-2026 or 2026")
    ap.add_argument("--weeks", default="1-18")
    args = ap.parse_args()

    if args.seasons:
        a, _, b = args.seasons.partition("-")
        seasons = range(int(a), int(b or a) + 1)
    else:
        seasons = [time.gmtime().tm_year if time.gmtime().tm_mon >= 3
                   else time.gmtime().tm_year - 1]
    w0, _, w1 = args.weeks.partition("-")
    weeks = range(int(w0), int(w1 or w0) + 1)

    cache = load_cache()
    added = skipped = missing = 0
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def flush() -> None:
        """Write the cache out. Called after every week, not just at the end:
        a full backfill is ~10 minutes of somebody else's rate limit, and a
        run that dies at minute nine should not throw away minutes one to
        eight. The cache is keyed by game_id, so a resumed run skips whatever
        already landed."""
        tmp = OUT_CSV + ".tmp"
        with open(tmp, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
            w.writeheader()
            for r in sorted(cache.values(), key=lambda r: r["game_id"]):
                w.writerow(r)
        os.replace(tmp, OUT_CSV)

    for season in seasons:
        for week in weeks:
            board = get(SCOREBOARD, dates=str(season), seasontype=2, week=week)
            if not board:
                continue
            for ev in board.get("events", []):
                comp = (ev.get("competitions") or [{}])[0]
                teams = {c.get("homeAway"): canon(
                    c.get("team", {}).get("abbreviation", "")) for c in comp.get("competitors", [])}
                home, away = teams.get("home"), teams.get("away")
                if not home or not away:
                    continue
                gid = f"{season}_{week:02d}_{away}_{home}"
                if gid in cache:
                    skipped += 1
                    continue
                p = projection(ev["id"])
                time.sleep(DELAY)
                if not p:
                    missing += 1
                    continue
                prob, rh, ra, mq = p
                cache[gid] = {"game_id": gid, "season": season, "week": week,
                              "away": away, "home": home, "espn_event_id": ev["id"],
                              "fpi_home_prob": round(prob, 5),
                              "fpi_raw_home": rh, "fpi_raw_away": ra,
                              "matchup_quality": mq, "fetched_at": stamp}
                added += 1
            print(f"  {season} wk{week:>2}: +{added} cached (skipped {skipped}, "
                  f"no projection {missing})", file=sys.stderr, flush=True)
            flush()

    flush()
    print(f"wrote {OUT_CSV}: {len(cache)} games (+{added} new, {missing} without a projection)")


if __name__ == "__main__":
    main()
