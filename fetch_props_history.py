"""Archive historical DraftKings player-prop closing lines from The Odds API
into props_lines_historical/<season>.csv — pulled once while a paid tier is
active, stored locally forever (gitignored: purchased data, public repo).

Cost: 40 credits per game (4 markets x 10) + 1 per event-id snapshot.
Stops cleanly at --floor remaining credits; reruns skip games already saved,
so a season can be filled across multiple runs or billing months.

    ODDS_API_KEY=... python fetch_props_history.py --season 2025 [--floor 300]
"""

import argparse
import csv
import os
import sys
import time
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import nflreadpy as nfl

from export_web import TEAMS
from fetch_props import MARKETS, BASE

OUT_DIR = Path("props_lines_historical")
NAME_TO_ABBR = {name: abbr for abbr, (name, _) in TEAMS.items()}
ET = ZoneInfo("America/New_York")
SNAP_BEFORE_MIN = 30      # snapshot 30 minutes before kickoff = closing-ish


def kickoff_utc(gameday: str, gametime) -> pd.Timestamp:
    hhmm = str(gametime) if isinstance(gametime, str) and ":" in str(gametime) \
        else "13:00"
    local = pd.Timestamp(f"{gameday} {hhmm}").tz_localize(ET)
    return local.tz_convert("UTC")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--floor", type=int, default=300,
                    help="stop when this many credits remain")
    args = ap.parse_args()

    key = os.environ.get("ODDS_API_KEY")
    if not key:
        sys.exit("Set ODDS_API_KEY")

    OUT_DIR.mkdir(exist_ok=True)
    out_path = OUT_DIR / f"{args.season}.csv"
    fields = ["game_id", "season", "week", "game_type", "snapshot", "commence",
              "event_id", "home", "away", "stat", "player", "side", "line",
              "odds", "book"]
    done_games = set()
    if out_path.exists():
        done_games = set(pd.read_csv(out_path)["game_id"].unique())
    else:
        with out_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    sched = nfl.load_schedules([args.season]).to_pandas()
    sched = sched[sched["game_type"] == "REG"].sort_values(["gameday"])
    print(f"{args.season}: {len(sched)} REG games, {len(done_games)} already saved")

    # Map schedule games to API event ids via sparse listing snapshots (1
    # credit each): each listing shows all upcoming events, so one snapshot
    # per week of kickoffs covers everything after it.
    event_ids: dict = {}       # (utc_kick_iso, home_abbr, away_abbr) -> id
    remaining = None

    def listing(snap_iso: str):
        nonlocal remaining
        r = requests.get(
            f"{BASE}/historical/sports/americanfootball_nfl/events",
            params={"apiKey": key, "date": snap_iso}, timeout=30)
        r.raise_for_status()
        remaining = int(float(r.headers.get("x-requests-remaining", 0)))
        for ev in r.json()["data"]:
            h = NAME_TO_ABBR.get(ev["home_team"])
            a = NAME_TO_ABBR.get(ev["away_team"])
            if h and a:
                event_ids[(ev["commence_time"], h, a)] = ev["id"]

    fetched = skipped = 0
    week_seen = set()
    with out_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        for g in sched.itertuples(index=False):
            if g.game_id in done_games:
                continue
            kick = kickoff_utc(g.gameday, g.gametime)
            snap = (kick - timedelta(minutes=SNAP_BEFORE_MIN)) \
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            kick_iso = kick.strftime("%Y-%m-%dT%H:%M:%SZ")
            home, away = g.home_team, g.away_team

            if remaining is not None and remaining <= args.floor:
                print(f"Credit floor reached ({remaining}); stopping.")
                break
            if g.week not in week_seen:
                week_seen.add(g.week)
                listing(snap)

            ev_id = event_ids.get((kick_iso, home, away))
            if ev_id is None:
                # kickoff moved (flex) — listing at this game's own snapshot
                listing(snap)
                ev_id = event_ids.get((kick_iso, home, away))
            if ev_id is None:
                print(f"  no event id: {g.game_id}")
                skipped += 1
                continue

            r = requests.get(
                f"{BASE}/historical/sports/americanfootball_nfl/events/{ev_id}/odds",
                params={"apiKey": key, "date": snap,
                        "bookmakers": "draftkings",
                        "markets": ",".join(MARKETS),
                        "oddsFormat": "american"},
                timeout=30)
            remaining = int(float(r.headers.get("x-requests-remaining",
                                                remaining or 0)))
            if r.status_code != 200:
                print(f"  {g.game_id}: HTTP {r.status_code}")
                skipped += 1
                continue
            n = 0
            for bk in r.json()["data"].get("bookmakers", []):
                for market in bk.get("markets", []):
                    stat = MARKETS.get(market["key"])
                    if not stat:
                        continue
                    for oc in market.get("outcomes", []):
                        writer.writerow({
                            "game_id": g.game_id, "season": g.season,
                            "week": g.week, "game_type": g.game_type,
                            "snapshot": snap, "commence": kick_iso,
                            "event_id": ev_id, "home": home, "away": away,
                            "stat": stat, "player": oc.get("description", ""),
                            "side": oc.get("name", ""),
                            "line": oc.get("point"), "odds": oc.get("price"),
                            "book": bk["key"],
                        })
                        n += 1
            f.flush()
            fetched += 1
            if fetched % 25 == 0:
                print(f"  {fetched} games fetched, credits left {remaining}")
            time.sleep(0.25)

    print(f"Done: {fetched} games fetched, {skipped} skipped, "
          f"credits remaining {remaining}")


if __name__ == "__main__":
    main()
