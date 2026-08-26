"""Resolve Questionable players from ESPN's live game statuses.

The official injury report is filed Wednesday-Friday. The inactives list lands
90 minutes before kickoff, and the market moves on it while a model built on
Friday's report is still guessing. Measured over 2021-24, that guess covers a
lot of ground: 6,120 Questionable listings, 3.2 per team per game, and only
63.3% of them played — 40.6% for quarterbacks.

This fills the gap with ESPN's public scoreboard, which needs no API key and
carries a per-team status block that flips to Out when inactives post. Writes
inactives.csv; features.py reads it and counts a resolved player in full or
not at all instead of at QUESTIONABLE_WEIGHT.

    python fetch_inactives.py            # this week's games
    python fetch_inactives.py --season 2026 --week 3
"""

import argparse
import csv
import os
import sys
from datetime import datetime, timezone

import requests

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
SUMMARY = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/summary"
OUT_CSV = "inactives.csv"
TIMEOUT = 25

# ESPN status strings that mean the player is not available. "Questionable" is
# deliberately absent: an unresolved Q must stay unresolved, so features.py
# falls back to the league-average sit rate rather than inventing a decision.
OUT_STATUSES = {"out", "inactive", "injured reserve", "suspension",
                "doubtful", "physically unable to perform", "non football injury"}
ACTIVE_STATUSES = {"active", "probable"}

# ESPN abbreviations that differ from nflverse.
ESPN_TO_NFLVERSE = {"WSH": "WAS", "LAR": "LA"}


def canon(team: str) -> str:
    return ESPN_TO_NFLVERSE.get(team.upper(), team.upper())


def fetch(url: str, **params) -> dict:
    r = requests.get(url, params=params or None, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def collect(season: int | None, week: int | None) -> list:
    params = {}
    if season and week:
        params = {"dates": str(season), "seasontype": 2, "week": week}
    board = fetch(SCOREBOARD, **params)
    events = board.get("events", [])
    rows, seen_games = [], 0

    for ev in events:
        try:
            summary = fetch(SUMMARY, event=ev["id"])
        except requests.RequestException as e:
            print(f"  summary {ev['id']} failed: {e}", file=sys.stderr)
            continue
        season_n = ev.get("season", {}).get("year")
        week_n = ev.get("week", {}).get("number")
        # seasontype 2 is the regular season. Without this guard a default
        # (no-argument) run during August records preseason week 3 as if it
        # were regular-season week 3 and poisons a real game's statuses.
        if ev.get("season", {}).get("type") != 2:
            continue
        if not season_n or not week_n:
            continue
        seen_games += 1
        for block in summary.get("injuries", []):
            team = canon(block.get("team", {}).get("abbreviation", ""))
            if not team:
                continue
            for entry in block.get("injuries", []):
                status = str(entry.get("status", "")).strip().lower()
                athlete = entry.get("athlete", {}) or {}
                name = athlete.get("displayName")
                if not name:
                    continue
                if status in OUT_STATUSES:
                    is_out = True
                elif status in ACTIVE_STATUSES:
                    is_out = False
                else:
                    continue  # Questionable and anything unrecognised: unresolved
                rows.append({
                    "season": season_n, "week": week_n, "team": team,
                    "player": name,
                    "position": (athlete.get("position", {}) or {}).get("abbreviation", ""),
                    "status": entry.get("status", ""),
                    "is_out": int(is_out),
                    "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
    print(f"{seen_games} games, {len(rows)} resolved statuses "
          f"({sum(r['is_out'] for r in rows)} out)")
    return rows


def merge(rows: list) -> list:
    """Keep the newest status per (season, week, team, player).

    Runs accumulate through the week — a Friday run records what is known then,
    the Sunday-morning run overwrites it once inactives post — so a later
    fetch always wins, and weeks nobody re-fetched survive untouched.
    """
    keep: dict = {}
    if os.path.exists(OUT_CSV):
        try:
            with open(OUT_CSV, newline="") as f:
                for r in csv.DictReader(f):
                    keep[(r["season"], r["week"], r["team"], r["player"])] = r
        except Exception as e:
            print(f"  could not read existing {OUT_CSV}: {e}", file=sys.stderr)
    for r in rows:
        keep[(str(r["season"]), str(r["week"]), r["team"], r["player"])] = r
    return list(keep.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int)
    ap.add_argument("--week", type=int)
    args = ap.parse_args()

    rows = collect(args.season, args.week)
    if not rows:
        print("nothing resolved; leaving inactives.csv alone")
        return
    merged = merge(rows)
    fields = ["season", "week", "team", "player", "position", "status",
              "is_out", "fetched_at"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(merged)
    print(f"wrote {OUT_CSV} ({len(merged)} rows)")


if __name__ == "__main__":
    main()
