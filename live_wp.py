"""Live win probability, priced off the state of the game as it stands.

The pregame model answers one question once. The world model can be restarted
from any state, so the same dynamics that price a game on Thursday price it
again on 3rd and 7 from the opponent's 34 with four minutes left -- something a
game-level classifier structurally cannot do, because its only input is a
pregame feature vector that stopped being true at kickoff.

State comes from ESPN's public scoreboard feed: free, no key, no account, and
it carries exactly the fields the simulator needs.

    situation.down / .distance / .yardsToEndzone   ->  down, ydstogo, yardline_100
    situation.possession                           ->  who has the ball
    status.period / .displayClock                  ->  half and clock
    competitors[].score                            ->  score

`yardsToEndzone` is the same quantity nflverse calls `yardline_100`, so no
conversion is needed and no orientation can be got backwards.

Team strength still comes from features.parquet, so a live number inherits the
same Elo, rolling EPA, injury and weather work as the pregame one.

    python live_wp.py                 # every game in progress right now
    python live_wp.py --json          # payload for the dashboard
    python live_wp.py --date 20251123 # a past slate, for checking the wiring
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

import pandas as pd

from simulate import Simulator, game_seed

SCOREBOARD = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
TIMEOUT = 10
# Send no User-Agent override at all. ESPN's edge 403s unrecognised custom
# agents *and* partial browser strings, but passes urllib's own default -- so
# the honest client identifier is also the one that works. Measured, not
# guessed: curl/8.7.1 and Python-urllib both return 200 while
# "nfl-predictor/1.0" and a short "Mozilla/5.0 ... Chrome/126.0" both return
# 403. Do not "fix" this by inventing a browser string; that is what breaks it.
MAX_TRIES = 3
BACKOFF_SECONDS = 5
# ESPN spells two clubs differently from nflverse; the rest match.
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA"}

# Quarter lengths, used to turn "period 3, 7:12 on the clock" into seconds
# remaining in the half -- which is what the heads were trained on.
QUARTER_SECONDS = 900


def fetch_scoreboard(date: str | None = None) -> dict:
    """One request, with a bounded retry so a rate limit is not a failure.

    Bounded on purpose: this runs on a schedule, and an unbounded retry against
    someone else's free endpoint is how you get blocked for good.
    """
    url = SCOREBOARD + (f"?dates={date}" if date else "")
    req = urllib.request.Request(url)
    for attempt in range(MAX_TRIES):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return json.load(r)
        except urllib.error.HTTPError as exc:
            # 429 is a rate limit worth waiting out; a 403 here means the agent
            # was rejected and retrying will not change that.
            if exc.code != 429 or attempt == MAX_TRIES - 1:
                raise
            time.sleep(BACKOFF_SECONDS * (attempt + 1))
    raise RuntimeError("unreachable")


def _clock_seconds(display: str) -> float:
    """'7:12' -> 432. ESPN occasionally sends 'MM:SS' or plain seconds."""
    try:
        if ":" in display:
            m, s = display.split(":")[-2:]
            return int(m) * 60 + int(s)
        return float(display)
    except (ValueError, AttributeError):
        return 0.0


def parse_games(payload: dict) -> list[dict]:
    """Pull the live state out of every game that has one."""
    games = []
    for event in payload.get("events", []):
        comp = (event.get("competitions") or [{}])[0]
        status = event.get("status", {})
        state = status.get("type", {}).get("state")
        situation = comp.get("situation") or {}
        teams = {}
        for c in comp.get("competitors", []):
            abbr = c.get("team", {}).get("abbreviation", "")
            teams[c.get("homeAway")] = {
                "team": ESPN_TEAM.get(abbr, abbr),
                "id": c.get("team", {}).get("id"),
                "score": int(c.get("score") or 0),
            }
        if "home" not in teams or "away" not in teams:
            continue

        period = int(status.get("period") or 0)
        clock = _clock_seconds(status.get("displayClock", "0:00"))
        # Seconds left in the current half; overtime is its own phase.
        if period <= 2:
            half, half_clock = 1, clock + (QUARTER_SECONDS if period == 1 else 0)
        elif period <= 4:
            half, half_clock = 2, clock + (QUARTER_SECONDS if period == 3 else 0)
        else:
            half, half_clock = 3, clock

        pos_id = situation.get("possession")
        games.append({
            "event_id": event.get("id"),
            "name": event.get("shortName"),
            "state": state,
            "home_team": teams["home"]["team"], "away_team": teams["away"]["team"],
            "home_score": teams["home"]["score"], "away_score": teams["away"]["score"],
            "period": period, "clock": status.get("displayClock"),
            "half": half, "half_seconds_remaining": half_clock,
            "down": situation.get("down"),
            "ydstogo": situation.get("distance"),
            "yardline_100": situation.get("yardsToEndzone"),
            "pos_is_home": (pos_id is not None
                            and str(pos_id) == str(teams["home"]["id"])),
            "summary": situation.get("downDistanceText"),
        })
    return games


def find_row(feat: pd.DataFrame, home: str, away: str) -> pd.Series | None:
    """The pregame feature row for this matchup, newest first."""
    m = feat[(feat.home_team == home) & (feat.away_team == away)]
    if m.empty:
        return None
    return m.sort_values(["season", "week"]).iloc[-1]


def price(sim: Simulator, row: pd.Series, g: dict, n: int) -> dict:
    """Simulate the rest of the game from where it actually is."""
    out = sim.simulate_from(
        row,
        pos_is_home=g["pos_is_home"],
        yardline_100=float(g["yardline_100"]),
        down=int(g["down"]), ydstogo=float(g["ydstogo"]),
        half_seconds_remaining=float(g["half_seconds_remaining"]),
        half=int(g["half"]),
        home_score=int(g["home_score"]), away_score=int(g["away_score"]),
        n=n, seed=game_seed(f"{g['event_id']}:{g['period']}:{g['clock']}"),
    )
    return {
        "home_win_prob": out["home_win_prob"],
        "proj_home": float(out["home_score"].mean()),
        "proj_away": float(out["away_score"].mean()),
        "proj_margin": float(out["margin"].mean()),
    }


def write_payload(rows: list, path: str) -> None:
    """Write live.json atomically.

    The browser polls this file every 45 seconds; a partially written one would
    surface as a parse error mid-game. Rename is atomic on the same filesystem,
    so a reader sees either the old payload or the new one.
    """
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(rows, fh, separators=(",", ":"))
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="world_model.pkl")
    ap.add_argument("--date", default=None, help="YYYYMMDD, for checking the wiring")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--json", action="store_true", help="emit the dashboard payload")
    ap.add_argument("--out", default=None,
                    help="write the payload here instead of stdout, atomically")
    args = ap.parse_args()

    try:
        payload = fetch_scoreboard(args.date)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        # A live feed being down must never take the daily rebuild with it.
        # Leave whatever is on disk alone: a blip in someone else's feed must
        # not blank out a live game that is still being played.
        print(f"scoreboard unavailable: {exc}", file=sys.stderr)
        if args.json:
            print("[]")
        return

    games = parse_games(payload)
    playable = [g for g in games
                if g["state"] == "in" and g["down"] and g["yardline_100"] is not None]
    if not playable:
        states = ", ".join(sorted({g["state"] or "?" for g in games})) or "none"
        print(f"no game is mid-drive right now ({len(games)} on the slate: {states})",
              file=sys.stderr)
        if args.out:
            write_payload([], args.out)
        if args.json:
            print("[]")
        return

    feat = pd.read_parquet("features.parquet")
    sim = Simulator.load(args.model)
    rows = []
    for g in playable:
        row = find_row(feat, g["home_team"], g["away_team"])
        if row is None:
            print(f"  no pregame row for {g['name']}, skipping", file=sys.stderr)
            continue
        r = price(sim, row, g, args.n)
        rows.append({**{k: g[k] for k in
                        ("event_id", "name", "home_team", "away_team",
                         "home_score", "away_score", "period", "clock", "summary")},
                     **r})

    if args.out:
        write_payload(rows, args.out)
        print(f"wrote {len(rows)} live game(s) to {args.out}", file=sys.stderr)
    if args.json:
        print(json.dumps(rows, indent=1))
        return
    for r in rows:
        print(f"{r['away_team']} {r['away_score']} @ {r['home_team']} {r['home_score']}"
              f"   Q{r['period']} {r['clock']}   {r['summary'] or ''}")
        print(f"   {r['home_team']} win {r['home_win_prob']:.1%}   "
              f"projected final {r['proj_away']:.0f}-{r['proj_home']:.0f}")


if __name__ == "__main__":
    main()
