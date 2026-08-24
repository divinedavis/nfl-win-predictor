"""Publish game and player results to Supabase so picks can be graded.

The picks page lets a visitor call a winner or an over/under. Grading those in
the browser would make the leaderboard editable with dev tools, so the answers
live in Postgres and only this script — running from cron with the service
key — writes them.

Needs SUPABASE_URL and SUPABASE_SERVICE_KEY in the environment; refresh.sh
sources .env, which is not in the repo. Does nothing (loudly) without them.

    python publish_results.py
"""

import json
import os
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon
from export_web import PROJECTION_LOG, STAT_LABELS

CHUNK = 500


PK = {"game_results": "season,week,ref",
      "prop_results": "season,week,ref",
      "pick_deadlines": "season,week,kind,ref"}


def _kickoffs(sched: pd.DataFrame) -> dict:
    """(week, away, home) and (week, team) -> kickoff instant.

    gametime is Eastern wall-clock, so it goes through America/New_York rather
    than a fixed offset: the season crosses the end of daylight saving."""
    ET = ZoneInfo("America/New_York")
    by_game, by_team = {}, {}
    for r in sched.itertuples(index=False):
        if not isinstance(r.gametime, str) or ":" not in r.gametime:
            continue
        hh, mm = r.gametime.split(":")[:2]
        stamp = datetime.combine(pd.Timestamp(r.gameday).date(),
                                 datetime.min.time(), tzinfo=ET)
        stamp = stamp.replace(hour=int(hh), minute=int(mm))
        iso = stamp.astimezone(ZoneInfo("UTC")).isoformat()
        home, away, week = canon(r.home_team), canon(r.away_team), int(r.week)
        by_game[(week, away, home)] = iso
        by_team[(week, home)] = iso
        by_team[(week, away)] = iso
    return {"game": by_game, "team": by_team}


def _post(url: str, key: str, table: str, rows: list) -> None:
    """Upsert in chunks. on_conflict names the primary key so a re-run
    overwrites rather than colliding — every refresh republishes the season."""
    if not rows:
        return
    pk = PK[table]
    for i in range(0, len(rows), CHUNK):
        body = json.dumps(rows[i:i + CHUNK]).encode()
        req = urllib.request.Request(
            f"{url}/rest/v1/{table}?on_conflict={pk}", data=body, method="POST",
            headers={"apikey": key, "Authorization": f"Bearer {key}",
                     "Content-Type": "application/json",
                     "Prefer": "resolution=merge-duplicates,return=minimal"})
        with urllib.request.urlopen(req, timeout=60) as r:
            if r.status >= 300:
                raise SystemExit(f"{table}: HTTP {r.status}")
    print(f"{table}: {len(rows)} rows")


def game_rows(sched: pd.DataFrame) -> list:
    rows = []
    for r in sched[sched.home_score.notna()].itertuples(index=False):
        home, away = canon(r.home_team), canon(r.away_team)
        if r.home_score == r.away_score:
            continue          # a tie settles no pick either way
        rows.append({"season": int(r.season), "week": int(r.week),
                     "ref": f"{away}@{home}",
                     "winner": home if r.home_score > r.away_score else away})
    return rows


def prop_rows() -> list:
    """Only players the page actually offered — the projection log is the
    record of what anyone could have picked."""
    if not PROJECTION_LOG.exists():
        return []
    log = pd.read_csv(PROJECTION_LOG)
    log = log[log["season"] == LAST_SEASON]
    if log.empty:
        return []
    try:
        wk = (nfl.load_player_stats([LAST_SEASON], summary_level="week")
              .select(["season", "week", "player_id"] + list(STAT_LABELS))
              .to_pandas())
    except Exception:
        return []             # season not published yet
    rows = []
    for stat in STAT_LABELS:
        want = log[log["stat"] == stat][["week", "player_id"]].drop_duplicates()
        merged = want.merge(wk[["week", "player_id", stat]],
                            on=["week", "player_id"], how="inner")
        for r in merged.itertuples(index=False):
            v = getattr(r, stat)
            if pd.notna(v):
                rows.append({"season": LAST_SEASON, "week": int(r.week),
                             "ref": f"{r.player_id}|{stat}",
                             "actual": round(float(v), 2)})
    return rows


def deadline_rows(sched: pd.DataFrame) -> list:
    """When each pickable thing closes. Games come straight off the schedule;
    a player prop closes when his team plays, so it inherits that kickoff.

    Every game in the season is published, not only the ones already offered,
    so the lock is in place before anyone can reach a row — the trigger lets
    an unknown ref through, and a missing deadline would be an open door."""
    kicks = _kickoffs(sched)
    rows = [{"season": LAST_SEASON, "week": wk, "kind": "game",
             "ref": f"{away}@{home}", "kickoff": iso}
            for (wk, away, home), iso in kicks["game"].items()]
    if PROJECTION_LOG.exists():
        log = pd.read_csv(PROJECTION_LOG)
        log = log[(log["season"] == LAST_SEASON) & log["team"].notna()]
        for r in log.itertuples(index=False):
            iso = kicks["team"].get((int(r.week), canon(r.team)))
            if iso:
                rows.append({"season": LAST_SEASON, "week": int(r.week),
                             "kind": "prop",
                             "ref": f"{r.player_id}|{r.stat}", "kickoff": iso})
    return rows


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — "
                         "nothing published")
    sched = nfl.load_schedules([LAST_SEASON]).to_pandas()
    sched = sched[sched.game_type == "REG"]
    _post(url, key, "pick_deadlines", deadline_rows(sched))
    _post(url, key, "game_results", game_rows(sched))
    _post(url, key, "prop_results", prop_rows())


if __name__ == "__main__":
    main()
