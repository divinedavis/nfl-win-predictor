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
from pathlib import Path

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon
from export_web import PROJECTION_LOG, STAT_LABELS

CHUNK = 500


def _post(url: str, key: str, table: str, rows: list) -> None:
    """Upsert in chunks. on_conflict names the primary key so a re-run
    overwrites rather than colliding — every refresh republishes the season."""
    if not rows:
        return
    pk = "season,week,ref"
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


def game_rows() -> list:
    sched = nfl.load_schedules([LAST_SEASON]).to_pandas()
    sched = sched[(sched.game_type == "REG") & sched.home_score.notna()]
    rows = []
    for r in sched.itertuples(index=False):
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


def main() -> None:
    url = os.environ.get("SUPABASE_URL", "").rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY", "")
    if not url or not key:
        raise SystemExit("SUPABASE_URL / SUPABASE_SERVICE_KEY not set — "
                         "nothing published")
    _post(url, key, "game_results", game_rows())
    _post(url, key, "prop_results", prop_rows())


if __name__ == "__main__":
    main()
