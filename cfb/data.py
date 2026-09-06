"""Every college data source, in one place.

Two kinds of source, both keyless:

  sportsdataverse releases   season parquet files rebuilt nightly by the
                             cfbfastR maintainers: schedules with scores,
                             per-team-game box + EPA, ESPN spreads, player
                             box scores, rosters, team metadata, talent
  ESPN public API            the scoreboard for the CURRENT season's full
                             slate (the releases only carry games already
                             played), the per-game FPI predictor, injuries

Files land in cfb_cache/ (gitignored). Past seasons never change, so they
download once; the current season's files are re-pulled when older than
REFRESH_HOURS so a daily cron picks up new results without hammering GitHub.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

CACHE = Path("cfb_cache")
RELEASE = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SCOREBOARD = ("https://site.api.espn.com/apis/site/v2/sports/football/"
              "college-football/scoreboard")
FIRST_SEASON = 2005
LAST_SEASON = 2026
REFRESH_HOURS = 6
TIMEOUT = 60
FBS_GROUP = 80          # ESPN's group id for the FBS slate

# kind -> (release tag, file pattern)
ASSETS = {
    "schedule": ("espn_cfb_schedules", "cfb_schedule_{y}.parquet"),
    "adv": ("espn_cfb_adv_team_gamelog", "adv_team_gamelog_{y}.parquet"),
    "betting": ("espn_cfb_betting", "betting_{y}.parquet"),
    "player_box": ("espn_cfb_player_box", "player_box_{y}.parquet"),
    "rosters": ("espn_cfb_rosters", "cfb_rosters_{y}.parquet"),
    "teams": ("espn_cfb_teams", "cfb_teams_{y}.parquet"),
    "talent": ("cfb_team_talent", "cfb_team_talent_{y}.parquet"),
}
INJURIES = ("espn_cfb_injuries", "injuries_{y}.parquet")


def _stale(path: Path, season: int) -> bool:
    if not path.exists() or path.stat().st_size < 200:
        return True
    if season < LAST_SEASON:
        return False
    return time.time() - path.stat().st_mtime > REFRESH_HOURS * 3600


def fetch(kind: str, season: int, refresh: bool = True) -> Path | None:
    """Path to the cached parquet for one season, downloading if needed.
    None when the release has no file for that season (early years of a
    dataset, or a season that has not started)."""
    tag, pat = ASSETS[kind] if kind in ASSETS else INJURIES
    CACHE.mkdir(exist_ok=True)
    path = CACHE / pat.format(y=season)
    if refresh and _stale(path, season):
        url = f"{RELEASE}/{tag}/{pat.format(y=season)}"
        try:
            r = requests.get(url, timeout=TIMEOUT)
        except requests.RequestException:
            return path if path.exists() else None
        if r.status_code == 200 and len(r.content) > 200:
            path.write_bytes(r.content)
        elif not path.exists():
            return None
        else:
            path.touch()   # keep the copy we have; try again next cycle
    return path if path.exists() else None


def load(kind: str, seasons=None, refresh: bool = True,
         columns: list | None = None) -> pd.DataFrame:
    """Concatenate one dataset across seasons. Pass `columns` for the big
    ones: the player box has 61 columns and 22 seasons of it will not fit
    in the droplet's 2 GB alongside everything else."""
    seasons = list(seasons or range(FIRST_SEASON, LAST_SEASON + 1))
    frames = []
    for y in seasons:
        p = fetch(kind, y, refresh)
        if p is None:
            continue
        try:
            if columns:
                import pyarrow.parquet as pq
                have = set(pq.read_schema(p).names)
                d = pd.read_parquet(p, columns=[c for c in columns if c in have])
            else:
                d = pd.read_parquet(p)
        except Exception:
            continue
        if "season" not in d.columns:
            d["season"] = y
        frames.append(d)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------- teams ----
class Teams:
    """team_id -> display fields, keyed off the newest season's team file
    and back-filled from older ones for programs that no longer exist.

    Abbreviations are unique among FBS teams but not across all 800+
    programs, so FCS opponents that collide with an FBS code get a suffix:
    the page keys everything by abbreviation."""

    def __init__(self):
        frames = []
        for y in range(LAST_SEASON, FIRST_SEASON - 1, -1):
            p = fetch("teams", y)
            if p is not None:
                frames.append(pd.read_parquet(p))
        t = pd.concat(frames, ignore_index=True)
        t["team_id"] = t["team_id"].astype(int)
        t = t.drop_duplicates("team_id", keep="first")   # newest wins
        latest = frames[0]
        self.fbs = set(latest[latest["is_fbs"] == True]["team_id"].astype(int))
        self.rows = {int(r.team_id): r for r in t.itertuples(index=False)}
        self._abbr: dict = {}
        taken = set()
        # FBS first so they always keep their real code.
        order = sorted(self.rows, key=lambda i: (i not in self.fbs, i))
        for tid in order:
            r = self.rows[tid]
            ab = str(r.abbreviation or "").upper() or f"T{tid}"
            if ab in taken:
                ab = f"{ab}{str(tid)[-2:]}"
                while ab in taken:
                    ab += "X"
            taken.add(ab)
            self._abbr[tid] = ab

    def abbr(self, tid) -> str:
        tid = int(tid)
        if tid not in self._abbr:
            self._abbr[tid] = f"T{tid}"
        return self._abbr[tid]

    def name(self, tid) -> str:
        r = self.rows.get(int(tid))
        return str(r.display_name) if r is not None else f"Team {tid}"

    def short(self, tid) -> str:
        r = self.rows.get(int(tid))
        return str(r.short_display_name) if r is not None else f"Team {tid}"

    def conf(self, tid) -> str:
        r = self.rows.get(int(tid))
        if r is None:
            return "Other"
        c = r.conference_short_name if isinstance(r.conference_short_name, str) else ""
        return c or ("FBS Indep." if int(tid) in self.fbs else "FCS")

    def is_fbs(self, tid) -> bool:
        return int(tid) in self.fbs

    def dome(self, tid) -> bool:
        r = self.rows.get(int(tid))
        return bool(r is not None and r.dome == True)

    def names_for_match(self, tid) -> list:
        """Every name a market or feed might use for this team."""
        r = self.rows.get(int(tid))
        if r is None:
            return []
        out = [r.display_name, r.short_display_name, r.location, r.school,
               r.alt_name1, r.alt_name2, r.alt_name3, r.nickname]
        return [str(x) for x in out if isinstance(x, str) and x]


# ------------------------------------------------------------- schedule ----
SCHED_COLS = ["game_id", "season", "week", "season_type", "game_date",
              "neutral_site", "conference_competition", "home_id", "away_id",
              "home_score", "away_score", "status", "venue"]


def scoreboard_week(season: int, week: int, season_type: int = 2,
                    refresh: bool = True) -> list:
    """One ESPN scoreboard call, cached as JSON. The releases only hold games
    that have been played; this is where the rest of the season comes from."""
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"scoreboard_{season}_t{season_type}_w{week}.json"
    if refresh and _stale(path, season):
        try:
            r = requests.get(SCOREBOARD, params={
                "groups": FBS_GROUP, "week": week, "seasontype": season_type,
                "dates": season, "limit": 400}, timeout=TIMEOUT)
            if r.status_code == 200:
                path.write_text(r.text)
        except requests.RequestException:
            pass
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text()).get("events", [])
    except Exception:
        return []


def _event_row(ev: dict, season: int) -> dict | None:
    comp = (ev.get("competitions") or [{}])[0]
    home = away = None
    for c in comp.get("competitors", []):
        if c.get("homeAway") == "home":
            home = c
        elif c.get("homeAway") == "away":
            away = c
    if not home or not away:
        return None
    st = ((comp.get("status") or {}).get("type") or {}).get("name", "")
    final = st == "STATUS_FINAL"

    def score(c):
        try:
            return int(c.get("score")) if final else None
        except (TypeError, ValueError):
            return None
    return {
        "game_id": int(ev["id"]), "season": season,
        "week": int((ev.get("week") or {}).get("number", 0)),
        "season_type": int((ev.get("season") or {}).get("type", 2)),
        "game_date": ev.get("date"),
        "neutral_site": bool(comp.get("neutralSite")),
        "conference_competition": bool(comp.get("conferenceCompetition")),
        "home_id": int(home["team"]["id"]), "away_id": int(away["team"]["id"]),
        "home_score": score(home), "away_score": score(away),
        "status": st, "venue": (comp.get("venue") or {}).get("fullName"),
    }


def load_schedule(refresh: bool = True) -> pd.DataFrame:
    """Every FBS game 2005-now: released seasons as published, the current
    season from ESPN's scoreboard so unplayed weeks are on the page too."""
    past = load("schedule", range(FIRST_SEASON, LAST_SEASON), refresh)
    past = past[past["season_type"].isin([2, 3])][SCHED_COLS].copy()
    # A game only counts as played once ESPN calls it final: the release can
    # carry in-progress rows with partial scores on a Saturday night.
    unfinished = past["status"] != "STATUS_FINAL"
    past.loc[unfinished, ["home_score", "away_score"]] = None

    rows = []
    for stype, weeks in ((2, range(1, 17)), (3, range(1, 6))):
        for wk in weeks:
            for ev in scoreboard_week(LAST_SEASON, wk, stype, refresh):
                r = _event_row(ev, LAST_SEASON)
                if r:
                    rows.append(r)
    cur = pd.DataFrame(rows, columns=SCHED_COLS)
    cur = cur.drop_duplicates("game_id")
    if cur.empty:   # scoreboard unreachable: fall back to the release
        cur = load("schedule", [LAST_SEASON], refresh)
        if not cur.empty:
            cur = cur[cur["season_type"].isin([2, 3])][SCHED_COLS]
    df = pd.concat([past, cur], ignore_index=True)
    df["game_date"] = pd.to_datetime(df["game_date"], utc=True)
    for c in ("home_id", "away_id", "game_id", "week", "season", "season_type"):
        df[c] = df[c].astype(int)
    df["game_type"] = df["season_type"].map({2: "REG", 3: "POST"})
    return df.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def load_team_games(refresh: bool = True) -> pd.DataFrame:
    """One row per team per game with box + EPA, from the ESPN-derived
    advanced gamelog. The opponent's offensive row IS this team's defensive
    line, joined in here so every row carries both sides."""
    a = load("adv", refresh=refresh)
    keep = ["season", "week", "season_type", "game_id", "start_date", "team_id",
            "opponent_id", "is_home", "neutral_site", "points_for",
            "points_against", "margin", "win", "EPA_per_play", "yards_per_play",
            "pass_yards", "rush_yards", "passes", "rushes", "scrimmage_plays",
            "EPA_passing_per_play", "EPA_rushing_per_play", "total_off_yards"]
    a = a[[c for c in keep if c in a.columns]].copy()
    for c in ("team_id", "opponent_id", "game_id"):
        a[c] = a[c].astype(int)
    opp = a[["game_id", "team_id", "EPA_per_play", "yards_per_play",
             "pass_yards", "rush_yards"]].rename(columns={
                 "team_id": "opponent_id", "EPA_per_play": "def_epa_allowed",
                 "yards_per_play": "def_ypp_allowed",
                 "pass_yards": "pass_yards_allowed",
                 "rush_yards": "rush_yards_allowed"})
    a = a.merge(opp, on=["game_id", "opponent_id"], how="left")
    a["start_date"] = pd.to_datetime(a["start_date"], utc=True, errors="coerce")
    return a.sort_values(["start_date", "game_id"]).reset_index(drop=True)


def load_betting(refresh: bool = True) -> pd.DataFrame:
    b = load("betting", refresh=refresh)
    if b.empty:
        return b
    b = b[b["game_spread_available"] == True].copy()
    b["game_id"] = b["game_id"].astype(int)
    # nflverse convention on the page: positive spread_line = home favored.
    b["spread_line"] = -b["home_team_spread"].astype(float)
    return b[["game_id", "spread_line", "over_under"]].drop_duplicates("game_id")


def load_talent(refresh: bool = True) -> dict:
    """(season, team_id) -> 247 talent composite (preseason roster talent)."""
    t = load("talent", refresh=refresh)
    if t.empty:
        return {}
    t["team_id"] = pd.to_numeric(t["team_id"], errors="coerce")
    t = t[t["team_id"].notna()]
    return {(int(r.season), int(r.team_id)): float(r.talent_composite)
            for r in t.itertuples(index=False)}


def load_injuries(refresh: bool = True) -> pd.DataFrame:
    p = fetch("injuries", LAST_SEASON, refresh)
    if p is None:
        return pd.DataFrame()
    d = pd.read_parquet(p)
    if d.empty:
        return d
    d["as_of_date"] = pd.to_datetime(d["as_of_date"])
    latest = d["as_of_date"].max()
    d = d[d["as_of_date"] == latest]
    d["team_id"] = d["team_id"].astype(int)
    return d
