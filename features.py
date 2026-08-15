"""Build the game-level feature table for NFL win prediction.

Every feature for a game uses ONLY information available before kickoff:
- Elo ratings updated through the previous game (538-style, with margin-of-victory
  multiplier and 1/3 reversion to the mean between seasons)
- Rolling form over each team's last 8 games (point differential, win rate,
  points for/against, offensive EPA, defensive EPA allowed)
- Schedule context (rest days, divisional game, dome, week)

Future games (e.g. the upcoming 2026 season) get features from history and NaN
targets, so the same table serves training, backtesting, and weekly predictions.
"""

import math

import numpy as np
import pandas as pd
import requests
import nflreadpy as nfl

FIRST_SEASON = 1999
LAST_SEASON = 2026
INJURIES_FIRST_SEASON = 2009  # nflverse injury reports start here
SNAPS_FIRST_SEASON = 2013     # snap counts start 2012; first full season of history

# Position -> position group for injury impact features.
POS_GROUP = {
    "QB": "qb",
    "RB": "rb", "FB": "rb", "HB": "rb",
    "WR": "wr",
    "TE": "te",
    "T": "ol", "G": "ol", "C": "ol", "OL": "ol", "OT": "ol", "OG": "ol",
    "DE": "dl", "DT": "dl", "NT": "dl", "DL": "dl", "EDGE": "dl",
    "LB": "lb", "OLB": "lb", "ILB": "lb", "MLB": "lb",
    "CB": "db", "S": "db", "FS": "db", "SS": "db", "DB": "db", "SAF": "db",
}
POS_GROUPS = ["qb", "rb", "wr", "te", "ol", "dl", "lb", "db"]
OFF_GROUPS = ["qb", "rb", "wr", "te"]  # per-player value = EPA per game
DEF_GROUPS = ["dl", "lb", "db"]        # per-player value = box-score production
VALUE_GROUPS = OFF_GROUPS + DEF_GROUPS  # everything except OL (no public stats)

# No public per-player defensive EPA exists, so defenders are valued by
# weighted box-score playmaking per game (roughly EPA-scaled weights).
DEF_VALUE_WEIGHTS = {
    "def_sacks": 2.0, "def_interceptions": 4.0, "def_pass_defended": 1.0,
    "def_tackles_for_loss": 1.0, "def_fumbles_forced": 2.0,
    "def_qb_hits": 0.5, "def_tds": 4.0, "def_tackles_solo": 0.1,
}
OFF_POSITIONS = ["QB", "RB", "FB", "WR", "TE"]
DEF_POSITIONS = ["DE", "DT", "NT", "DL", "LB", "OLB", "ILB", "MLB",
                 "CB", "S", "FS", "SS", "DB"]
DEFAULT_SNAP_SHARE = 0.15  # assumed share for a listed-out player with no snap history
RATING_WINDOW = 10   # games in a player's rolling value
RATING_SHRINK = 3    # pseudo-games of league-average (0) mixed in
# Box scores underrate block-eaters (a double-teamed NT frees teammates but
# shows no sacks), so a defender's value is floored by his playing time:
# a 100%-snap defender never rates below this, whatever his stat line.
DEF_SNAP_VALUE_FLOOR = 1.5
# Measured league-wide 2022-24: 37% of Questionable players took zero snaps,
# so a Questionable player counts at this fraction of his value.
QUESTIONABLE_WEIGHT = 0.37
# A player on IR counts as out only while his absence is news — after this
# many weeks his loss is already baked into the team's recent form and Elo.
IR_RECENT_WEEKS = 5

# Offensive line: no public per-player stats exist, so linemen carry no rating
# and the snap-weighted {side}_ol_out_wt treats five half-starters the same as
# two full ones. A line's cohesion is not linear in bodies — losing a second
# starter forces a backup AND shuffles the survivors' assignments — so count
# STARTERS specifically (recent snap share at or above OL_STARTER_SHARE) and
# flag the multi-out threshold separately.
OL_STARTER_SHARE = 0.6
OL_MULTI_OUT = 2.0  # starters out at which a line is considered patched-together

# Stadium coordinates for predict-time weather forecasts (outdoor games only).
STADIUMS = {
    "ARI": (33.5276, -112.2626), "ATL": (33.7550, -84.4010),
    "BAL": (39.2780, -76.6227), "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528), "CHI": (41.8623, -87.6167),
    "CIN": (39.0955, -84.5161), "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945), "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456), "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107), "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373), "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3392), "LAC": (33.9535, -118.3392),
    "LV": (36.0909, -115.1833), "MIA": (25.9580, -80.2389),
    "MIN": (44.9737, -93.2577), "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812), "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745), "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158), "SEA": (47.5952, -122.3316),
    "SF": (37.4030, -121.9700), "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713), "WAS": (38.9076, -76.8645),
}

# Hours behind Eastern for each team's home market (upset-indicator features:
# body-clock disadvantage for west-coast teams in early Eastern kickoffs).
TEAM_TZ = {
    "BUF": 0, "MIA": 0, "NE": 0, "NYJ": 0, "NYG": 0, "PHI": 0, "PIT": 0,
    "WAS": 0, "BAL": 0, "CIN": 0, "CLE": 0, "ATL": 0, "CAR": 0, "JAX": 0,
    "TB": 0, "IND": 0, "DET": 0,
    "CHI": 1, "GB": 1, "MIN": 1, "DAL": 1, "HOU": 1, "TEN": 1, "KC": 1, "NO": 1,
    "DEN": 2, "ARI": 2,
    "LA": 3, "LAC": 3, "LV": 3, "SEA": 3, "SF": 3,
}

PYTH_EXP = 2.37          # NFL Pythagorean exponent

# "Finishing" features: EPA/play in the last 5 minutes of one-score games,
# rolled over 16 team-games (small per-game samples need the long window).
# Improved walk-forward Brier in 11/11 backtest seasons (2026-08-13); the
# same-window all-plays control added nothing, so the signal is situational
# (late-game QB + pass rush), not generic team quality.
CLUTCH_ROLL = 16
CLUTCH_MIN_PLAYS = 40
FTN_FIRST_SEASON = 2022  # FTN charting (blitz / pass-rusher counts) starts here
# A season is "dead" for a team once even winning out can't reach this many
# wins (playoffs realistically out of reach) — a nothing-to-play-for proxy.
PLAYOFF_WINS = 9

ELO_START = 1505.0
ELO_MEAN = 1505.0
ELO_K = 20.0
ELO_HFA = 52.0          # home-field advantage in Elo points
ELO_REVERT = 1 / 3      # fraction reverted to the mean each offseason
ROLL_N = 8              # games in the rolling-form window

# Franchise moves: key Elo/history by current abbreviation so ratings carry over.
FRANCHISE = {"SD": "LAC", "STL": "LA", "OAK": "LV", "LAR": "LA"}


def canon(team: str) -> str:
    return FRANCHISE.get(team, team)


def load_games() -> pd.DataFrame:
    sched = nfl.load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1))).to_pandas()
    sched["home_team"] = sched["home_team"].map(canon)
    sched["away_team"] = sched["away_team"].map(canon)
    sched["gameday"] = pd.to_datetime(sched["gameday"])
    sched = sched.sort_values(["gameday", "game_id"]).reset_index(drop=True)
    return sched


def load_team_epa() -> pd.DataFrame:
    """Per team-game offensive EPA, turnover margin, and sack counts,
    keyed by (season, week, team)."""
    cols = ["season", "week", "team", "passing_epa", "rushing_epa",
            "passing_yards", "rushing_yards", "attempts", "carries",
            "passing_interceptions", "fumbles_lost_total",
            "def_interceptions", "fumble_recovery_opp",
            "sacks_suffered", "def_sacks"]
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            df = nfl.load_team_stats([season], summary_level="week").to_pandas()
            frames.append(df.reindex(columns=cols))
        except Exception:
            # Season not published yet (e.g. upcoming season before kickoff).
            continue
    ts = pd.concat(frames, ignore_index=True)
    ts["team"] = ts["team"].map(canon)
    ts["off_epa"] = ts["passing_epa"].fillna(0) + ts["rushing_epa"].fillna(0)
    ts["to_margin"] = (
        ts["def_interceptions"].fillna(0) + ts["fumble_recovery_opp"].fillna(0)
        - ts["passing_interceptions"].fillna(0) - ts["fumbles_lost_total"].fillna(0)
    )
    return ts[["season", "week", "team", "off_epa", "to_margin",
               "sacks_suffered", "def_sacks", "passing_yards",
               "rushing_yards", "attempts", "carries"]]


def load_ftn_defense() -> dict:
    """(season, week, defense_team) -> (blitz_rate, avg_pass_rushers) from FTN
    charting (2022+). Charted plays carry no team, so join to play-by-play."""
    index: dict = {}
    for season in range(FTN_FIRST_SEASON, LAST_SEASON + 1):
        try:
            # Column-trim in polars BEFORE the pandas copy: full pbp is ~400
            # columns and the droplet only has 2GB of RAM.
            ftn = nfl.load_ftn_charting([season]).select(
                ["nflverse_game_id", "nflverse_play_id", "season", "week",
                 "n_blitzers", "n_pass_rushers"]).to_pandas()
            pbp = nfl.load_pbp([season]).select(
                ["game_id", "play_id", "defteam"]).to_pandas()
        except Exception:
            continue  # season not charted/published yet
        m = ftn.merge(pbp, left_on=["nflverse_game_id", "nflverse_play_id"],
                      right_on=["game_id", "play_id"], how="inner")
        m = m[m["defteam"].notna()]
        m["defteam"] = m["defteam"].map(canon)
        m["blitz"] = (m["n_blitzers"].fillna(0) > 0).astype(float)
        agg = m.groupby(["season", "week", "defteam"]).agg(
            blitz_rate=("blitz", "mean"),
            rushers=("n_pass_rushers", "mean"),
        )
        for (s, w, team), r in agg.iterrows():
            index[(s, w, team)] = (float(r.blitz_rate), float(r.rushers))
    return index


def haversine_miles(a: tuple, b: tuple) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = (math.sin((lat2 - lat1) / 2) ** 2
         + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2)
    return 3958.8 * 2 * math.asin(math.sqrt(h))


def norm_name(name: str) -> str:
    """Normalize a player name for joining injuries to snap counts."""
    n = "".join(ch for ch in str(name).lower() if ch.isalpha() or ch == " ")
    parts = [p for p in n.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def load_injury_reports() -> dict:
    """(season, week, team) -> counts of Out/Doubtful and Questionable players,
    the gsis_ids of those ruled Out/Doubtful (to spot a sidelined QB), and
    (normalized name, position group) pairs for snap-share weighting."""
    cols = ["season", "week", "team", "gsis_id", "position", "full_name",
            "report_status"]
    frames = []
    for season in range(INJURIES_FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_injuries([season]).to_pandas()[cols])
        except Exception:
            continue  # season not published yet
    inj = pd.concat(frames, ignore_index=True)
    inj["team"] = inj["team"].map(canon)
    reports: dict = {}
    for r in inj.itertuples(index=False):
        d = reports.setdefault(
            (r.season, r.week, r.team),
            {"n_out": 0, "n_quest": 0, "out_ids": set(), "out_players": []},
        )
        group = POS_GROUP.get(str(r.position).upper())
        player = {
            "norm": norm_name(r.full_name),
            "name": str(r.full_name),
            "gsis": r.gsis_id if pd.notna(r.gsis_id) else None,
            "pos": str(r.position).upper(),
            "group": group,
        }
        if r.report_status in ("Out", "Doubtful"):
            d["n_out"] += 1
            if pd.notna(r.gsis_id):
                d["out_ids"].add(r.gsis_id)
            if group:
                d["out_players"].append(player)
        elif r.report_status == "Questionable":
            d["n_quest"] += 1
            if group:
                d.setdefault("quest_players", []).append(player)
    return reports


def load_ir_index() -> tuple:
    """Players on injured reserve, from weekly roster status (public days
    before kickoff, unlike game-status reports which never list IR).
    Returns ((season, week, team) -> [player dicts], and per (season, team)
    the latest known week's list for upcoming games with no roster row yet)."""
    cols = ["season", "week", "team", "gsis_id", "full_name", "position",
            "status"]
    frames = []
    for season in range(INJURIES_FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_rosters_weekly([season]).to_pandas()[cols])
        except Exception:
            continue
    ros = pd.concat(frames, ignore_index=True)
    ros["team"] = ros["team"].map(canon)
    ir = ros[ros["status"] == "RES"]
    index: dict = {}
    for r in ir.itertuples(index=False):
        group = POS_GROUP.get(str(r.position).upper())
        if not group or pd.isna(r.gsis_id):
            continue
        index.setdefault((r.season, r.week, r.team), []).append({
            "norm": norm_name(r.full_name), "name": str(r.full_name),
            "gsis": r.gsis_id, "pos": str(r.position).upper(), "group": group,
        })
    latest: dict = {}
    for (season, week, team), players in index.items():
        if week >= latest.get((season, team), (0, None))[0]:
            latest[(season, team)] = (week, players)
    return index, latest


def load_player_value_index() -> dict:
    """(season, week, team) -> list of (gsis_id, name, group, value_that_game).
    Offensive value is EPA per game; defensive value is weighted box-score
    playmaking (DEF_VALUE_WEIGHTS) — the raw material for player ratings."""
    cols = ["season", "week", "team", "player_id", "player_display_name",
            "position", "passing_epa", "rushing_epa", "receiving_epa",
            *DEF_VALUE_WEIGHTS]
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            df = nfl.load_player_stats([season], summary_level="week").to_pandas()
            frames.append(
                df[df["position"].isin(OFF_POSITIONS + DEF_POSITIONS)][cols]
            )
        except Exception:
            continue
    ps = pd.concat(frames, ignore_index=True)
    ps["team"] = ps["team"].map(canon)
    is_def = ps["position"].isin(DEF_POSITIONS)
    off_value = (ps["passing_epa"].fillna(0) + ps["rushing_epa"].fillna(0)
                 + ps["receiving_epa"].fillna(0))
    def_value = sum(w * ps[c].fillna(0) for c, w in DEF_VALUE_WEIGHTS.items())
    ps["value_game"] = off_value.where(~is_def, def_value)
    index: dict = {}
    for r in ps.itertuples(index=False):
        index.setdefault((r.season, r.week, r.team), []).append(
            (r.player_id, r.player_display_name,
             POS_GROUP[str(r.position).upper()], float(r.value_game))
        )
    return index


def load_snap_index() -> dict:
    """(season, week, team) -> list of ((name, group), snap_share) for that game.
    Share is the player's fraction of his side's snaps (offense or defense)."""
    cols = ["season", "week", "team", "player", "position",
            "offense_pct", "defense_pct"]
    frames = []
    for season in range(SNAPS_FIRST_SEASON - 1, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_snap_counts([season]).to_pandas()[cols])
        except Exception:
            continue
    sc = pd.concat(frames, ignore_index=True)
    sc["team"] = sc["team"].map(canon)
    index: dict = {}
    for r in sc.itertuples(index=False):
        group = POS_GROUP.get(str(r.position).upper())
        if not group:
            continue
        share = max(r.offense_pct or 0, r.defense_pct or 0)
        index.setdefault((r.season, r.week, r.team), []).append(
            ((norm_name(r.player), group), share)
        )
    return index


_forecast_cache: dict = {}


def forecast_weather(home_team: str, gameday, gametime) -> tuple:
    """Open-Meteo forecast (free, no key) for a future outdoor game within the
    16-day forecast horizon. Returns (temp_F, wind_mph) or (nan, nan)."""
    coords = STADIUMS.get(home_team)
    date = pd.Timestamp(gameday)
    days_out = (date - pd.Timestamp.now().normalize()).days
    if coords is None or not (0 <= days_out <= 15):
        return (np.nan, np.nan)
    key = (home_team, date.date())
    if key in _forecast_cache:
        return _forecast_cache[key]
    hour = 16
    if isinstance(gametime, str) and ":" in gametime:
        hour = min(23, int(gametime.split(":")[0]))
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": coords[0], "longitude": coords[1],
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit", "wind_speed_unit": "mph",
                "timezone": "America/New_York",
                "start_date": str(date.date()), "end_date": str(date.date()),
            },
            timeout=10,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        result = (hourly["temperature_2m"][hour], hourly["wind_speed_10m"][hour])
    except Exception:
        result = (np.nan, np.nan)
    _forecast_cache[key] = result
    return result


def elo_win_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def load_depth_qb1() -> dict:
    """team -> gsis_id of the current official depth-chart QB1 (nflverse
    mirrors the league feed daily). Authoritative for upcoming games — the
    last-game starter is wrong after finale rest weeks and injury fill-ins."""
    try:
        dc = nfl.load_depth_charts([LAST_SEASON]).to_pandas()
    except Exception:
        return {}
    qb = dc[(dc["pos_abb"] == "QB") & (dc["pos_rank"] == 1)
            & dc["gsis_id"].notna()].copy()
    if qb.empty:
        return {}
    qb["team"] = qb["team"].map(canon)
    latest = qb.sort_values("dt").groupby("team").tail(1)
    return dict(zip(latest["team"], latest["gsis_id"]))


def load_clutch_rolling() -> pd.DataFrame:
    """Rolling last-5-min one-score EPA/play per (season, week, team), from
    the build_clutch.py cache. Empty frame (all-NaN features) if absent."""
    try:
        c = pd.read_csv("clutch_team_weeks.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["season", "week", "team",
                                     "clutch_off16", "clutch_def16"])
    c = c.sort_values(["season", "week"])
    g = c.groupby("team")
    for side in ("off", "def"):
        epa_r = g[f"{side}_clutch_epa"].transform(
            lambda s: s.shift(1).rolling(CLUTCH_ROLL, min_periods=8).sum())
        plays_r = g[f"{side}_clutch_plays"].transform(
            lambda s: s.shift(1).rolling(CLUTCH_ROLL, min_periods=8).sum())
        rate = epa_r / plays_r.replace(0, np.nan)
        c[f"clutch_{side}16"] = rate.where(plays_r >= CLUTCH_MIN_PLAYS)
    return c[["season", "week", "team", "clutch_off16", "clutch_def16"]]


def build_features() -> pd.DataFrame:
    games = load_games()
    epa = load_team_epa()
    epa_map = {
        (r.season, r.week, r.team): r.off_epa for r in epa.itertuples(index=False)
    }
    stat_map = {
        (r.season, r.week, r.team): (r.to_margin, r.sacks_suffered, r.def_sacks)
        for r in epa.itertuples(index=False)
    }
    yds_map = {
        (r.season, r.week, r.team): (r.passing_yards, r.rushing_yards)
        for r in epa.itertuples(index=False)
    }
    ftn_map = load_ftn_defense()
    injuries = load_injury_reports()
    snaps = load_snap_index()
    player_values = load_player_value_index()
    ir_index, ir_latest = load_ir_index()

    elo: dict[str, float] = {}
    season_of: dict[str, int] = {}
    # Per-team chronological history of completed games (newest last).
    hist: dict[str, list[dict]] = {}
    # Regular-season record to date, for the nothing-to-play-for flag.
    season_wins: dict[str, float] = {}
    season_played: dict[str, int] = {}

    # Next regular-season opponent per (game_id, team) — the schedule is public
    # all season, so a lookahead feature built from it is leak-free. Playoff
    # opponents are excluded: they weren't known during the regular season.
    next_opp: dict[tuple, str] = {}
    reg = games[games["game_type"] == "REG"]
    team_sched: dict[tuple, list] = {}
    for g in reg.itertuples(index=False):
        team_sched.setdefault((g.season, g.home_team), []).append(
            (g.game_id, g.away_team))
        team_sched.setdefault((g.season, g.away_team), []).append(
            (g.game_id, g.home_team))
    for (_, team), sched in team_sched.items():
        for (gid, _), (_, upcoming) in zip(sched, sched[1:]):
            next_opp[(gid, team)] = upcoming
    last_qb: dict[str, str] = {}  # team -> gsis_id of most recent starter
    starts_hist: dict[str, list] = {}  # team -> starter ids, last 17 games
    depth_qb1 = load_depth_qb1()

    def expected_qb(team: str):
        """Expected starter for a FUTURE game: official depth chart first,
        else the most frequent starter of the last 17 games (ties -> most
        recent) — never just the last game's starter, which is polluted by
        finale rest weeks and injury fill-ins."""
        pid = depth_qb1.get(team)
        if pid is not None:
            return pid
        hist_q = starts_hist.get(team, [])
        if hist_q:
            from collections import Counter
            counts = Counter(hist_q)
            top = max(counts.values())
            cands = {p for p, c in counts.items() if c == top}
            for p in reversed(hist_q):
                if p in cands:
                    return p
        return last_qb.get(team)
    # (name, group) -> recent snap shares (newest last), updated post-game so
    # lookups during feature building only ever see pregame information.
    player_shares: dict[tuple, list] = {}
    # gsis_id -> recent per-game EPA (newest last); the player's rating is a
    # shrunk rolling mean, so rookies/backups sit near 0 and stars well above.
    val_hist: dict[str, list] = {}
    player_names: dict[str, str] = {}
    player_group: dict[str, str] = {}
    player_last_seen: dict[str, tuple] = {}  # pid -> (season, week)

    # (name, group) -> (season, week) last played, from snap counts. Linemen
    # produce no stat lines, so player_last_seen never holds them and their IR
    # recency has to come from playing time instead.
    snap_last_seen: dict[tuple, tuple] = {}

    def snap_seen_recently(norm: str, group: str, season: int, week: int) -> bool:
        last = snap_last_seen.get((norm, group))
        if last is None:
            return False
        if last[0] == season:
            return week - last[1] <= IR_RECENT_WEEKS
        return last[0] == season - 1 and week <= IR_RECENT_WEEKS

    def seen_recently(pid, season: int, week: int) -> bool:
        """Was this player on the field recently enough that his IR absence
        is still news rather than old form the ratings already absorbed?"""
        last = player_last_seen.get(pid)
        if last is None:
            return False
        if last[0] == season:
            return week - last[1] <= IR_RECENT_WEEKS
        return last[0] == season - 1 and week <= IR_RECENT_WEEKS

    def rating(pid) -> float:
        games_ = val_hist.get(pid, [])
        recent = games_[-RATING_WINDOW:]
        return float(sum(recent) / (len(recent) + RATING_SHRINK))

    def weighted_outs(team: str, season: int, week: int) -> dict:
        """Snap-share-weighted sum of Out/Doubtful players per position group:
        a full-time starter out ≈ +1.0, a rotational player ≈ his share."""
        if season < SNAPS_FIRST_SEASON:
            return {grp: np.nan for grp in POS_GROUPS}
        rep = injuries.get((season, week, team))
        wtd = {grp: 0.0 for grp in POS_GROUPS}
        if rep:
            for p in rep["out_players"]:
                shares = player_shares.get((p["norm"], p["group"]))
                wtd[p["group"]] += (float(np.mean(shares[-4:])) if shares
                                    else DEFAULT_SNAP_SHARE)
        return wtd

    def is_ol_starter(p: dict) -> bool:
        """Was this lineman a starter in his recent games? Linemen have no
        stat line, so playing time is the only public evidence of a starting
        job — and a starter is exactly the man whose replacement hurts."""
        shares = player_shares.get((p["norm"], "ol"))
        return bool(shares) and float(np.mean(shares[-4:])) >= OL_STARTER_SHARE

    def ol_starters_out(team: str, season: int, week: int) -> tuple:
        """(starters out, multi-out flag) for a team's offensive line.
        Out/Doubtful count in full, Questionable at the measured sit rate, and
        recently-lost IR linemen in full — the same accounting as every other
        group, but by bodies rather than value, since linemen have no ratings.
        NaN before snap counts exist, when starters can't be identified."""
        if season < SNAPS_FIRST_SEASON:
            return np.nan, np.nan
        rep = injuries.get((season, week, team))
        n = 0.0
        counted = set()
        if rep:
            for p in rep["out_players"]:
                if p["group"] == "ol" and is_ol_starter(p):
                    n += 1.0
                    counted.add(p["norm"])
            for p in rep.get("quest_players", []):
                if (p["group"] == "ol" and p["norm"] not in counted
                        and is_ol_starter(p)):
                    n += QUESTIONABLE_WEIGHT
                    counted.add(p["norm"])
        ir_players = ir_index.get((season, week, team))
        if ir_players is None:
            latest = ir_latest.get((season, team))
            ir_players = latest[1] if latest else []
        for p in ir_players:
            if (p["group"] == "ol" and p["norm"] not in counted
                    and snap_seen_recently(p["norm"], "ol", season, week)
                    and is_ol_starter(p)):
                n += 1.0
                counted.add(p["norm"])
        return n, float(n >= OL_MULTI_OUT)

    def player_value(p: dict) -> float:
        """A player's effective value: rating, floored by snap share for
        defenders (box scores underrate block-eaters)."""
        val = rating(p["gsis"])
        if p["group"] in DEF_GROUPS:
            shares = player_shares.get((p["norm"], p["group"]))
            share = float(np.mean(shares[-4:])) if shares else 0.0
            val = max(val, DEF_SNAP_VALUE_FLOOR * share)
        return val

    def out_value_and_names(team: str, season: int, week: int) -> tuple:
        """Rating-weighted absences per position group (star out >> backup
        out) — offense in EPA/game, defense in box-score playmaking/game.
        Out/Doubtful count in full, Questionable at the measured 37% sit
        rate, and recently-lost IR players (never on game-status reports) in
        full. Also returns a display string naming the biggest absences."""
        rep = injuries.get((season, week, team))
        if season < INJURIES_FIRST_SEASON:
            return {grp: np.nan for grp in VALUE_GROUPS}, "", 0.0
        out_val = {grp: 0.0 for grp in VALUE_GROUPS}
        names = []
        counted = set()
        if rep:
            for p in rep["out_players"]:
                if p["group"] in VALUE_GROUPS and p["gsis"]:
                    val = player_value(p)
                    out_val[p["group"]] += val
                    counted.add(p["gsis"])
                    names.append((val, f'{p["pos"]} {p["name"]} ({val:+.1f})'))
                elif p["group"] == "ol":
                    shares = player_shares.get((p["norm"], p["group"]))
                    if shares and np.mean(shares[-4:]) >= 0.6:
                        names.append((0.5, f'{p["pos"]} {p["name"]} (starter)'))
            for p in rep.get("quest_players", []):
                if p["group"] in VALUE_GROUPS and p["gsis"]:
                    val = QUESTIONABLE_WEIGHT * player_value(p)
                    out_val[p["group"]] += val
                    counted.add(p["gsis"])
                    if val >= 0.5:
                        names.append((val, f'{p["pos"]} {p["name"]} (Q, {val:+.1f})'))
        ir_players = ir_index.get((season, week, team))
        if ir_players is None:
            latest = ir_latest.get((season, team))
            ir_players = latest[1] if latest else []
        ir_wt = 0.0
        for p in ir_players:
            if (p["group"] in VALUE_GROUPS and p["gsis"] not in counted
                    and seen_recently(p["gsis"], season, week)):
                val = player_value(p)
                out_val[p["group"]] += val
                names.append((val, f'{p["pos"]} {p["name"]} (IR, {val:+.1f})'))
                shares = player_shares.get((p["norm"], p["group"]))
                ir_wt += float(np.mean(shares[-4:])) if shares else 0.0
        names.sort(key=lambda n: n[0], reverse=True)
        return out_val, "; ".join(n for _, n in names[:3]), ir_wt

    rows = []
    for g in games.itertuples(index=False):
        home, away = g.home_team, g.away_team

        for team in (home, away):
            elo.setdefault(team, ELO_START)
            hist.setdefault(team, [])
            # Offseason reversion the first time a team appears in a new season.
            if season_of.get(team) is not None and season_of[team] != g.season:
                elo[team] = elo[team] + ELO_REVERT * (ELO_MEAN - elo[team])
            if season_of.get(team) != g.season:
                season_wins[team] = 0.0
                season_played[team] = 0
            season_of[team] = g.season

        def roll(team: str) -> dict:
            past = hist[team][-ROLL_N:]
            keys = ["pdiff", "winrate", "pf", "pa", "off_epa", "def_epa",
                    "pyth", "luck", "to_margin", "sacked", "def_sacks",
                    "blitz_rate", "rushers", "pass_allowed", "rush_allowed",
                    "pass_off", "rush_off"]
            if not past:
                return dict.fromkeys(keys, np.nan)
            out = {
                "pdiff": np.mean([p["pf"] - p["pa"] for p in past]),
                "winrate": np.mean([p["won"] for p in past]),
                "pf": np.mean([p["pf"] for p in past]),
                "pa": np.mean([p["pa"] for p in past]),
            }
            for k in keys[4:]:
                if k in ("pyth", "luck"):
                    continue
                with np.errstate(all="ignore"):
                    vals = [p.get(k, np.nan) for p in past]
                    out[k] = np.nan if all(pd.isna(v) for v in vals) else np.nanmean(vals)
            # Pythagorean expectation from rolling points; "luck" is how far
            # the record outruns it (winning close games, turnover bounces) —
            # the classic regression-candidate profile of a fake favorite.
            pf_e, pa_e = out["pf"] ** PYTH_EXP, out["pa"] ** PYTH_EXP
            out["pyth"] = pf_e / (pf_e + pa_e) if pf_e + pa_e > 0 else 0.5
            out["luck"] = out["winrate"] - out["pyth"]
            return out

        rh, ra = roll(home), roll(away)
        elo_diff = elo[home] + ELO_HFA - elo[away]

        # --- situational spots (all from public pregame info) ---
        def letdown(team: str) -> float:
            """Coming off an emotional divisional win — the classic flat spot."""
            past = hist[team]
            return float(past[-1].get("div_win", 0.0)) if past else np.nan

        def lookahead(team: str, opponent: str) -> float:
            """How much stronger the NEXT opponent is than today's — a big gap
            invites looking past a weak opponent. Zero at season end."""
            if g.game_type != "REG":
                return 0.0
            nxt = next_opp.get((g.game_id, team))
            if nxt is None:
                return 0.0
            return elo.get(nxt, ELO_START) - elo.get(opponent, ELO_START)

        def low_stakes(team: str) -> float:
            """Nothing left to play for: mathematically dead-ish (can't reach
            PLAYOFF_WINS even winning out) or locked up late (resting risk)."""
            if g.game_type != "REG":
                return 0.0
            total = 17 if g.season >= 2021 else 16
            wins = season_wins.get(team, 0.0)
            remaining = total - season_played.get(team, 0)
            return float(wins + remaining < PLAYOFF_WINS
                         or (wins >= 12 and g.week >= 17))

        # --- travel / body clock for the road team ---
        h_coord, a_coord = STADIUMS.get(home), STADIUMS.get(away)
        away_travel = (haversine_miles(a_coord, h_coord)
                       if h_coord and a_coord else np.nan)
        h_tz, a_tz = TEAM_TZ.get(home), TEAM_TZ.get(away)
        tz_shift = (a_tz - h_tz) if h_tz is not None and a_tz is not None else np.nan
        hour = 16
        if isinstance(g.gametime, str) and ":" in g.gametime:
            hour = int(g.gametime.split(":")[0])
        west_early = float(a_tz is not None and h_tz is not None
                           and a_tz >= 2 and h_tz <= 1 and hour <= 13)

        # Injury report counts (NaN before 2009; 0 = reports exist, none listed).
        def inj_counts(team: str) -> tuple:
            rep = injuries.get((g.season, g.week, team))
            if rep is not None:
                return rep["n_out"], rep["n_quest"], rep["out_ids"]
            if g.season >= INJURIES_FIRST_SEASON:
                return 0, 0, set()
            return np.nan, np.nan, set()

        h_out, h_quest, h_out_ids = inj_counts(home)
        a_out, a_quest, a_out_ids = inj_counts(away)

        # QB change vs the team's previous game. Starters are announced pregame,
        # so using the schedule's starter id for played games is leak-free; for
        # future games, flag if the incumbent is Out/Doubtful on the report.
        def qb_changed(team: str, qb_id, out_ids: set) -> float:
            prev = last_qb.get(team)
            if prev is None:
                return np.nan
            if pd.notna(qb_id):
                return float(qb_id != prev)
            exp = expected_qb(team)
            return float(exp in out_ids) if exp else float(prev in out_ids)

        h_qb_changed = qb_changed(home, g.home_qb_id, h_out_ids)
        a_qb_changed = qb_changed(away, g.away_qb_id, a_out_ids)

        h_wtd = weighted_outs(home, g.season, g.week)
        a_wtd = weighted_outs(away, g.season, g.week)
        h_out_val, h_key_outs, h_ir_wt = out_value_and_names(home, g.season, g.week)
        a_out_val, a_key_outs, a_ir_wt = out_value_and_names(away, g.season, g.week)
        h_ol_n, h_ol_multi = ol_starters_out(home, g.season, g.week)
        a_ol_n, a_ol_multi = ol_starters_out(away, g.season, g.week)

        # Individual QB rating for the expected starter: the schedule's actual
        # starter when known (announced pregame), otherwise the incumbent —
        # unless he's Out/Doubtful or on IR, in which case a backup rates 0.
        def ir_ids(team: str) -> set:
            players = ir_index.get((g.season, g.week, team))
            if players is None:
                latest = ir_latest.get((g.season, team))
                players = latest[1] if latest else []
            return {p["gsis"] for p in players}

        def qb_value(team: str, qb_id, out_ids: set) -> tuple:
            pid = qb_id if pd.notna(qb_id) else expected_qb(team)
            if pid is None or (pd.isna(qb_id) and pid in (out_ids | ir_ids(team))):
                return 0.0, ""
            return rating(pid), player_names.get(pid, "")

        h_qb_val, h_qb_name = qb_value(home, g.home_qb_id, h_out_ids)
        a_qb_val, a_qb_name = qb_value(away, g.away_qb_id, a_out_ids)

        # Weather: recorded temp/wind for past games; forecast for upcoming
        # outdoor games inside Open-Meteo's 16-day horizon.
        temp, wind = g.temp, g.wind
        if (pd.isna(g.home_score) and pd.isna(temp)
                and str(g.roof) not in ("dome", "closed")):
            temp, wind = forecast_weather(home, g.gameday, g.gametime)

        rows.append({
            "game_id": g.game_id,
            "season": g.season,
            "week": g.week,
            "game_type": g.game_type,
            "gameday": g.gameday,
            "home_team": home,
            "away_team": away,
            "home_score": g.home_score,
            "away_score": g.away_score,
            # --- features ---
            "elo_home": elo[home],
            "elo_away": elo[away],
            "elo_diff": elo_diff,
            "elo_prob": elo_win_prob(elo_diff),
            "home_rest": g.home_rest,
            "away_rest": g.away_rest,
            "rest_diff": (g.home_rest - g.away_rest)
            if pd.notna(g.home_rest) and pd.notna(g.away_rest) else np.nan,
            "div_game": g.div_game,
            "is_dome": 1 if str(g.roof) in ("dome", "closed") else 0,
            "temp": temp,
            "wind": wind,
            "home_n_out": h_out, "away_n_out": a_out,
            "home_n_quest": h_quest, "away_n_quest": a_quest,
            "home_qb_changed": h_qb_changed, "away_qb_changed": a_qb_changed,
            **{f"home_{grp}_out_wt": h_wtd[grp] for grp in POS_GROUPS},
            **{f"away_{grp}_out_wt": a_wtd[grp] for grp in POS_GROUPS},
            **{f"home_{grp}_out_epa": h_out_val[grp] for grp in OFF_GROUPS},
            **{f"away_{grp}_out_epa": a_out_val[grp] for grp in OFF_GROUPS},
            **{f"home_{grp}_out_val": h_out_val[grp] for grp in DEF_GROUPS},
            **{f"away_{grp}_out_val": a_out_val[grp] for grp in DEF_GROUPS},
            # --- offensive line (candidate features, ablation-tested) ---
            "home_ol_starters_out": h_ol_n, "away_ol_starters_out": a_ol_n,
            "home_ol_multi_out": h_ol_multi, "away_ol_multi_out": a_ol_multi,
            "ol_starters_out_diff": h_ol_n - a_ol_n,
            "ol_multi_out_diff": h_ol_multi - a_ol_multi,
            "home_qb_val": h_qb_val, "away_qb_val": a_qb_val,
            "qb_val_diff": h_qb_val - a_qb_val,
            "home_qb_pred_name": h_qb_name, "away_qb_pred_name": a_qb_name,
            "home_key_outs": h_key_outs, "away_key_outs": a_key_outs,
            # Display-only (not model features): snap weight of recent-IR
            # players, so the app's starters-out count includes them.
            "home_ir_wt": h_ir_wt, "away_ir_wt": a_ir_wt,
            "home_pdiff8": rh["pdiff"], "away_pdiff8": ra["pdiff"],
            "pdiff8_diff": rh["pdiff"] - ra["pdiff"],
            "home_winrate8": rh["winrate"], "away_winrate8": ra["winrate"],
            "home_pf8": rh["pf"], "home_pa8": rh["pa"],
            "away_pf8": ra["pf"], "away_pa8": ra["pa"],
            "home_off_epa8": rh["off_epa"], "away_off_epa8": ra["off_epa"],
            "home_def_epa8": rh["def_epa"], "away_def_epa8": ra["def_epa"],
            # display-only matchup context (per game, last 8)
            "home_pass_all8": rh["pass_allowed"], "away_pass_all8": ra["pass_allowed"],
            "home_rush_all8": rh["rush_allowed"], "away_rush_all8": ra["rush_allowed"],
            "home_pass_off8": rh["pass_off"], "away_pass_off8": ra["pass_off"],
            "home_rush_off8": rh["rush_off"], "away_rush_off8": ra["rush_off"],
            "off_epa8_diff": rh["off_epa"] - ra["off_epa"],
            "def_epa8_diff": rh["def_epa"] - ra["def_epa"],
            # --- upset indicators (candidate features, ablation-tested) ---
            "home_pyth8": rh["pyth"], "away_pyth8": ra["pyth"],
            "home_luck8": rh["luck"], "away_luck8": ra["luck"],
            "luck8_diff": rh["luck"] - ra["luck"],
            "home_to_margin8": rh["to_margin"], "away_to_margin8": ra["to_margin"],
            "to_margin8_diff": rh["to_margin"] - ra["to_margin"],
            "home_sacked8": rh["sacked"], "away_sacked8": ra["sacked"],
            "home_def_sacks8": rh["def_sacks"], "away_def_sacks8": ra["def_sacks"],
            # positive = home QB in more danger than away QB
            "sack_mismatch": (rh["sacked"] + ra["def_sacks"])
            - (ra["sacked"] + rh["def_sacks"]),
            "home_blitz_rate8": rh["blitz_rate"], "away_blitz_rate8": ra["blitz_rate"],
            "home_rushers8": rh["rushers"], "away_rushers8": ra["rushers"],
            "away_travel": away_travel,
            "tz_shift": tz_shift,
            "west_early": west_early,
            "home_letdown": letdown(home), "away_letdown": letdown(away),
            "home_lookahead": lookahead(home, away),
            "away_lookahead": lookahead(away, home),
            "home_low_stakes": low_stakes(home), "away_low_stakes": low_stakes(away),
            # --- benchmark (NOT a model feature): Vegas closing spread ---
            "spread_line": g.spread_line,
            # --- target ---
            "home_win": (
                np.nan if pd.isna(g.home_score) or g.home_score == g.away_score
                else float(g.home_score > g.away_score)
            ),
        })

        # Update Elo + history only for completed games.
        if pd.notna(g.home_score):
            margin = g.home_score - g.away_score
            p_home = elo_win_prob(elo_diff)
            actual = 0.5 if margin == 0 else float(margin > 0)
            winner_diff = abs(elo_diff) if (margin > 0) == (elo_diff > 0) else -abs(elo_diff)
            mov = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            shift = ELO_K * mov * (actual - p_home)
            elo[home] += shift
            elo[away] -= shift

            h_epa = epa_map.get((g.season, g.week, home), np.nan)
            a_epa = epa_map.get((g.season, g.week, away), np.nan)

            def game_stats(team: str) -> dict:
                to, sacked, dsacks = stat_map.get(
                    (g.season, g.week, team), (np.nan, np.nan, np.nan))
                blitz, rushers = ftn_map.get(
                    (g.season, g.week, team), (np.nan, np.nan))
                return {"to_margin": to, "sacked": sacked, "def_sacks": dsacks,
                        "blitz_rate": blitz, "rushers": rushers}

            h_yds = yds_map.get((g.season, g.week, home), (np.nan, np.nan))
            a_yds = yds_map.get((g.season, g.week, away), (np.nan, np.nan))
            hist[home].append({"pf": g.home_score, "pa": g.away_score,
                               "won": float(margin > 0), "off_epa": h_epa,
                               "def_epa": a_epa,
                               "pass_allowed": a_yds[0], "rush_allowed": a_yds[1],
                               "pass_off": h_yds[0], "rush_off": h_yds[1],
                               "div_win": float(margin > 0 and g.div_game == 1),
                               **game_stats(home)})
            hist[away].append({"pf": g.away_score, "pa": g.home_score,
                               "won": float(margin < 0), "off_epa": a_epa,
                               "def_epa": h_epa,
                               "pass_allowed": h_yds[0], "rush_allowed": h_yds[1],
                               "pass_off": a_yds[0], "rush_off": a_yds[1],
                               "div_win": float(margin < 0 and g.div_game == 1),
                               **game_stats(away)})
            if g.game_type == "REG":
                season_wins[home] = season_wins.get(home, 0.0) + float(margin > 0)
                season_wins[away] = season_wins.get(away, 0.0) + float(margin < 0)
                season_played[home] = season_played.get(home, 0) + 1
                season_played[away] = season_played.get(away, 0) + 1
            if pd.notna(g.home_qb_id):
                last_qb[home] = g.home_qb_id
                starts_hist.setdefault(home, []).append(g.home_qb_id)
                starts_hist[home] = starts_hist[home][-17:]
            if pd.notna(g.away_qb_id):
                last_qb[away] = g.away_qb_id
                starts_hist.setdefault(away, []).append(g.away_qb_id)
                starts_hist[away] = starts_hist[away][-17:]
            for team in (home, away):
                for key, share in snaps.get((g.season, g.week, team), []):
                    player_shares.setdefault(key, []).append(share)
                    snap_last_seen[key] = (g.season, g.week)
                for pid, name, grp, value_game in player_values.get(
                        (g.season, g.week, team), []):
                    val_hist.setdefault(pid, []).append(value_game)
                    player_names[pid] = name
                    player_group[pid] = grp
                    player_last_seen[pid] = (g.season, g.week)

    # Snapshot of current player ratings (EPA per game, shrunk) for analysis
    # and the web app's player pages.
    ratings = pd.DataFrame([
        {"gsis_id": pid, "name": player_names.get(pid, ""),
         "group": player_group.get(pid, ""), "games": len(h),
         "rating": round(rating(pid), 2)}
        for pid, h in val_hist.items()
        if player_last_seen.get(pid, (0, 0))[0] >= LAST_SEASON - 1
    ]).sort_values("rating", ascending=False)
    ratings.to_csv("player_ratings.csv", index=False)

    df = pd.DataFrame(rows)
    clutch = load_clutch_rolling()
    for side in ("home", "away"):
        df = df.merge(
            clutch.rename(columns={
                "team": f"{side}_team",
                "clutch_off16": f"{side}_clutch_off16",
                "clutch_def16": f"{side}_clutch_def16"}),
            on=["season", "week", f"{side}_team"], how="left")
    df["clutch_net_diff"] = (
        (df["home_clutch_off16"].fillna(0) - df["home_clutch_def16"].fillna(0))
        - (df["away_clutch_off16"].fillna(0) - df["away_clutch_def16"].fillna(0)))
    return df


FEATURES = [
    "elo_diff", "elo_home", "elo_away", "elo_prob",
    "home_rest", "away_rest", "rest_diff",
    "div_game", "is_dome", "week",
    "temp", "wind",
    "home_n_out", "away_n_out", "home_n_quest", "away_n_quest",
    "home_qb_changed", "away_qb_changed",
    *[f"home_{grp}_out_wt" for grp in POS_GROUPS],
    *[f"away_{grp}_out_wt" for grp in POS_GROUPS],
    *[f"home_{grp}_out_epa" for grp in OFF_GROUPS],
    *[f"away_{grp}_out_epa" for grp in OFF_GROUPS],
    *[f"home_{grp}_out_val" for grp in DEF_GROUPS],
    *[f"away_{grp}_out_val" for grp in DEF_GROUPS],
    "home_qb_val", "away_qb_val", "qb_val_diff",
    "home_pdiff8", "away_pdiff8", "pdiff8_diff",
    "home_winrate8", "away_winrate8",
    "home_pf8", "home_pa8", "away_pf8", "away_pa8",
    "home_off_epa8", "away_off_epa8", "off_epa8_diff",
    "home_def_epa8", "away_def_epa8", "def_epa8_diff",
    "home_clutch_off16", "home_clutch_def16",
    "away_clutch_off16", "away_clutch_def16", "clutch_net_diff",
]

# Candidate upset-indicator groups, ablation-tested by backtest_groups.py.
# Promote a group into FEATURES only if it improves the walk-forward Brier.
CANDIDATE_GROUPS = {
    "luck": ["home_pyth8", "away_pyth8", "home_luck8", "away_luck8",
             "luck8_diff", "home_to_margin8", "away_to_margin8",
             "to_margin8_diff"],
    "sacks": ["home_sacked8", "away_sacked8", "home_def_sacks8",
              "away_def_sacks8", "sack_mismatch"],
    "ftn": ["home_blitz_rate8", "away_blitz_rate8",
            "home_rushers8", "away_rushers8"],
    "travel": ["away_travel", "tz_shift", "west_early"],
    "situational": ["home_letdown", "away_letdown",
                    "home_lookahead", "away_lookahead",
                    "home_low_stakes", "away_low_stakes"],
    "oline": ["home_ol_starters_out", "away_ol_starters_out",
              "home_ol_multi_out", "away_ol_multi_out",
              "ol_starters_out_diff", "ol_multi_out_diff"],
}


if __name__ == "__main__":
    df = build_features()
    df.to_parquet("features.parquet", index=False)
    done = df["home_win"].notna().sum()
    print(f"Wrote features.parquet: {len(df)} games ({done} completed) "
          f"{df.season.min()}-{df.season.max()}")
