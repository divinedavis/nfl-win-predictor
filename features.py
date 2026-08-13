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
DEFAULT_SNAP_SHARE = 0.15  # assumed share for a listed-out player with no snap history

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
    """Per team-game offensive EPA, keyed by (season, week, team)."""
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(
                nfl.load_team_stats([season], summary_level="week").to_pandas()
            )
        except Exception:
            # Season not published yet (e.g. upcoming season before kickoff).
            continue
    ts = pd.concat(frames, ignore_index=True)
    ts["team"] = ts["team"].map(canon)
    ts["off_epa"] = ts["passing_epa"].fillna(0) + ts["rushing_epa"].fillna(0)
    return ts[["season", "week", "team", "off_epa"]]


def norm_name(name: str) -> str:
    """Normalize a player name for joining injuries to snap counts."""
    n = "".join(ch for ch in str(name).lower() if ch.isalpha() or ch == " ")
    parts = [p for p in n.split() if p not in ("jr", "sr", "ii", "iii", "iv", "v")]
    return " ".join(parts)


def load_injury_reports() -> dict:
    """(season, week, team) -> counts of Out/Doubtful and Questionable players,
    the gsis_ids of those ruled Out/Doubtful (to spot a sidelined QB), and
    (normalized name, position group) pairs for snap-share weighting."""
    frames = []
    for season in range(INJURIES_FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_injuries([season]).to_pandas())
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
        if r.report_status in ("Out", "Doubtful"):
            d["n_out"] += 1
            if pd.notna(r.gsis_id):
                d["out_ids"].add(r.gsis_id)
            group = POS_GROUP.get(str(r.position).upper())
            if group:
                d["out_players"].append((norm_name(r.full_name), group))
        elif r.report_status == "Questionable":
            d["n_quest"] += 1
    return reports


def load_snap_index() -> dict:
    """(season, week, team) -> list of ((name, group), snap_share) for that game.
    Share is the player's fraction of his side's snaps (offense or defense)."""
    frames = []
    for season in range(SNAPS_FIRST_SEASON - 1, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_snap_counts([season]).to_pandas())
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


def build_features() -> pd.DataFrame:
    games = load_games()
    epa = load_team_epa()
    epa_map = {
        (r.season, r.week, r.team): r.off_epa for r in epa.itertuples(index=False)
    }
    injuries = load_injury_reports()
    snaps = load_snap_index()

    elo: dict[str, float] = {}
    season_of: dict[str, int] = {}
    # Per-team chronological history of completed games (newest last).
    hist: dict[str, list[dict]] = {}
    last_qb: dict[str, str] = {}  # team -> gsis_id of most recent starter
    # (name, group) -> recent snap shares (newest last), updated post-game so
    # lookups during feature building only ever see pregame information.
    player_shares: dict[tuple, list] = {}

    def weighted_outs(team: str, season: int, week: int) -> dict:
        """Snap-share-weighted sum of Out/Doubtful players per position group:
        a full-time starter out ≈ +1.0, a rotational player ≈ his share."""
        if season < SNAPS_FIRST_SEASON:
            return {grp: np.nan for grp in POS_GROUPS}
        rep = injuries.get((season, week, team))
        wtd = {grp: 0.0 for grp in POS_GROUPS}
        if rep:
            for name, grp in rep["out_players"]:
                shares = player_shares.get((name, grp))
                wtd[grp] += (float(np.mean(shares[-4:])) if shares
                             else DEFAULT_SNAP_SHARE)
        return wtd

    rows = []
    for g in games.itertuples(index=False):
        home, away = g.home_team, g.away_team

        for team in (home, away):
            elo.setdefault(team, ELO_START)
            hist.setdefault(team, [])
            # Offseason reversion the first time a team appears in a new season.
            if season_of.get(team) is not None and season_of[team] != g.season:
                elo[team] = elo[team] + ELO_REVERT * (ELO_MEAN - elo[team])
            season_of[team] = g.season

        def roll(team: str) -> dict:
            past = hist[team][-ROLL_N:]
            if not past:
                return {"pdiff": np.nan, "winrate": np.nan, "pf": np.nan,
                        "pa": np.nan, "off_epa": np.nan, "def_epa": np.nan}
            return {
                "pdiff": np.mean([p["pf"] - p["pa"] for p in past]),
                "winrate": np.mean([p["won"] for p in past]),
                "pf": np.mean([p["pf"] for p in past]),
                "pa": np.mean([p["pa"] for p in past]),
                "off_epa": np.nanmean([p["off_epa"] for p in past]),
                "def_epa": np.nanmean([p["def_epa"] for p in past]),
            }

        rh, ra = roll(home), roll(away)
        elo_diff = elo[home] + ELO_HFA - elo[away]

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
            return float(prev in out_ids)

        h_qb_changed = qb_changed(home, g.home_qb_id, h_out_ids)
        a_qb_changed = qb_changed(away, g.away_qb_id, a_out_ids)

        h_wtd = weighted_outs(home, g.season, g.week)
        a_wtd = weighted_outs(away, g.season, g.week)

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
            "home_pdiff8": rh["pdiff"], "away_pdiff8": ra["pdiff"],
            "pdiff8_diff": rh["pdiff"] - ra["pdiff"],
            "home_winrate8": rh["winrate"], "away_winrate8": ra["winrate"],
            "home_pf8": rh["pf"], "home_pa8": rh["pa"],
            "away_pf8": ra["pf"], "away_pa8": ra["pa"],
            "home_off_epa8": rh["off_epa"], "away_off_epa8": ra["off_epa"],
            "home_def_epa8": rh["def_epa"], "away_def_epa8": ra["def_epa"],
            "off_epa8_diff": rh["off_epa"] - ra["off_epa"],
            "def_epa8_diff": rh["def_epa"] - ra["def_epa"],
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
            hist[home].append({"pf": g.home_score, "pa": g.away_score,
                               "won": float(margin > 0), "off_epa": h_epa,
                               "def_epa": a_epa})
            hist[away].append({"pf": g.away_score, "pa": g.home_score,
                               "won": float(margin < 0), "off_epa": a_epa,
                               "def_epa": h_epa})
            if pd.notna(g.home_qb_id):
                last_qb[home] = g.home_qb_id
            if pd.notna(g.away_qb_id):
                last_qb[away] = g.away_qb_id
            for team in (home, away):
                for key, share in snaps.get((g.season, g.week, team), []):
                    player_shares.setdefault(key, []).append(share)

    return pd.DataFrame(rows)


FEATURES = [
    "elo_diff", "elo_home", "elo_away", "elo_prob",
    "home_rest", "away_rest", "rest_diff",
    "div_game", "is_dome", "week",
    "temp", "wind",
    "home_n_out", "away_n_out", "home_n_quest", "away_n_quest",
    "home_qb_changed", "away_qb_changed",
    *[f"home_{grp}_out_wt" for grp in POS_GROUPS],
    *[f"away_{grp}_out_wt" for grp in POS_GROUPS],
    "home_pdiff8", "away_pdiff8", "pdiff8_diff",
    "home_winrate8", "away_winrate8",
    "home_pf8", "home_pa8", "away_pf8", "away_pa8",
    "home_off_epa8", "away_off_epa8", "off_epa8_diff",
    "home_def_epa8", "away_def_epa8", "def_epa8_diff",
]


if __name__ == "__main__":
    df = build_features()
    df.to_parquet("features.parquet", index=False)
    done = df["home_win"].notna().sum()
    print(f"Wrote features.parquet: {len(df)} games ({done} completed) "
          f"{df.season.min()}-{df.season.max()}")
