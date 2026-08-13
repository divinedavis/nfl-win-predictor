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
import nflreadpy as nfl

FIRST_SEASON = 1999
LAST_SEASON = 2026

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


def elo_win_prob(elo_diff: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


def build_features() -> pd.DataFrame:
    games = load_games()
    epa = load_team_epa()
    epa_map = {
        (r.season, r.week, r.team): r.off_epa for r in epa.itertuples(index=False)
    }

    elo: dict[str, float] = {}
    season_of: dict[str, int] = {}
    # Per-team chronological history of completed games (newest last).
    hist: dict[str, list[dict]] = {}

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

    return pd.DataFrame(rows)


FEATURES = [
    "elo_diff", "elo_home", "elo_away", "elo_prob",
    "home_rest", "away_rest", "rest_diff",
    "div_game", "is_dome", "week",
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
