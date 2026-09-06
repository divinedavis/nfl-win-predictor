"""Game-level features for FBS: cfb_features.parquet.

The NFL feature builder, re-derived for college from the ESPN-based team
gamelog: an Elo rating with margin-of-victory and neutral-site handling,
eight-game rolling form on both sides of the ball, rest, conference play,
preseason roster talent, and the market's own number where a spread exists.
Every rolling window is shifted so a game never sees its own line.

    python -m cfb.features        -> cfb_features.parquet
"""

import math

import numpy as np
import pandas as pd

from cfb.data import (LAST_SEASON, Teams, load_betting, load_schedule,
                      load_talent, load_team_games)

# Elo. College has bigger talent gaps and fewer games than the NFL, so the
# rating moves faster and blowouts are allowed to count (538-style margin
# multiplier). Non-FBS programs start and revert lower: an FCS team that
# never plays FBS opposition has no business sitting at 1500.
ELO_FBS = 1500.0
ELO_FCS = 1250.0
ELO_K = 30.0
ELO_HFA = 60.0          # points; neutral-site games get none
ELO_REVERT = 1 / 3      # pulled toward the tier mean each offseason
ROLL = 8
# Spread -> probability. The margin's scatter around the closing line: measured
# on the ESPN spreads 2005-2025 in build_features() and stored on every row.
DEFAULT_SIGMA = 17.0


def elo_win_prob(diff: float) -> float:
    return 1.0 / (1.0 + 10 ** (-diff / 400.0))


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def run_elo(sched: pd.DataFrame, teams: Teams) -> pd.DataFrame:
    """Pre-game Elo for both sides of every game, in kickoff order."""
    rating: dict = {}
    last_season: dict = {}
    tier = lambda tid: ELO_FBS if teams.is_fbs(tid) else ELO_FCS
    eh, ea, ep = [], [], []
    for r in sched.itertuples(index=False):
        for tid in (r.home_id, r.away_id):
            if tid not in rating:
                rating[tid] = tier(tid)
                last_season[tid] = r.season
            elif last_season[tid] != r.season:
                rating[tid] += ELO_REVERT * (tier(tid) - rating[tid])
                last_season[tid] = r.season
        hfa = 0.0 if r.neutral_site else ELO_HFA
        h, a = rating[r.home_id], rating[r.away_id]
        p = elo_win_prob(h + hfa - a)
        eh.append(h); ea.append(a); ep.append(p)
        if pd.notna(r.home_score) and pd.notna(r.away_score):
            margin = float(r.home_score - r.away_score)
            result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
            winner_diff = (h + hfa - a) if margin > 0 else (a - h - hfa)
            mult = math.log(abs(margin) + 1) * 2.2 / (winner_diff * 0.001 + 2.2)
            shift = ELO_K * mult * (result - p)
            rating[r.home_id] = h + shift
            rating[r.away_id] = a - shift
    out = sched[["game_id"]].copy()
    out["elo_home"], out["elo_away"], out["elo_prob"] = eh, ea, ep
    return out


def rolling_form(tg: pd.DataFrame) -> pd.DataFrame:
    """Per team-game, the ROLL-game trailing means of everything the page and
    the model cite, shifted one game so the row's own result is excluded."""
    tg = tg.sort_values(["team_id", "start_date", "game_id"]).copy()
    tg["pdiff"] = tg["points_for"] - tg["points_against"]
    tg["won"] = (tg["pdiff"] > 0).astype(float)
    cols = {"pdiff": "pdiff8", "won": "winrate8", "points_for": "pf8",
            "points_against": "pa8", "EPA_per_play": "off_epa8",
            "def_epa_allowed": "def_epa8", "yards_per_play": "ypp8",
            "def_ypp_allowed": "def_ypp8", "rush_yards": "rush_off8",
            "pass_yards": "pass_off8", "rush_yards_allowed": "rush_all8",
            "pass_yards_allowed": "pass_all8"}
    g = tg.groupby("team_id")
    for src, dst in cols.items():
        tg[dst] = g[src].transform(
            lambda s: s.shift(1).rolling(ROLL, min_periods=3).mean())
    tg["rest"] = g["start_date"].diff().dt.days
    tg["games_prior"] = g.cumcount()
    return tg[["game_id", "team_id", "rest", "games_prior"] + list(cols.values())]


def current_form(tg: pd.DataFrame) -> pd.DataFrame:
    """The same trailing means as of NOW, for games not yet played."""
    tg = tg.sort_values(["team_id", "start_date", "game_id"]).copy()
    tg["pdiff"] = tg["points_for"] - tg["points_against"]
    tg["won"] = (tg["pdiff"] > 0).astype(float)
    cols = {"pdiff": "pdiff8", "won": "winrate8", "points_for": "pf8",
            "points_against": "pa8", "EPA_per_play": "off_epa8",
            "def_epa_allowed": "def_epa8", "yards_per_play": "ypp8",
            "def_ypp_allowed": "def_ypp8", "rush_yards": "rush_off8",
            "pass_yards": "pass_off8", "rush_yards_allowed": "rush_all8",
            "pass_yards_allowed": "pass_all8"}
    rows = []
    for tid, grp in tg.groupby("team_id"):
        t = grp.tail(ROLL)
        row = {"team_id": tid, "last_date": grp["start_date"].max(),
               "games_prior": len(grp)}
        for src, dst in cols.items():
            row[dst] = float(t[src].mean()) if len(t) >= 3 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_features(refresh: bool = True) -> pd.DataFrame:
    teams = Teams()
    sched = load_schedule(refresh)
    tg = load_team_games(refresh)
    bets = load_betting(refresh)
    talent = load_talent(refresh)

    df = sched.copy()
    df["home_team"] = df["home_id"].map(teams.abbr)
    df["away_team"] = df["away_id"].map(teams.abbr)
    df["home_fbs"] = df["home_id"].map(teams.is_fbs).astype(int)
    df["away_fbs"] = df["away_id"].map(teams.is_fbs).astype(int)
    df["gameday"] = df["game_date"].dt.tz_convert("America/New_York").dt.date.astype(str)
    df["home_win"] = np.where(df["home_score"].isna(), np.nan,
                              (df["home_score"] > df["away_score"]).astype(float))
    df["neutral"] = df["neutral_site"].astype(int)
    df["conf_game"] = df["conference_competition"].astype(int)
    df["is_dome"] = df["home_id"].map(teams.dome).astype(int)

    df = df.merge(run_elo(df, teams), on="game_id", how="left")
    df["elo_diff"] = df["elo_home"] + np.where(df["neutral"] == 1, 0, ELO_HFA) - df["elo_away"]

    # Rolling form: played games take the as-of-then window from the gamelog;
    # unplayed games take each team's current window.
    form = rolling_form(tg)
    now = current_form(tg)
    for side in ("home", "away"):
        f = form.rename(columns={c: f"{side}_{c}" for c in form.columns
                                 if c not in ("game_id", "team_id")})
        f = f.rename(columns={"team_id": f"{side}_id"})
        df = df.merge(f, on=["game_id", f"{side}_id"], how="left")
        n = now.rename(columns={c: f"{side}_{c}_now" for c in now.columns
                                if c != "team_id"}).rename(columns={"team_id": f"{side}_id"})
        df = df.merge(n, on=f"{side}_id", how="left")
        unplayed = df["home_win"].isna()
        for c in ("pdiff8", "winrate8", "pf8", "pa8", "off_epa8", "def_epa8",
                  "ypp8", "def_ypp8", "rush_off8", "pass_off8", "rush_all8",
                  "pass_all8", "games_prior"):
            df.loc[unplayed, f"{side}_{c}"] = df.loc[unplayed, f"{side}_{c}_now"]
        # Rest for an unplayed game: days since that team's last game.
        last = df.loc[unplayed, f"{side}_last_date_now"]
        df.loc[unplayed, f"{side}_rest"] = (df.loc[unplayed, "game_date"] - last).dt.days
        df = df.drop(columns=[c for c in df.columns if c.endswith("_now")])
        # A season opener has no "rest"; cap at the offseason so it reads as
        # fully rested rather than as a 250-day outlier the trees latch onto.
        df[f"{side}_rest"] = df[f"{side}_rest"].clip(upper=21)

    for a, b, d in (("pdiff8", "pdiff8", "pdiff8_diff"),
                    ("off_epa8", "off_epa8", "off_epa8_diff"),
                    ("def_epa8", "def_epa8", "def_epa8_diff")):
        df[d] = df[f"home_{a}"] - df[f"away_{b}"]
    df["rest_diff"] = df["home_rest"] - df["away_rest"]

    df["home_talent"] = [talent.get((s, t), np.nan) for s, t in zip(df.season, df.home_id)]
    df["away_talent"] = [talent.get((s, t), np.nan) for s, t in zip(df.season, df.away_id)]
    df["talent_diff"] = df["home_talent"] - df["away_talent"]

    # Market. The spread's own scatter gives the probability conversion.
    df = df.merge(bets, on="game_id", how="left")
    done = df[df["home_win"].notna() & df["spread_line"].notna()]
    resid = (done["home_score"] - done["away_score"]) - done["spread_line"]
    sigma = float(resid.std()) if len(done) > 500 else DEFAULT_SIGMA
    df["spread_sigma"] = sigma
    df["mkt_prob"] = [norm_cdf(s / sigma) if pd.notna(s) else np.nan
                      for s in df["spread_line"]]
    return df


FEATURES = [
    "elo_diff", "elo_home", "elo_away", "elo_prob",
    "home_rest", "away_rest", "rest_diff",
    "conf_game", "neutral", "week", "home_fbs", "away_fbs",
    "home_pdiff8", "away_pdiff8", "pdiff8_diff",
    "home_winrate8", "away_winrate8",
    "home_pf8", "home_pa8", "away_pf8", "away_pa8",
    "home_off_epa8", "away_off_epa8", "off_epa8_diff",
    "home_def_epa8", "away_def_epa8", "def_epa8_diff",
    "home_ypp8", "away_ypp8", "home_def_ypp8", "away_def_ypp8",
    "home_talent", "away_talent", "talent_diff",
    "mkt_prob",
]

if __name__ == "__main__":
    df = build_features()
    df.to_parquet("cfb_features.parquet", index=False)
    done = df["home_win"].notna().sum()
    print(f"Wrote cfb_features.parquet: {len(df)} games ({done} completed) "
          f"{df.season.min()}-{df.season.max()}, spread sigma "
          f"{df.spread_sigma.iloc[0]:.1f}")
