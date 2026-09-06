"""Player-prop projections for FBS: cfb_props_projections.csv.

The NFL quantile-props model rebuilt on ESPN player box scores (2005+):
P10/P25/P50/P75/P90 of a player's next-game passing, rushing and receiving
yards and receptions, from leak-free rolling usage and production, his share
of the team's volume, what the opponent allows to his position, and the game
context from cfb_features.parquet. Same output columns as props.py so the
page renders both leagues with one code path.

College has no target data, so receiving usage is receptions; no Next Gen
Stats, so the tracking block is absent; no official injury report, so status
comes from ESPN's injury feed where it exists.

    python -m cfb.props
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from cfb.data import (LAST_SEASON, Teams, load, load_injuries,
                      load_schedule)

FIRST_SEASON = 2013         # enough history for the quantile models
WINDOW_DAYS = 8          # project every team's next game inside this window
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
# stat -> (positions, usage column, minimum rolling usage to project)
STATS = {
    "passing_yards": (["QB"], "attempts", 12.0),
    "rushing_yards": (["RB", "QB"], "carries", 6.0),
    "receiving_yards": (["WR", "TE", "RB"], "receptions", 2.0),
    "receptions": (["WR", "TE", "RB"], "receptions", 2.0),
}
POOL = {"passing_yards": "attempts", "rushing_yards": "carries",
        "receiving_yards": "receptions", "receptions": "receptions"}
PARAMS = dict(n_estimators=300, learning_rate=0.05, max_depth=4,
              min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
              reg_lambda=2.0, objective="reg:quantileerror",
              tree_method="hist", random_state=42)
FEATS = ["y4", "y10", "ystd10", "use4", "use10", "games_prior", "is_home",
         "team_elo_prob", "team_off_epa8", "opp_def_epa8", "opp_allowed8",
         "week", "team_vol8", "opp_vol_faced8", "share_t8", "share_trend"]


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_player_weeks(refresh: bool = True) -> pd.DataFrame:
    """One row per player per game with the four stat lines, joined to the
    schedule for week/opponent/home and to rosters for position."""
    pb = load("player_box", range(FIRST_SEASON, LAST_SEASON + 1), refresh,
              columns=["season", "game_id", "team_id", "athlete_id", "athlete_name",
                       "category", "completions/passingAttempts", "passingYards",
                       "rushingAttempts", "rushingYards", "receptions", "receivingYards"])
    pb = pb[pb["category"].isin(["passing", "rushing", "receiving"])].copy()
    pb["athlete_id"] = _num(pb["athlete_id"]).astype("Int64")
    pb["team_id"] = _num(pb["team_id"]).astype("Int64")
    pb["game_id"] = _num(pb["game_id"]).astype("Int64")
    pb = pb.dropna(subset=["athlete_id", "team_id", "game_id"])
    ca = pb["completions/passingAttempts"].astype(str).str.split("/", expand=True)
    pb["attempts"] = _num(ca[1]) if ca.shape[1] > 1 else np.nan
    pb["passing_yards"] = _num(pb["passingYards"])
    pb["carries"] = _num(pb["rushingAttempts"])
    pb["rushing_yards"] = _num(pb["rushingYards"])
    pb["receptions"] = _num(pb["receptions"])
    pb["receiving_yards"] = _num(pb["receivingYards"])
    cols = ["attempts", "passing_yards", "carries", "rushing_yards",
            "receptions", "receiving_yards"]
    # A player appears once per category he recorded; collapse to one row.
    ps = (pb.groupby(["season", "game_id", "team_id", "athlete_id"])
          .agg({**{c: "max" for c in cols}, "athlete_name": "first"})
          .reset_index())
    for c in ("game_id", "team_id", "athlete_id"):
        ps[c] = ps[c].astype(int)

    sched = load_schedule(refresh)
    sched = sched[sched["home_win"].notna()] if "home_win" in sched else sched
    keep = ["game_id", "season", "week", "game_type", "game_date", "home_id", "away_id"]
    home = sched[keep].copy()
    home = home.rename(columns={"home_id": "team_id", "away_id": "opponent_id"})
    home["is_home"] = 1
    away = sched[keep].copy()
    away = away.rename(columns={"away_id": "team_id", "home_id": "opponent_id"})
    away["is_home"] = 0
    ctx = pd.concat([home, away]).drop(columns=["season"])
    ps = ps.merge(ctx, on=["game_id", "team_id"], how="inner")

    ro = load("rosters", range(FIRST_SEASON, LAST_SEASON + 1), refresh,
              columns=["season", "athlete_id", "position_abbreviation"])
    ro["athlete_id"] = _num(ro["athlete_id"]).astype("Int64")
    ro = ro.dropna(subset=["athlete_id"])
    pos = (ro.sort_values("season").groupby("athlete_id")["position_abbreviation"]
           .last().to_dict())
    ps["position"] = ps["athlete_id"].map(pos)
    # Box scores without a roster match: infer from the stat line.
    guess = np.where(ps["attempts"].fillna(0) >= 5, "QB",
             np.where(ps["carries"].fillna(0) > ps["receptions"].fillna(0), "RB", "WR"))
    ps["position"] = ps["position"].fillna(pd.Series(guess, index=ps.index))
    ps = ps[ps["position"].isin(["QB", "RB", "WR", "TE"])]
    return ps.sort_values(["game_date", "game_id"]).reset_index(drop=True)


def game_context() -> pd.DataFrame:
    """(game_id, team_id) -> Elo win prob, own offense form, opponent's
    defensive form, from the feature table."""
    df = pd.read_parquet("cfb_features.parquet")
    rows = []
    for r in df.itertuples(index=False):
        rows.append({"game_id": r.game_id, "team_id": r.home_id,
                     "team_elo_prob": r.elo_prob, "team_off_epa8": r.home_off_epa8,
                     "opp_def_epa8": r.away_def_epa8, "played": pd.notna(r.home_win),
                     "season": r.season, "week": r.week, "game_type": r.game_type})
        rows.append({"game_id": r.game_id, "team_id": r.away_id,
                     "team_elo_prob": 1 - r.elo_prob, "team_off_epa8": r.away_off_epa8,
                     "opp_def_epa8": r.home_def_epa8, "played": pd.notna(r.home_win),
                     "season": r.season, "week": r.week, "game_type": r.game_type})
    return pd.DataFrame(rows)


def build_table(ps: pd.DataFrame, ctx: pd.DataFrame, stat: str) -> pd.DataFrame:
    positions, use_col, _ = STATS[stat]
    d = ps[ps["position"].isin(positions)].copy()
    d = d.sort_values(["athlete_id", "game_date"]).reset_index(drop=True)
    d["y"] = d[stat].fillna(0)
    d["use"] = d[use_col].fillna(0)
    g = d.groupby("athlete_id")
    d["y4"] = g["y"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["y10"] = g["y"].transform(lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    d["ystd10"] = g["y"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    d["use4"] = g["use"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["use10"] = g["use"].transform(lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    d["games_prior"] = g.cumcount()
    d = d.merge(ctx[["game_id", "team_id", "team_elo_prob", "team_off_epa8",
                     "opp_def_epa8", "season", "week"]].drop_duplicates(["game_id", "team_id"]),
                on=["game_id", "team_id"], how="inner", suffixes=("", "_ctx"))
    if "week_ctx" in d.columns:
        d["week"] = d["week_ctx"]
        d = d.drop(columns=["week_ctx"])
    if "season_ctx" in d.columns:
        d = d.drop(columns=["season_ctx"])

    # Opponent allowed to these positions, trailing 8 games, prior only.
    per_game = (d.groupby(["game_id", "game_date", "opponent_id"])["y"].sum()
                .reset_index().sort_values("game_date"))
    per_game["opp_allowed8"] = per_game.groupby("opponent_id")["y"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(per_game[["game_id", "opponent_id", "opp_allowed8"]],
                on=["game_id", "opponent_id"], how="left")

    # Volume block: team pool, what the opponent has been facing, the
    # player's rolled share and its trend.
    pool_col = POOL[stat]
    tp = (ps.groupby(["game_id", "game_date", "team_id"])[pool_col].sum()
          .reset_index().rename(columns={pool_col: "pool"}).sort_values("game_date"))
    tp["team_vol8"] = tp.groupby("team_id")["pool"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(tp[["game_id", "team_id", "pool", "team_vol8"]], on=["game_id", "team_id"], how="left")
    fp = (ps.groupby(["game_id", "game_date", "opponent_id"])[pool_col].sum()
          .reset_index().rename(columns={pool_col: "faced"}).sort_values("game_date"))
    fp["opp_vol_faced8"] = fp.groupby("opponent_id")["faced"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(fp[["game_id", "opponent_id", "opp_vol_faced8"]], on=["game_id", "opponent_id"], how="left")
    d["share_raw"] = d["use"] / d["pool"].replace(0, np.nan)
    g = d.groupby("athlete_id")
    d["share_t8"] = g["share_raw"].transform(lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    share_t4 = g["share_raw"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["share_trend"] = share_t4 - d["share_t8"]
    return d[d["y4"].notna() & (d["games_prior"] >= 2)].copy()


def fit_quantiles(train: pd.DataFrame) -> dict:
    return {q: XGBRegressor(**PARAMS, quantile_alpha=q).fit(train[FEATS], train["y"])
            for q in QUANTILES}


def predict_quantiles(models: dict, rows: pd.DataFrame) -> np.ndarray:
    preds = np.column_stack([models[q].predict(rows[FEATS]) for q in QUANTILES])
    return np.maximum(np.sort(preds, axis=1), 0.0)


def project(refresh: bool = True) -> None:
    teams = Teams()
    ps = load_player_weeks(refresh)
    ctx = game_context()
    inj = load_injuries(refresh)
    out_by_team: dict = {}
    if len(inj):
        for r in inj.itertuples(index=False):
            if str(r.status).lower() in ("out", "injured reserve", "doubtful"):
                out_by_team.setdefault(int(r.team_id), set()).add(
                    str(r.athlete_display_name).lower())

    # Each team's NEXT game inside the projection window. College weeks run
    # Thursday to Monday and the page shows two weeks out, so this is per
    # team rather than one league-wide week number.
    sched = load_schedule(refresh)
    now = pd.Timestamp.now(tz="UTC")
    up = sched[(sched.season == LAST_SEASON) & sched.home_score.isna()
               & (sched.game_date >= now - pd.Timedelta(hours=6))
               & (sched.game_date <= now + pd.Timedelta(days=WINDOW_DAYS))]
    if up.empty:
        raise SystemExit("no unplayed games inside the projection window")
    opp_of, home_of, week_of, gid_of = {}, {}, {}, {}
    for r in up.sort_values("game_date").itertuples(index=False):
        for tid, opp, hm in ((r.home_id, r.away_id, 1), (r.away_id, r.home_id, 0)):
            if tid in opp_of:
                continue
            opp_of[tid], home_of[tid], week_of[tid], gid_of[tid] = opp, hm, int(r.week), int(r.game_id)
    ctx_by = {(r.game_id, r.team_id): r for r in ctx.itertuples(index=False)}
    week = int(up["week"].min())
    print(f"Projecting {LAST_SEASON} from week {week}: {len(opp_of)} team slates in the next {WINDOW_DAYS} days")

    # Current roster: a player projects for the team whose 2026 roster lists
    # him, so transfers move with their new school even before they play.
    ro = load("rosters", [LAST_SEASON], refresh, columns=["athlete_id", "team_id"])
    ro["athlete_id"] = _num(ro["athlete_id"]).astype("Int64")
    cur_team = {int(a): int(t) for a, t in zip(ro["athlete_id"].dropna(), ro.loc[ro["athlete_id"].notna(), "team_id"])}

    out_rows = []
    for stat, (positions, use_col, min_use) in STATS.items():
        table = build_table(ps, ctx, stat)
        train = table[table["season"] >= 2012]
        models = fit_quantiles(train)
        pool_col = POOL[stat]
        tp = (ps.groupby(["game_id", "game_date", "team_id"])[pool_col].sum()
              .reset_index().rename(columns={pool_col: "pool"}).sort_values("game_date"))
        team_vol_now = tp.groupby("team_id")["pool"].apply(lambda s: s.tail(8).mean()).to_dict()
        pool_map = {(r.game_id, r.team_id): r.pool for r in tp.itertuples(index=False)}
        fp = (ps.groupby(["game_id", "game_date", "opponent_id"])[pool_col].sum()
              .reset_index().sort_values("game_date"))
        opp_faced_now = fp.groupby("opponent_id")[pool_col].apply(lambda s: s.tail(8).mean()).to_dict()
        d = ps[ps["position"].isin(positions)].copy().sort_values(["athlete_id", "game_date"])
        d["y"] = d[stat].fillna(0)
        d["use"] = d[use_col].fillna(0)
        allowed = (d.groupby(["game_id", "game_date", "opponent_id"])["y"].sum()
                   .reset_index().sort_values("game_date"))
        allowed_now = allowed.groupby("opponent_id")["y"].apply(lambda s: s.tail(8).mean()).to_dict()

        last = d.groupby("athlete_id").tail(1)
        recent = last[last["season"] >= LAST_SEASON - 1]
        # The starter at QB: whoever threw the most in the team's last game.
        qb1 = {}
        if stat == "passing_yards":
            lastgame = d.sort_values("game_date").groupby("team_id").tail(60)
            for tid, grp in lastgame.groupby("team_id"):
                g = grp[grp.game_id == grp.game_id.iloc[-1]]
                if len(g):
                    qb1[tid] = int(g.loc[g["use"].idxmax(), "athlete_id"])
        for r in recent.itertuples(index=False):
            team_now = cur_team.get(int(r.athlete_id), int(r.team_id))
            if team_now not in opp_of:
                continue  # bye, or not on the slate
            if stat == "passing_yards" and qb1.get(team_now) not in (None, int(r.athlete_id)):
                continue
            h = d[d.athlete_id == r.athlete_id]
            if len(h) < 2:
                continue
            tail4, tail10 = h.tail(4), h.tail(10)
            if tail4["use"].mean() < min_use:
                continue
            opp = opp_of[team_now]
            c = ctx_by.get((gid_of[team_now], team_now))
            if c is None:
                continue
            week = week_of[team_now]
            shares = [u / pool_map[k] for u, k in zip(h["use"], zip(h["game_id"], h["team_id"]))
                      if pool_map.get(k)]
            row = pd.DataFrame([{
                "y4": tail4["y"].mean(), "y10": tail10["y"].mean(),
                "ystd10": tail10["y"].std(), "use4": tail4["use"].mean(),
                "use10": tail10["use"].mean(), "games_prior": len(h),
                "is_home": home_of[team_now], "team_elo_prob": c.team_elo_prob,
                "team_off_epa8": c.team_off_epa8, "opp_def_epa8": c.opp_def_epa8,
                "opp_allowed8": allowed_now.get(opp, np.nan), "week": week,
                "team_vol8": team_vol_now.get(team_now, np.nan),
                "opp_vol_faced8": opp_faced_now.get(opp, np.nan),
                "share_t8": float(np.mean(shares[-8:])) if shares else np.nan,
                "share_trend": (float(np.mean(shares[-4:]) - np.mean(shares[-8:]))
                                if len(shares) >= 4 else np.nan),
            }])
            q = predict_quantiles(models, row)[0]
            vs = h[h["opponent_id"] == opp]
            vs_log = ";".join(f"'{int(v.season) % 100} wk{int(v.week)}: {v.y:g}"
                              for v in vs.tail(5).itertuples(index=False))
            lg = h.iloc[-1]
            status = "OUT" if str(r.athlete_name).lower() in out_by_team.get(team_now, set()) else ""
            out_rows.append({
                "season": LAST_SEASON, "week": week, "stat": stat,
                "player_id": int(r.athlete_id), "player": r.athlete_name,
                "position": r.position, "team": teams.abbr(team_now),
                "opp": teams.abbr(opp), "is_home": home_of[team_now],
                "p10": round(q[0], 1), "p25": round(q[1], 1), "p50": round(q[2], 1),
                "p75": round(q[3], 1), "p90": round(q[4], 1),
                "vs_opp_n": len(vs),
                "vs_opp_avg": round(float(vs["y"].mean()), 1) if len(vs) else None,
                "vs_opp_med": round(float(vs["y"].median()), 1) if len(vs) else None,
                "vs_opp_log": vs_log,
                "career_avg": round(float(h["y"].mean()), 1),
                "last_yds": round(float(lg.y), 0),
                "last_season": int(lg.season), "last_week": int(lg.week),
                "last_opp": teams.abbr(lg.opponent_id),
                "last_post": int(lg.game_type == "POST"),
                "status": status,
            })
    out = pd.DataFrame(out_rows).sort_values(["stat", "p50"], ascending=[True, False])
    out.to_csv("cfb_props_projections.csv", index=False)
    print(f"Wrote cfb_props_projections.csv ({len(out)} projections, "
          f"{out.player_id.nunique()} players)")


if __name__ == "__main__":
    project()
