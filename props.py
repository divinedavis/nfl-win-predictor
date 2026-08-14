"""Player-prop projections: distributions for the four liquid prop stats.

Predicts quantiles (P10/P25/P50/P75/P90) of a player's next-game stat line —
passing yards, rushing yards, receiving yards, receptions — from leak-free
rolling usage/efficiency, opponent positional defense, and game context taken
from the team model's feature table.

    python props.py --validate     walk-forward backtest vs naive baseline
    python props.py                train on everything, project the next week
                                   -> props_projections.csv

A prop bet is P(stat > line), so the deliverable is a distribution, not a
point estimate: prob_over() interpolates the quantiles into a CDF.
"""

import argparse

import numpy as np
import pandas as pd
import nflreadpy as nfl
from xgboost import XGBRegressor

from features import (LAST_SEASON, canon, load_depth_qb1,
                      load_injury_reports, load_ir_index, norm_name)

FIRST_SEASON = 2006          # modern passing era; enough history for training
QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
VALIDATE_SEASONS = range(2021, 2026)

# stat -> (positions, usage column, minimum rolling usage to project)
STATS = {
    "passing_yards": (["QB"], "attempts", 15.0),
    "rushing_yards": (["RB", "QB"], "carries", 6.0),
    "receiving_yards": (["WR", "TE", "RB"], "targets", 3.0),
    "receptions": (["WR", "TE", "RB"], "targets", 3.0),
}

PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="reg:quantileerror",
    tree_method="hist",
    random_state=42,
)

FEATS = ["y4", "y10", "ystd10", "use4", "use10", "share4", "games_prior",
         "is_home", "is_dome", "team_elo_prob", "team_off_epa8",
         "opp_def_epa8", "opp_allowed8", "week"]

# Technique blocks promoted from the props_lab sweep (2026-08-13): each
# improved walk-forward pinball and the gains stacked additively. None move
# performance against closing lines — that gap is informational — but these
# make the projections themselves measurably sharper.
VOL_FEATS = ["team_vol8", "opp_vol_faced8", "share_t8", "share_trend"]
NGS_FEATS = ["ngs_sep4", "ngs_air_share4", "ngs_xyac_gap4", "ngs_catch4",
             "ngs_ryoe4"]
VAC_FEATS = ["vacated_share", "n_out_skill"]
FEATS_V2 = FEATS + VOL_FEATS + NGS_FEATS + VAC_FEATS

# team volume pool feeding each stat's opportunities
VOL_COL = {"passing_yards": "attempts", "rushing_yards": "carries",
           "receiving_yards": "targets", "receptions": "targets"}
TEAM_VOL = {"attempts": "attempts", "carries": "carries", "targets": "attempts"}


def load_player_weeks() -> pd.DataFrame:
    cols = ["season", "week", "team", "opponent_team", "player_id",
            "player_display_name", "position", "attempts", "passing_yards",
            "carries", "rushing_yards", "targets", "receptions",
            "receiving_yards", "target_share"]
    frames = []
    for season in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_player_stats([season], summary_level="week")
                          .select(cols).to_pandas())
        except Exception:
            continue  # season not published yet
    ps = pd.concat(frames, ignore_index=True)
    ps["team"] = ps["team"].map(canon)
    ps["opponent_team"] = ps["opponent_team"].map(canon)
    positions = sorted({p for pos, _, _ in STATS.values() for p in pos})
    ps = ps[ps["position"].isin(positions)].copy()
    ps = ps.sort_values(["season", "week"]).reset_index(drop=True)
    return ps


def game_context() -> pd.DataFrame:
    """(season, week, team) -> game context from the team model's features:
    home/away, dome, the team's Elo win prob, its offense form, and the
    OPPONENT's rolling defensive EPA allowed. Regular season only."""
    df = pd.read_parquet("features.parquet")
    df = df[df["game_type"] == "REG"]
    rows = []
    for r in df.itertuples(index=False):
        rows.append({"season": r.season, "week": r.week, "team": r.home_team,
                     "opp": r.away_team, "is_home": 1, "is_dome": r.is_dome,
                     "team_elo_prob": r.elo_prob,
                     "team_off_epa8": r.home_off_epa8,
                     "opp_def_epa8": r.away_def_epa8,
                     "gameday": r.gameday, "played": pd.notna(r.home_score)})
        rows.append({"season": r.season, "week": r.week, "team": r.away_team,
                     "opp": r.home_team, "is_home": 0, "is_dome": r.is_dome,
                     "team_elo_prob": 1 - r.elo_prob,
                     "team_off_epa8": r.away_off_epa8,
                     "opp_def_epa8": r.home_def_epa8,
                     "gameday": r.gameday, "played": pd.notna(r.home_score)})
    return pd.DataFrame(rows)


def _allowed_per_game(ps: pd.DataFrame, stat: str, positions: list) -> pd.DataFrame:
    per_game = (ps[ps["position"].isin(positions)]
                .groupby(["season", "week", "opponent_team"])[stat]
                .sum().reset_index()
                .rename(columns={"opponent_team": "defteam", stat: "allowed"}))
    return per_game.sort_values(["season", "week"])


def defense_allowed(ps: pd.DataFrame, stat: str, positions: list) -> pd.DataFrame:
    """Rolling 8-game mean of `stat` a defense allows per game to these
    positions, shifted so a game's value only reflects PRIOR games."""
    per_game = _allowed_per_game(ps, stat, positions)
    per_game["opp_allowed8"] = (
        per_game.groupby("defteam")["allowed"]
        .transform(lambda s: s.shift(1).rolling(8, min_periods=3).mean()))
    return per_game[["season", "week", "defteam", "opp_allowed8"]]


def current_allowed_map(ps: pd.DataFrame, stat: str, positions: list) -> dict:
    """defteam -> mean allowed over its most recent 8 completed games, for
    projecting a week that has no stat rows yet."""
    per_game = _allowed_per_game(ps, stat, positions)
    return (per_game.groupby("defteam")["allowed"]
            .apply(lambda s: s.tail(8).mean()).to_dict())


def build_stat_table(ps: pd.DataFrame, ctx: pd.DataFrame, stat: str) -> pd.DataFrame:
    """One modeling row per player-game: leak-free rolling features + context.
    Rolling windows use shift(1) so a game never sees its own line."""
    positions, use_col, _ = STATS[stat]
    d = ps[ps["position"].isin(positions)].copy()
    d = d.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = d.groupby("player_id")
    d["y"] = d[stat].fillna(0)
    d["use"] = d[use_col].fillna(0)
    d["y4"] = g["y"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["y10"] = g["y"].transform(lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    d["ystd10"] = g["y"].transform(lambda s: s.shift(1).rolling(10, min_periods=3).std())
    d["use4"] = g["use"].transform(lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["use10"] = g["use"].transform(lambda s: s.shift(1).rolling(10, min_periods=2).mean())
    d["share4"] = g["target_share"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean()).fillna(0)
    d["games_prior"] = g.cumcount()
    d = d.merge(ctx.drop(columns=["gameday"]),
                on=["season", "week", "team"], how="inner")
    d = d.merge(defense_allowed(ps, stat, positions),
                left_on=["season", "week", "opp"],
                right_on=["season", "week", "defteam"], how="left")
    return d[d["y4"].notna() & (d["games_prior"] >= 2)].copy()


def load_current_teams() -> dict:
    """gsis_id -> current team from the official depth-chart feed (all
    positions, latest snapshot). Projection candidates are keyed to THIS,
    not their last observed team — otherwise offseason trades misattribute
    projections and retired players ghost-project under old teams."""
    try:
        dc = nfl.load_depth_charts([LAST_SEASON]).to_pandas()
    except Exception:
        return {}
    dc = dc[dc["gsis_id"].notna()].sort_values("dt")
    latest = dc.groupby("gsis_id").tail(1)
    return {r.gsis_id: canon(r.team) for r in latest.itertuples(index=False)}


def load_ngs() -> pd.DataFrame:
    """Next Gen Stats tracking-derived weekly features (2016+), receiving and
    rushing merged, keyed by (season, week, gsis player id)."""
    frames = []
    for kind in ("receiving", "rushing"):
        n = nfl.load_nextgen_stats(seasons=True, stat_type=kind).to_pandas()
        n = n[n["week"] > 0]  # week 0 rows are season aggregates
        cols = {"receiving": {
                    "avg_separation": "ngs_sep",
                    "percent_share_of_intended_air_yards": "ngs_air_share",
                    "catch_percentage": "ngs_catch"},
                "rushing": {
                    "rush_yards_over_expected_per_att": "ngs_ryoe"}}[kind]
        keep = ["season", "week", "player_gsis_id"] + list(cols)
        if kind == "receiving":
            n["xyac_gap"] = n["avg_yac"] - n["avg_expected_yac"]
            keep.append("xyac_gap")
        n = n[keep].rename(columns={**cols, "xyac_gap": "ngs_xyac_gap",
                                    "player_gsis_id": "player_id"})
        frames.append(n)
    return frames[0].merge(frames[1], on=["season", "week", "player_id"],
                           how="outer")


def add_volume_block(d: pd.DataFrame, ps: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Game-script volume: the team's opportunity pool, what the opponent's
    defense concedes, and the player's rolled share of his team's pool."""
    pool_col = TEAM_VOL[VOL_COL[stat]]
    tp = (ps.groupby(["season", "week", "team"])[pool_col].sum()
          .reset_index().rename(columns={pool_col: "pool"}))
    tp = tp.sort_values(["season", "week"])
    tp["team_vol8"] = tp.groupby("team")["pool"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(tp[["season", "week", "team", "pool", "team_vol8"]],
                on=["season", "week", "team"], how="left")
    fp = (ps.groupby(["season", "week", "opponent_team"])[pool_col].sum()
          .reset_index().rename(columns={pool_col: "faced",
                                         "opponent_team": "fdefteam"}))
    fp = fp.sort_values(["season", "week"])
    fp["opp_vol_faced8"] = fp.groupby("fdefteam")["faced"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(fp[["season", "week", "fdefteam", "opp_vol_faced8"]],
                left_on=["season", "week", "opp"],
                right_on=["season", "week", "fdefteam"], how="left")
    d["share_raw"] = d["use"] / d["pool"].replace(0, np.nan)
    g = d.groupby("player_id")
    d["share_t8"] = g["share_raw"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    share_t4 = g["share_raw"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["share_trend"] = share_t4 - d["share_t8"]
    return d


def add_ngs_block(d: pd.DataFrame, ngs: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(ngs, on=["season", "week", "player_id"], how="left")
    g = d.groupby("player_id")
    for col in ["ngs_sep", "ngs_air_share", "ngs_xyac_gap", "ngs_catch",
                "ngs_ryoe"]:
        d[f"{col}4"] = g[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    return d


def add_vacated_block(d: pd.DataFrame, injuries: dict) -> pd.DataFrame:
    """Share of the team's volume pool vacated by skill players ruled Out
    this week, valued at each absent player's own last-known rolling share."""
    from features import norm_name as _norm
    seq: dict = {}
    tmp = d[["player_display_name", "season", "week", "team", "share_t8"]]
    for r in tmp.sort_values(["season", "week"]).itertuples(index=False):
        if pd.notna(r.share_t8):
            seq.setdefault((r.team, _norm(r.player_display_name)), []).append(
                (r.season * 100 + r.week, r.share_t8))

    def share_asof(team, norm, key):
        val = 0.0
        for k, s in seq.get((team, norm), []):
            if k >= key:
                break
            val = s
        return val

    vac, n_out = [], []
    for r in d.itertuples(index=False):
        rep = injuries.get((r.season, r.week, r.team))
        key = r.season * 100 + r.week
        total = cnt = 0.0
        if rep:
            for p in rep["out_players"]:
                if p["group"] in ("wr", "te", "rb"):
                    total += share_asof(r.team, p["norm"], key)
                    cnt += 1
        vac.append(total)
        n_out.append(cnt)
    d["vacated_share"] = vac
    d["n_out_skill"] = n_out
    return d


def build_stat_table_v2(ps, ctx, stat, ngs, injuries) -> pd.DataFrame:
    """The production table: base features + the promoted technique blocks."""
    d = build_stat_table(ps, ctx, stat)
    d = add_volume_block(d, ps, stat)
    d = add_ngs_block(d, ngs)
    d = add_vacated_block(d, injuries)
    return d


def fit_quantiles(train: pd.DataFrame, feats: list = FEATS) -> dict:
    models = {}
    for q in QUANTILES:
        m = XGBRegressor(**PARAMS, quantile_alpha=q)
        m.fit(train[feats], train["y"])
        models[q] = m
    return models


def predict_quantiles(models: dict, rows: pd.DataFrame,
                      feats: list = FEATS) -> np.ndarray:
    """(n, 5) quantile matrix, re-sorted row-wise: independently fit quantile
    models can cross, and a CDF must be monotone."""
    preds = np.column_stack([models[q].predict(rows[feats]) for q in QUANTILES])
    return np.maximum(np.sort(preds, axis=1), 0.0)


def prob_over(quants: np.ndarray, line: float) -> float:
    """P(stat > line) from the quantile grid: linear CDF interpolation inside
    P10-P90, flat-slope extrapolation outside, clipped away from 0/1 (a model
    should never claim certainty from five quantiles)."""
    qs = np.asarray(QUANTILES)
    v = np.asarray(quants, dtype=float)
    if line <= v[0]:
        below = qs[0] * (line / v[0]) if v[0] > 0 else 0.0
        return float(np.clip(1 - below, 0.02, 0.98))
    if line >= v[-1]:
        # extend the P75-P90 slope into the upper tail
        slope = (qs[-1] - qs[-2]) / max(v[-1] - v[-2], 1e-6)
        return float(np.clip(1 - (qs[-1] + slope * (line - v[-1])), 0.02, 0.98))
    cdf = float(np.interp(line, v, qs))
    return float(np.clip(1 - cdf, 0.02, 0.98))


def pinball(y: np.ndarray, pred: np.ndarray) -> float:
    losses = []
    for j, q in enumerate(QUANTILES):
        diff = y - pred[:, j]
        losses.append(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return float(np.mean(losses))


def naive_quantiles(test: pd.DataFrame, ps: pd.DataFrame, stat: str) -> np.ndarray:
    """Baseline a book could set half-asleep: the player's own last-10-games
    empirical quantiles (leak-free: prior games only)."""
    positions, _, _ = STATS[stat]
    d = ps[ps["position"].isin(positions)].copy()
    d = d.sort_values(["player_id", "season", "week"])
    d["y"] = d[stat].fillna(0)
    out = np.zeros((len(test), len(QUANTILES)))
    hist_map = {pid: grp["y"].to_numpy() for pid, grp in d.groupby("player_id")}
    for row_i, r in enumerate(test.itertuples(index=False)):
        h = hist_map.get(r.player_id)
        n = int(r.games_prior)
        window = h[max(0, n - 10):n] if h is not None else np.array([0.0])
        out[row_i] = np.quantile(window, QUANTILES)
    return out


def validate() -> None:
    ps = load_player_weeks()
    ctx = game_context()
    ngs = load_ngs()
    injuries = load_injury_reports()
    print(f"{'stat':16s} {'season':6s} {'n':>6s}  {'pinball':>8s} {'naive':>8s} "
          f"{'gain':>6s}  {'cov50':>6s} {'cov80':>6s}")
    for stat in STATS:
        table = build_stat_table_v2(ps, ctx, stat, ngs, injuries)
        table = table[table["played"]]
        agg_n = agg_p = agg_b = 0.0
        for season in VALIDATE_SEASONS:
            train = table[table.season < season]
            test = table[table.season == season]
            if test.empty:
                continue
            models = fit_quantiles(train, FEATS_V2)
            pred = predict_quantiles(models, test, FEATS_V2)
            y = test["y"].to_numpy()
            base = naive_quantiles(test, ps, stat)
            pb, nb = pinball(y, pred), pinball(y, base)
            cov50 = np.mean((y >= pred[:, 1]) & (y <= pred[:, 3]))
            cov80 = np.mean((y >= pred[:, 0]) & (y <= pred[:, 4]))
            agg_n += len(y); agg_p += pb * len(y); agg_b += nb * len(y)
            print(f"{stat:16s} {season:<6d} {len(y):6d}  {pb:8.3f} {nb:8.3f} "
                  f"{100*(nb-pb)/nb:5.1f}%  {cov50:6.3f} {cov80:6.3f}")
        print(f"{stat:16s} TOTAL  {int(agg_n):6d}  {agg_p/agg_n:8.3f} "
              f"{agg_b/agg_n:8.3f} {100*(agg_b/agg_n-agg_p/agg_n)/(agg_b/agg_n):5.1f}%\n")


def upcoming_week(ctx: pd.DataFrame) -> tuple:
    future = ctx[~ctx["played"]]
    if future.empty:
        raise SystemExit("No unplayed regular-season games in features.parquet")
    wk = future.sort_values("gameday").iloc[0]
    return int(wk.season), int(wk.week)


def project() -> None:
    ps = load_player_weeks()
    ctx = game_context()
    ngs = load_ngs()
    injuries = load_injury_reports()
    ngs_map = {pid: grp for pid, grp in ngs.groupby("player_id")}
    cur_team = load_current_teams()
    depth_qb1 = load_depth_qb1()
    ir_index, ir_latest = load_ir_index()
    season, week = upcoming_week(ctx)

    def team_ir(team: str) -> set:
        players = ir_index.get((season, week, team))
        if players is None:
            latest = ir_latest.get((season, team))  # same season only —
            players = latest[1] if latest else []   # last year's IR is stale
        return {p["gsis"] for p in players}
    week_ctx = ctx[(ctx.season == season) & (ctx.week == week)]
    print(f"Projecting {season} week {week} "
          f"({len(week_ctx)} team slates)")

    out_rows = []
    for stat, (positions, use_col, min_use) in STATS.items():
        table = build_stat_table_v2(ps, ctx, stat, ngs, injuries)
        models = fit_quantiles(table[table["played"]], FEATS_V2)

        # Current-state lookups for the block features
        pool_col = TEAM_VOL[VOL_COL[stat]]
        tp = (ps.groupby(["season", "week", "team"])[pool_col].sum()
              .reset_index().rename(columns={pool_col: "pool"})
              .sort_values(["season", "week"]))
        pool_map = {(r.season, r.week, r.team): r.pool
                    for r in tp.itertuples(index=False)}
        team_vol_now = tp.groupby("team")["pool"].apply(
            lambda s: s.tail(8).mean()).to_dict()
        fp = (ps.groupby(["season", "week", "opponent_team"])[pool_col].sum()
              .reset_index().sort_values(["season", "week"]))
        opp_faced_now = fp.groupby("opponent_team")[pool_col].apply(
            lambda s: s.tail(8).mean()).to_dict()
        played_t = table[table["played"] & table["share_t8"].notna()]
        last_share = {(r.team, norm_name(r.player_display_name)): r.share_t8
                      for r in played_t.sort_values(["season", "week"])
                      .itertuples(index=False)}

        # Candidates: players seen in the last ~1.5 seasons, keyed to their
        # most recent team; rolling features come from their full tail.
        d = ps[ps["position"].isin(positions)].copy()
        d = d.sort_values(["player_id", "season", "week"])
        d["y"] = d[stat].fillna(0)
        d["use"] = d[use_col].fillna(0)
        last = d.groupby("player_id").tail(1)
        recent = last[last.season >= season - 1]
        allowed_now = current_allowed_map(ps, stat, positions)
        for r in recent.itertuples(index=False):
            team_now = cur_team.get(r.player_id) if cur_team else r.team
            if team_now is None:
                continue  # not on any current roster (retired/unsigned)
            # Only the depth-chart starter projects at QB — a backup with
            # trailing usage data (Winston-on-NYG case) isn't playing.
            if (r.position == "QB" and depth_qb1.get(team_now)
                    not in (None, r.player_id)):
                continue
            h = d[d.player_id == r.player_id]
            if len(h) < 2:
                continue
            tail4, tail10 = h.tail(4), h.tail(10)
            if tail4["use"].mean() < min_use:
                continue
            game = week_ctx[week_ctx.team == team_now]
            if game.empty:
                continue  # bye week, or player's team not playing
            game = game.iloc[0]
            shares = [u / pool_map[k] for u, k in zip(
                h["use"], zip(h["season"], h["week"], h["team"]))
                if pool_map.get(k)]
            ngs_p = ngs_map.get(r.player_id)
            ngs_tail = (ngs_p.tail(4) if ngs_p is not None else None)

            def ngs_val(col):
                if ngs_tail is None or ngs_tail[col].notna().sum() == 0:
                    return np.nan
                return float(ngs_tail[col].mean())

            rep = injuries.get((season, week, team_now))
            vac = n_out = 0.0
            if rep:
                for p in rep["out_players"]:
                    if p["group"] in ("wr", "te", "rb"):
                        vac += last_share.get((team_now, p["norm"]), 0.0)
                        n_out += 1
            row = pd.DataFrame([{
                "y4": tail4["y"].mean(), "y10": tail10["y"].mean(),
                "ystd10": tail10["y"].std(), "use4": tail4["use"].mean(),
                "use10": tail10["use"].mean(),
                "share4": tail4["target_share"].fillna(0).mean(),
                "games_prior": len(h), "is_home": game.is_home,
                "is_dome": game.is_dome, "team_elo_prob": game.team_elo_prob,
                "team_off_epa8": game.team_off_epa8,
                "opp_def_epa8": game.opp_def_epa8,
                "opp_allowed8": allowed_now.get(game.opp, np.nan),
                "week": week,
                "team_vol8": team_vol_now.get(team_now, np.nan),
                "opp_vol_faced8": opp_faced_now.get(game.opp, np.nan),
                "share_t8": float(np.mean(shares[-8:])) if shares else np.nan,
                "share_trend": (float(np.mean(shares[-4:]) - np.mean(shares[-8:]))
                                if len(shares) >= 4 else np.nan),
                "ngs_sep4": ngs_val("ngs_sep"),
                "ngs_air_share4": ngs_val("ngs_air_share"),
                "ngs_xyac_gap4": ngs_val("ngs_xyac_gap"),
                "ngs_catch4": ngs_val("ngs_catch"),
                "ngs_ryoe4": ngs_val("ngs_ryoe"),
                "vacated_share": vac, "n_out_skill": n_out,
            }])
            q = predict_quantiles(models, row, FEATS_V2)[0]
            # History vs THIS opponent (team level — CB-on-WR matchup data
            # is not public), for the dashboard's context line.
            vs = h[h["opponent_team"] == game.opp]
            vs_log = ";".join(f"'{int(v.season) % 100} wk{int(v.week)}: {v.y:g}"
                              for v in vs.tail(5).itertuples(index=False))
            # Active status: game-week injury report + same-season IR list.
            pnorm = norm_name(r.player_display_name)
            status = ""
            if rep:
                if any(p.get("gsis") == r.player_id or p["norm"] == pnorm
                       for p in rep["out_players"]):
                    status = "OUT"
                elif any(p.get("gsis") == r.player_id or p["norm"] == pnorm
                         for p in rep.get("quest_players", [])):
                    status = "Q"
            if not status and r.player_id in team_ir(team_now):
                status = "IR"
            out_rows.append({
                "season": season, "week": week, "stat": stat,
                "player_id": r.player_id, "player": r.player_display_name,
                "position": r.position, "team": team_now, "opp": game.opp,
                "is_home": int(game.is_home),
                "p10": round(q[0], 1), "p25": round(q[1], 1),
                "p50": round(q[2], 1), "p75": round(q[3], 1),
                "p90": round(q[4], 1),
                "vs_opp_n": len(vs),
                "vs_opp_avg": round(float(vs["y"].mean()), 1) if len(vs) else None,
                "vs_opp_log": vs_log,
                "career_avg": round(float(h["y"].mean()), 1),
                "status": status,
            })

    out = pd.DataFrame(out_rows).sort_values(["stat", "p50"], ascending=[True, False])
    out.to_csv("props_projections.csv", index=False)
    print(f"Wrote props_projections.csv ({len(out)} projections, "
          f"{out.player_id.nunique()} players)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="walk-forward backtest instead of projecting")
    args = ap.parse_args()
    validate() if args.validate else project()
