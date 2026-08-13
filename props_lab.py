"""Research lab: candidate techniques for predicting player stats, scored on
(a) walk-forward pinball loss and (b) the only metric that matters — hit rate
and ROI against archived real DraftKings closing lines (2024-2025).

Techniques (see CONFIGS):
  feature blocks   vol (game-script/team volume), ngs (tracking-derived),
                   vac (vacated share from teammates ruled out)
  architectures    gbm  = direct quantile GBMs (current production approach)
                   comp = composition: volume GBM x empirical-Bayes efficiency
                          posteriors, Monte-Carlo composed into quantiles
  ensemble         mean of the best gbm and comp quantiles

    python props_lab.py --stats receiving_yards receptions rushing_yards
"""

import argparse

import numpy as np
import pandas as pd
import nflreadpy as nfl
from xgboost import XGBRegressor

from features import load_injury_reports, norm_name
from paper_trade import implied, payout
from props import (FEATS, PARAMS, QUANTILES, STATS, build_stat_table,
                   game_context, load_player_weeks, prob_over)
from real_props_backtest import LINES_DIR

EVAL_SEASONS = [2023, 2024, 2025]
LINE_SEASONS = [2024, 2025]
RNG = np.random.default_rng(42)
N_SIMS = 500

VOL_COL = {"receiving_yards": "targets", "receptions": "targets",
           "rushing_yards": "carries"}
TEAM_VOL = {"targets": "attempts", "carries": "carries"}  # team volume pool

VOL_FEATS = ["team_vol8", "opp_vol_faced8", "share_t8", "share_trend"]
NGS_FEATS = ["ngs_sep4", "ngs_air_share4", "ngs_xyac_gap4", "ngs_catch4",
             "ngs_ryoe4"]
VAC_FEATS = ["vacated_share", "n_out_skill"]

CONFIGS = {
    "gbm-base":  ("gbm", []),
    "gbm+vol":   ("gbm", VOL_FEATS),
    "gbm+ngs":   ("gbm", NGS_FEATS),
    "gbm+vac":   ("gbm", VAC_FEATS),
    "gbm+all":   ("gbm", VOL_FEATS + NGS_FEATS + VAC_FEATS),
    "comp-base": ("comp", []),
    "comp+all":  ("comp", VOL_FEATS + NGS_FEATS + VAC_FEATS),
}


# ---------- technique feature blocks (all leak-free: shift(1) rolls) ----------

def add_volume_block(d: pd.DataFrame, ps: pd.DataFrame, stat: str) -> pd.DataFrame:
    """Game-script volume: how many opportunities the TEAM generates, how many
    the opponent's defense faces, and the player's share of his team's pool."""
    vol = VOL_COL[stat]
    team_pool = TEAM_VOL[vol]
    tp = (ps.groupby(["season", "week", "team"])[team_pool].sum()
          .reset_index().rename(columns={team_pool: "pool"}))
    tp = tp.sort_values(["season", "week"])
    tp["team_vol8"] = tp.groupby("team")["pool"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(tp[["season", "week", "team", "pool", "team_vol8"]],
                on=["season", "week", "team"], how="left")
    fp = (ps.groupby(["season", "week", "opponent_team"])[team_pool].sum()
          .reset_index().rename(columns={team_pool: "faced",
                                         "opponent_team": "defteam"}))
    fp = fp.sort_values(["season", "week"])
    fp["opp_vol_faced8"] = fp.groupby("defteam")["faced"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=3).mean())
    d = d.merge(fp[["season", "week", "defteam", "opp_vol_faced8"]],
                left_on=["season", "week", "opp"],
                right_on=["season", "week", "defteam"],
                how="left", suffixes=("", "_f"))
    # Player's share of the team pool, rolled from his own past games.
    d["share_raw"] = d["use"] / d["pool"].replace(0, np.nan)
    g = d.groupby("player_id")
    d["share_t8"] = g["share_raw"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=2).mean())
    share_t4 = g["share_raw"].transform(
        lambda s: s.shift(1).rolling(4, min_periods=2).mean())
    d["share_trend"] = share_t4 - d["share_t8"]
    return d


def load_ngs() -> pd.DataFrame:
    frames = []
    for kind in ("receiving", "rushing"):
        n = nfl.load_nextgen_stats(seasons=True, stat_type=kind).to_pandas()
        n = n[n["week"] > 0]  # week 0 = season aggregate
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
    out = frames[0].merge(frames[1], on=["season", "week", "player_id"],
                          how="outer")
    return out


def add_ngs_block(d: pd.DataFrame, ngs: pd.DataFrame) -> pd.DataFrame:
    d = d.merge(ngs, on=["season", "week", "player_id"], how="left")
    g = d.groupby("player_id")
    for col in ["ngs_sep", "ngs_air_share", "ngs_xyac_gap", "ngs_catch",
                "ngs_ryoe"]:
        d[f"{col}4"] = g[col].transform(
            lambda s: s.shift(1).rolling(4, min_periods=1).mean())
    return d


def add_vacated_block(d: pd.DataFrame, injuries: dict) -> pd.DataFrame:
    """Share of team volume vacated by skill players ruled Out this week —
    each Out teammate contributes his own last-known rolling share."""
    share_hist = d[["player_id", "season", "week", "team", "share_t8"]].copy()
    share_hist["norm"] = d["player_display_name"].map(norm_name)
    last_share: dict = {}
    for r in share_hist.sort_values(["season", "week"]).itertuples(index=False):
        if pd.notna(r.share_t8):
            last_share[(r.team, r.norm)] = (r.season * 100 + r.week, r.share_t8)

    # last_share holds the FINAL value; rebuild as sorted series for as-of
    seq: dict = {}
    for r in share_hist.sort_values(["season", "week"]).itertuples(index=False):
        if pd.notna(r.share_t8):
            seq.setdefault((r.team, r.norm), []).append(
                (r.season * 100 + r.week, r.share_t8))

    def share_asof(team, norm, key):
        hist = seq.get((team, norm))
        if not hist:
            return 0.0
        val = 0.0
        for k, s in hist:
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


# ---------- architectures ----------

def fit_predict_gbm(train, test, feats):
    preds = []
    for q in QUANTILES:
        m = XGBRegressor(**PARAMS, quantile_alpha=q)
        m.fit(train[feats], train["y"])
        preds.append(m.predict(test[feats]))
    return np.maximum(np.sort(np.column_stack(preds), axis=1), 0.0)


def fit_predict_comp(train, test, feats, stat):
    """Volume GBM (mean) + NB dispersion, composed with empirical-Bayes
    per-player efficiency posteriors by Monte Carlo."""
    vol_feats = [f for f in ["use4", "use10", "share4", "games_prior",
                             "is_home", "team_elo_prob", "opp_allowed8",
                             "week"] + feats if f in test.columns]
    mv = XGBRegressor(**{**PARAMS, "objective": "reg:squarederror"})
    mv.fit(train[vol_feats], train["use"])
    mu_te = np.maximum(mv.predict(test[vol_feats]), 0.1)
    mu_tr = np.maximum(mv.predict(train[vol_feats]), 0.1)
    resid_var = np.mean((train["use"] - mu_tr) ** 2)
    mean_mu = mu_tr.mean()
    r = max(mean_mu ** 2 / max(resid_var - mean_mu, 0.05), 1.0)  # NB size

    k = 30.0 if VOL_COL[stat] == "targets" else 40.0
    eff_tr = train.groupby("player_id").agg(sy=("y", "sum"), sv=("use", "sum"))
    league = train["y"].sum() / max(train["use"].sum(), 1.0)
    post_mean = ((eff_tr["sy"] + k * league) / (eff_tr["sv"] + k)).to_dict()
    post_n = eff_tr["sv"].to_dict()
    per_event = train[train["use"] > 0]
    sigma_event = float(np.std(per_event["y"] / per_event["use"]))

    quants = np.zeros((len(test), len(QUANTILES)))
    for i, row in enumerate(test.itertuples(index=False)):
        mu = mu_te[i]
        p_nb = r / (r + mu)
        vol_draws = RNG.negative_binomial(r, p_nb, N_SIMS)
        m = post_mean.get(row.player_id, league)
        n_eff = post_n.get(row.player_id, 0.0)
        eff_sd = sigma_event / np.sqrt(1.0 + n_eff / k)
        if stat == "receptions":
            p_catch = np.clip(RNG.normal(m, eff_sd, N_SIMS), 0.05, 0.99)
            sims = RNG.binomial(vol_draws, p_catch)
        else:
            eff_draws = RNG.normal(m, eff_sd, N_SIMS)
            noise = RNG.normal(0, sigma_event * np.sqrt(np.maximum(vol_draws, 1)))
            sims = np.maximum(vol_draws * eff_draws + noise, 0.0)
        quants[i] = np.quantile(sims, QUANTILES)
    return quants


# ---------- evaluation ----------

def pinball(y, pred):
    losses = []
    for j, q in enumerate(QUANTILES):
        diff = y - pred[:, j]
        losses.append(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    return float(np.mean(losses))


def score_lines(test, quants, stat_lines):
    idx = {}
    for i, r in enumerate(test.itertuples(index=False)):
        idx[(r.week, r.norm, r.team)] = i
    rows = []
    for ln in stat_lines.itertuples(index=False):
        i = idx.get((ln.week, ln.norm, ln.home))
        if i is None:
            i = idx.get((ln.week, ln.norm, ln.away))
        if i is None:
            continue
        p_over = prob_over(quants[i], float(ln.line))
        p_side = p_over if ln.side == "Over" else 1 - p_over
        actual = float(test.iloc[i]["y"])
        if actual == ln.line:
            continue
        over_hit = actual > ln.line
        rows.append({"edge": p_side - implied(ln.odds), "odds": ln.odds,
                     "won": over_hit if ln.side == "Over" else not over_hit})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", nargs="*",
                    default=["receiving_yards", "receptions", "rushing_yards"])
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS))
    args = ap.parse_args()

    print("loading shared data...")
    ps = load_player_weeks()
    ctx = game_context()
    ngs = load_ngs()
    injuries = load_injury_reports()
    lines = pd.concat([pd.read_csv(f) for f in sorted(LINES_DIR.glob("*.csv"))],
                      ignore_index=True)
    lines = lines[lines["game_type"] == "REG"].dropna(subset=["line", "odds"])
    lines["norm"] = lines["player"].map(norm_name)

    all_bets: dict = {}
    for stat in args.stats:
        base = build_stat_table(ps, ctx, stat)
        base = base[base["played"]].copy()
        base = add_volume_block(base, ps, stat)
        base = add_ngs_block(base, ngs)
        base = add_vacated_block(base, injuries)
        base["norm"] = base["player_display_name"].map(norm_name)
        stat_lines = lines[lines["stat"] == stat]

        for name in args.configs:
            arch, extra = CONFIGS[name]
            feats = FEATS + [f for f in extra if f in base.columns]
            pin_n = pin_sum = 0.0
            bets = []
            for season in EVAL_SEASONS:
                train = base[base.season < season]
                test = base[base.season == season].reset_index(drop=True)
                if test.empty:
                    continue
                if arch == "gbm":
                    quants = fit_predict_gbm(train, test, feats)
                else:
                    quants = fit_predict_comp(train, test, extra, stat)
                y = test["y"].to_numpy()
                pin_sum += pinball(y, quants) * len(y)
                pin_n += len(y)
                if season in LINE_SEASONS and len(stat_lines):
                    bets.append(score_lines(
                        test, quants, stat_lines[stat_lines.season == season]))
            b = (pd.concat(bets, ignore_index=True) if bets
                 else pd.DataFrame(columns=["edge", "odds", "won"]))
            all_bets[(stat, name)] = b
            e5 = b[b.edge >= 0.05]
            roi = (np.mean(np.where(e5.won, [payout(o) for o in e5.odds], -1.0))
                   if len(e5) else np.nan)
            print(f"{stat:16s} {name:10s} pinball {pin_sum/pin_n:7.3f}  "
                  f"line-bets@5% {len(e5):5d}  win {e5.won.mean() if len(e5) else float('nan'):.3f}  "
                  f"ROI {roi:+.3%}" if len(e5) else
                  f"{stat:16s} {name:10s} pinball {pin_sum/pin_n:7.3f}  (no line bets)")

    # combined-across-stats summary per config
    print("\n=== combined across stats (edge>=0.05, real DK lines 2024-25) ===")
    for name in args.configs:
        b = pd.concat([all_bets[(s, name)] for s in args.stats
                       if (s, name) in all_bets], ignore_index=True)
        e5 = b[b.edge >= 0.05]
        if not len(e5):
            continue
        pnl = np.where(e5.won, [payout(o) for o in e5.odds], -1.0)
        print(f"{name:10s} bets {len(e5):5d}  win {e5.won.mean():.3f}  "
              f"ROI {pnl.mean():+.3%}")


if __name__ == "__main__":
    main()
