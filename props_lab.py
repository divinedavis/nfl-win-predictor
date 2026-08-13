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
from props import (FEATS, NGS_FEATS, PARAMS, QUANTILES, STATS, VAC_FEATS,
                   VOL_COL, VOL_FEATS, add_ngs_block, add_vacated_block,
                   add_volume_block, build_stat_table, game_context, load_ngs,
                   load_player_weeks, prob_over)
from real_props_backtest import LINES_DIR

EVAL_SEASONS = [2023, 2024, 2025]
LINE_SEASONS = [2024, 2025]
RNG = np.random.default_rng(42)
N_SIMS = 500

CONFIGS = {
    "gbm-base":  ("gbm", []),
    "gbm+vol":   ("gbm", VOL_FEATS),
    "gbm+ngs":   ("gbm", NGS_FEATS),
    "gbm+vac":   ("gbm", VAC_FEATS),
    "gbm+all":   ("gbm", VOL_FEATS + NGS_FEATS + VAC_FEATS),
    "comp-base": ("comp", []),
    "comp+all":  ("comp", VOL_FEATS + NGS_FEATS + VAC_FEATS),
}


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
