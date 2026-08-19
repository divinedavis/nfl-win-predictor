"""Does the simulator's distribution SHAPE improve the spread call?

The production ATS path (export_web.py) does two things: it maps win probability
to an expected margin with a linear fit, and it turns that into a cover
probability by assuming the margin is normal with one league-wide sigma -- the
same spread for a divisional rock-fight and a blowout waiting to happen.

sim_backtest.py already showed the simulator's own expected margin is worse than
that linear map (53.2% ATS against 55.2%). But that test replaced both halves at
once, and the two are separable:

    centre  = where the margin distribution sits   -- production wins
    shape   = how it spreads out around that       -- untested

So this holds the production centre fixed and swaps only the shape, by shifting
each game's simulated margins to have the production expected margin as their
mean and reading the cover probability straight off the shifted sample. Four
variants are scored against what actually happened:

    A  production centre, fixed normal sigma      the current live behaviour
    B  production centre, simulated shape         the hybrid this exists to test
    C  simulated centre, fixed normal sigma
    D  simulated centre, simulated shape          what sim_backtest.py measured

If B does not beat A, the shape is worth nothing here and the honest move is to
leave the ATS path alone.

    python ats_shape_test.py
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.special import erf
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, ELO_BLEND

BREAK_EVEN = 0.524   # -110 juice


def production_margin_map(hist: pd.DataFrame, probs: np.ndarray) -> tuple:
    """The exact linear probability-to-margin fit export_web.py uses."""
    hist_prob = hist["home_prob"].to_numpy()
    hist_margin = (hist["home_score"] - hist["away_score"]).to_numpy()
    A = np.column_stack([np.ones(len(hist_prob)), hist_prob])
    beta = np.linalg.lstsq(A, hist_margin, rcond=None)[0]
    sigma = float(np.std(hist_margin - A @ beta))
    return beta[0] + beta[1] * probs, sigma


def normal_cover(margin: np.ndarray, spread: np.ndarray, sigma: float) -> np.ndarray:
    return 1 - 0.5 * (1 + erf((spread - margin) / (sigma * np.sqrt(2))))


def score(name: str, p: np.ndarray, covered: np.ndarray) -> dict:
    pick = p > 0.5
    return {"variant": name, "ats": float((pick == covered).mean()),
            "brier": brier_score_loss(covered, np.clip(p, 1e-6, 1 - 1e-6)),
            "n": len(covered)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="sim_backtest.csv")
    ap.add_argument("--margins", default="sim_margins.npz")
    args = ap.parse_args()

    sim = pd.read_csv(args.cache)
    margins = np.load(args.margins)
    feat = pd.read_parquet("features.parquet")
    done = feat[feat.home_win.notna()].copy()

    # Walk-forward production probability, exactly as train.py blends it.
    done["home_prob"] = np.nan
    for season in sorted(sim.season.unique()):
        tr, te = done[done.season < season], done[done.season == season]
        if te.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(tr[FEATURES], tr["home_win"])
        raw = m.predict_proba(te[FEATURES])[:, 1]
        done.loc[te.index, "home_prob"] = (
            (1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].to_numpy())

    hist = done[done.season < sim.season.min()].copy()
    hist_m = XGBClassifier(**PARAMS)
    train_hist = done[done.season < hist.season.max()]
    hist_m.fit(train_hist[FEATURES], train_hist["home_win"])
    hist = hist[hist.season == hist.season.max()].copy()
    hist["home_prob"] = ((1 - ELO_BLEND) * hist_m.predict_proba(hist[FEATURES])[:, 1]
                         + ELO_BLEND * hist["elo_prob"].to_numpy())

    df = sim.merge(done[["game_id", "home_prob"]], on="game_id", how="left")
    df = df[df.spread_line.notna() & (df.spread_line != 0)
            & (df.actual_margin != df.spread_line) & df.home_prob.notna()].copy()

    prod_margin, sigma = production_margin_map(hist, df.home_prob.to_numpy())
    df["prod_margin"] = prod_margin
    covered = (df.actual_margin > df.spread_line).to_numpy().astype(int)
    spread = df.spread_line.to_numpy()

    # Shape variants read the cover probability off the simulated sample,
    # re-centred where the variant says the margin should sit.
    def sampled_cover(centre: np.ndarray | None) -> np.ndarray:
        out = np.empty(len(df))
        for i, (gid, sp) in enumerate(zip(df.game_id, spread)):
            m = margins[gid].astype(float)
            if centre is not None:
                m = m - m.mean() + centre[i]
            out[i] = float((m > sp).mean())
        return out

    results = [
        score("A  production centre + normal sigma",
              normal_cover(df.prod_margin.to_numpy(), spread, sigma), covered),
        score("B  production centre + simulated shape",
              sampled_cover(df.prod_margin.to_numpy()), covered),
        score("C  simulated centre + normal sigma",
              normal_cover(df.sim_margin.to_numpy(), spread, sigma), covered),
        score("D  simulated centre + simulated shape",
              sampled_cover(None), covered),
    ]

    print(f"\n=== Does the simulated shape help against the spread? ===")
    print(f"   {len(df)} games with a line, {sim.season.min()}-{sim.season.max()}; "
          f"break-even {BREAK_EVEN:.1%}; production sigma {sigma:.2f}\n")
    print(f"   {'variant':40s} {'ATS':>7} {'brier':>8}")
    for r in results:
        print(f"   {r['variant']:40s} {r['ats']:7.4f} {r['brier']:8.4f}")

    a, b = results[0], results[1]
    d_ats, d_brier = b["ats"] - a["ats"], b["brier"] - a["brier"]
    # Paired standard error on the ATS difference: the two variants call the
    # same games, so only the games they disagree on carry information.
    pa = normal_cover(df.prod_margin.to_numpy(), spread, sigma) > 0.5
    pb = sampled_cover(df.prod_margin.to_numpy()) > 0.5
    disagree = int((pa != pb).sum())
    print(f"\n   B minus A:  ATS {d_ats:+.4f}   brier {d_brier:+.4f}")
    print(f"   the two disagree on {disagree} of {len(df)} games "
          f"({disagree / len(df):.1%})")
    if disagree:
        b_right = int(((pb == covered) & (pa != pb)).sum())
        se = np.sqrt(disagree) / 2 / len(df)
        print(f"   of those, the shape variant is right {b_right}/{disagree}"
              f"   (1 s.e. on the ATS gap is {se:.4f})")
    # Distinguish "no better" from "actually worse": the sign matters, and a
    # result two standard errors the wrong way is a finding, not a null.
    se = np.sqrt(max(disagree, 1)) / 2 / len(df)
    if d_ats > 2 * se and d_brier < 0:
        verdict = "worth adopting"
    elif d_ats < -2 * se:
        verdict = (f"WORSE by {abs(d_ats) / se:.1f} standard errors -- "
                   "leave the ATS path alone")
    else:
        verdict = "not distinguishable from noise -- leave the ATS path alone"
    print(f"\n   verdict: {verdict}")


if __name__ == "__main__":
    main()
