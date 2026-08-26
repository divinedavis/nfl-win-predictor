"""Does ESPN's FPI carry anything this model does not already have?

FPI is the only outside opinion here with real history, so it is the only one
that can be tested rather than merely displayed. It gets the same treatment as
every other candidate feature in this repo: build it, run the walk-forward
backtest, and promote only on a Brier improvement across a clear majority of
seasons.

Two questions, because they are not the same question:

  1. Does adding fpi_prob to FEATURES improve the model? (a horse race)
  2. Does FPI carry information the MARKET does not already have? (a
     regression -- put the market's probability and FPI's on the right-hand
     side together and see whose coefficient survives)

The second matters more. A source that only repeats the market is worth showing
on the page for transparency and worth nothing as a feature.

    python fpi_ablation.py
"""

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import brier_score_loss  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from features import FEATURES  # noqa: E402
from train import BACKTEST_SEASONS, ELO_BLEND, PARAMS  # noqa: E402

LOGIT = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))


def main() -> None:
    df = pd.read_parquet("features.parquet")
    fpi = pd.read_csv("espn_fpi.csv")[["game_id", "fpi_home_prob"]]
    df = df.merge(fpi, on="game_id", how="left")
    done = df[df.home_win.notna()].copy()
    cov = done[done.season >= 2015].fpi_home_prob.notna().mean()
    print(f"FPI coverage on completed 2015+ games: {cov:.1%}")

    rows = []
    for season in BACKTEST_SEASONS:
        tr, te = done[done.season < season], done[done.season == season]
        if te.empty:
            continue
        out = {"season": season, "n": len(te), "y": te.home_win.values,
               "week": te.week.values, "fpi": te.fpi_home_prob.values,
               "mkt": te.mkt_prob.values}
        for tag, feats in [("base", FEATURES), ("fpi", FEATURES + ["fpi_home_prob"])]:
            m = XGBClassifier(**PARAMS)
            m.fit(tr[feats], tr.home_win)
            out[tag] = ((1 - ELO_BLEND) * m.predict_proba(te[feats])[:, 1]
                        + ELO_BLEND * te.elo_prob.values)
        rows.append(pd.DataFrame(out))
    d = pd.concat(rows, ignore_index=True)

    print("\n=== 1. horse race: does adding fpi_home_prob help? ===")
    print(f"{'variant':>22} {'accuracy':>9} {'brier':>8}")
    for tag, name in [("base", "shipped FEATURES"), ("fpi", "+ fpi_home_prob")]:
        acc = ((d[tag] > .5).astype(float) == d.y).mean()
        print(f"{name:>22} {acc:>9.4f} {brier_score_loss(d.y, d[tag]):>8.4f}")
    better = sum(brier_score_loss(g.y, g.fpi) < brier_score_loss(g.y, g.base)
                 for _, g in d.groupby("season"))
    print(f"  Brier improved in {better}/{d.season.nunique()} seasons "
          f"— promote only on a clear majority")

    print("\n=== 2. marginal information over the market ===")
    v = d[d.fpi.notna() & d.mkt.notna()]
    rng = np.random.default_rng(0)

    def marginal(mask, label):
        s = v[mask]
        if len(s) < 200:
            return
        X = np.column_stack([LOGIT(s.mkt), LOGIT(s.fpi)])
        y = s.y.values
        bs = np.array([LogisticRegression(max_iter=2000, C=np.inf)
                       .fit(X[i := rng.integers(0, len(s), len(s))], y[i]).coef_[0]
                       for _ in range(300)])
        b = LogisticRegression(max_iter=2000, C=np.inf).fit(X, y).coef_[0]
        lo, hi = np.percentile(bs[:, 1], [2.5, 97.5])
        verdict = "ADDS INFO" if lo > 0 else "nothing beyond the market"
        print(f"  {label:18s} n={len(s):5d}  fpi coef {b[1]:+.3f} "
              f"CI[{lo:+.3f},{hi:+.3f}]  P(>0)={(bs[:,1]>0).mean():.1%}  {verdict}")

    marginal(v.index >= 0, "all games")
    marginal(v.week <= 4, "weeks 1-4")
    marginal(v.week > 4, "weeks 5+")

    print("\n=== 3. standalone accuracy, for context ===")
    for name, p in [("FPI alone", v.fpi), ("market alone", v.mkt),
                    ("shipped model", v.base)]:
        acc = ((p > .5).astype(float) == v.y).mean()
        print(f"  {name:16s} {acc:.4f}  brier {brier_score_loss(v.y, p):.4f}")


if __name__ == "__main__":
    main()
