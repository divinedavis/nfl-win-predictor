"""Test RJ's rule of thumb: build a spread from preseason Vegas win totals.

  predicted home margin = (home_win_total - away_win_total) * MULT + HFA

RJ's numbers: MULT 5.5, HFA 1.75, claimed ~58% ATS over the last 7 years.
This backtests the literal rule, sweeps the multiplier the data actually wants,
slices it by week and by size of the win-total gap, and compares it to the game
model already in the repo.

Bets are graded against nflverse spread_line (verified = DK closing line) at
-110: breakeven 52.38%.

VERDICT (2026-08-24, 4,347 games 2010-2025): NO EDGE, do not bet it.
  - As stated (x5.5) it is a coin flip: 50.32% ATS, -3.9% ROI over 4,203 bets;
    49.87% over the last 7 seasons, nowhere near the claimed 58%.
  - x5.5 is the wrong scale. Regressing actual margin on the win-total gap
    gives 1.76 pts per win, and the closing line itself prices 1.60 pts per
    win -- so the market already applies this rule, with a better constant.
  - At the best multiplier in a full sweep it still only reaches 51.3%, under
    the 52.38% breakeven, and no edge filter rescues it.
  - The one live-looking slice is early season -- weeks 1-4 at x1.75 hit
    54.07% (n=958, +3.2% ROI) and the edge decays monotonically to 47.9% by
    week 15+, which is the right shape for preseason info going stale. It is
    not significant (p=0.155 vs breakeven, 9/16 seasons) and it is weaker in
    the recent half (55.7% 2010-17 vs 52.5% 2018-25). Not enough to bet.
"""
import argparse
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

ROOT = Path(__file__).parent
RJ_MULT, RJ_HFA = 5.5, 1.75
BREAKEVEN = 11 / 21  # -110


def load():
    wt = pd.read_csv(ROOT / "win_totals.csv")[["season", "team", "win_total"]]
    g = (pl.read_parquet(ROOT / "features.parquet")
         .select("season", "week", "home_team", "away_team",
                 "home_score", "away_score", "spread_line")
         .to_pandas())
    g = g[g.home_score.notna() & g.spread_line.notna()].copy()
    g["margin"] = g.home_score - g.away_score
    g = g.merge(wt.rename(columns={"team": "home_team", "win_total": "home_wt"}),
                on=["season", "home_team"], how="inner")
    g = g.merge(wt.rename(columns={"team": "away_team", "win_total": "away_wt"}),
                on=["season", "away_team"], how="inner")
    g["wt_diff"] = g.home_wt - g.away_wt
    return g


def grade(g, mult, hfa):
    """Return per-bet outcome: 1 win, 0 loss, NaN push/no-bet."""
    pred = g.wt_diff * mult + hfa
    live = pred != g.spread_line                       # no edge -> no bet
    pick_home = pred > g.spread_line
    cover = np.where(pick_home, g.margin > g.spread_line, g.margin < g.spread_line)
    push = g.margin == g.spread_line
    out = pd.Series(np.where(cover, 1.0, 0.0), index=g.index)
    out[push | ~live] = np.nan
    return out


def summarize(label, g, res, per_season=False):
    d = res.dropna()
    n, w = len(d), int(d.sum())
    if n == 0:
        print(f"{label}: no bets")
        return
    pct = w / n
    roi = (w * (100 / 110) - (n - w)) / n
    print(f"{label}: {pct:6.2%}  ({w}-{n - w}, n={n})  ROI {roi:+6.2%}")
    if per_season:
        for s, idx in g.groupby("season").groups.items():
            ds = res.loc[idx].dropna()
            if len(ds):
                print(f"    {s}: {ds.mean():6.2%}  ({int(ds.sum())}-{len(ds) - int(ds.sum())})")


def slices(g, mult):
    """Where, if anywhere, does the rule hold up: by week and by gap size."""
    from scipy import stats

    pred = g.wt_diff * mult + RJ_HFA
    res = grade(g, mult, RJ_HFA)
    print(f"\n--- slices at x{mult} ---")
    for lo, hi, label in [(1, 4, "weeks 1-4"), (5, 9, "weeks 5-9"),
                          (10, 14, "weeks 10-14"), (15, 22, "weeks 15+")]:
        d = res[(g.week >= lo) & (g.week <= hi)].dropna()
        print(f"  {label:<14} {d.mean():6.2%}  (n={len(d)})")
    for lo, hi, label in [(0, 2, "gap 0-2 wins"), (2, 4, "gap 2-4"), (4, 99, "gap 4+")]:
        d = res[(g.wt_diff.abs() >= lo) & (g.wt_diff.abs() < hi)].dropna()
        print(f"  {label:<14} {d.mean():6.2%}  (n={len(d)})")

    early = res[g.week <= 4].dropna()
    w, n = int(early.sum()), len(early)
    p = stats.binomtest(w, n, BREAKEVEN, alternative="greater").pvalue
    print(f"\n  weeks 1-4 detail: {w}/{n} = {w / n:.2%}, "
          f"p(beats breakeven) = {p:.3f}")
    seasons = g.loc[early.index].season
    above = sum(1 for _, d in early.groupby(seasons) if d.mean() > BREAKEVEN)
    print(f"    above breakeven in {above}/{seasons.nunique()} seasons")
    half = seasons <= seasons.median()
    print(f"    older half {early[half].mean():.2%} (n={half.sum()})  |  "
          f"recent half {early[~half].mean():.2%} (n={(~half).sum()})")


def vs_model(g, mult):
    """Does the win-total rule add anything to the shipped ATS model?

    Same walk-forward as ats_backtest.py: train on prior seasons, map blended
    win prob -> expected margin, bet the side the model likes vs the closing
    line.  Then check the rule as a co-signer and as a blend partner.
    """
    import pandas as pd
    from xgboost import XGBClassifier
    from features import FEATURES
    from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND

    df = pd.read_parquet(ROOT / "features.parquet")
    done = df[df.home_win.notna()].copy()
    done["margin"] = done.home_score - done.away_score
    wt = g[["season", "home_team", "away_team", "week", "wt_diff"]]
    done = done.merge(wt, on=["season", "home_team", "away_team", "week"], how="left")

    rows = []
    for season in BACKTEST_SEASONS:
        tr, te = done[done.season < season], done[done.season == season]
        if te.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(tr[FEATURES], tr["home_win"])
        p = ((1 - ELO_BLEND) * m.predict_proba(te[FEATURES])[:, 1]
             + ELO_BLEND * te["elo_prob"].values)
        tr_p = ((1 - ELO_BLEND) * m.predict_proba(tr[FEATURES])[:, 1]
                + ELO_BLEND * tr["elo_prob"].values)
        A = np.column_stack([np.ones(len(tr_p)), tr_p])
        beta = np.linalg.lstsq(A, tr["margin"].values, rcond=None)[0]
        rows.append(te.assign(mm=beta[0] + beta[1] * p))

    v = pd.concat(rows)
    v = v[v.spread_line.notna() & (v.margin != v.spread_line) & v.wt_diff.notna()]
    v = v.reset_index(drop=True)
    rule = v.wt_diff * mult + RJ_HFA

    def ats(pred, mask=None):
        d = v if mask is None else v[mask]
        pr = pred if mask is None else pred[mask]
        cov = np.where(pr > d.spread_line, d.margin > d.spread_line,
                       d.margin < d.spread_line)
        return cov.mean(), len(d)

    print(f"\n--- vs the shipped ATS model ({len(v)} games, "
          f"{v.season.min()}-{v.season.max()}, rule at x{mult}) ---")
    for label, pct_n in [("model alone", ats(v.mm)),
                         ("rule alone", ats(rule))]:
        print(f"  {label:<22} {pct_n[0]:6.2%}  (n={pct_n[1]})")

    agree = (v.mm > v.spread_line) == (rule > v.spread_line)
    pct, n = ats(v.mm, agree)
    print(f"  model, rule agrees     {pct:6.2%}  (n={n}, {n / len(v):.0%} of games)")
    pct, n = ats(v.mm, ~agree)
    print(f"  model, rule disagrees  {pct:6.2%}  (n={n})")

    print("  blend model/rule:")
    for w in [0.9, 0.8, 0.7, 0.5]:
        pct, n = ats(w * v.mm + (1 - w) * rule)
        print(f"    {w:.0%} model  {pct:6.2%}  (n={n})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=int, default=2010)
    ap.add_argument("--week1-only", action="store_true")
    ap.add_argument("--vs-model", action="store_true",
                    help="walk-forward comparison against the shipped ATS model")
    args = ap.parse_args()

    g = load()
    g = g[g.season >= args.since]
    if args.week1_only:
        g = g[g.week == 1]
    g = g.reset_index(drop=True)
    seasons = sorted(g.season.unique())
    print(f"{len(g)} games with win totals, {seasons[0]}-{seasons[-1]}\n")

    # --- how many points is one win of the total actually worth? ---------
    fit = np.polyfit(g.wt_diff, g.margin, 1)
    print(f"OLS margin ~ wt_diff:      slope {fit[0]:.2f} pts/win, intercept {fit[1]:.2f} (HFA)")
    fitv = np.polyfit(g.wt_diff, g.spread_line, 1)
    print(f"OLS vegas_line ~ wt_diff:  slope {fitv[0]:.2f} pts/win, intercept {fitv[1]:.2f} (HFA)")
    print(f"corr(wt_diff, margin) = {g.wt_diff.corr(g.margin):.3f}\n")

    print("--- RJ's rule as stated (x5.5, +1.75 home) ---")
    summarize("all seasons", g, grade(g, RJ_MULT, RJ_HFA), per_season=True)
    last7 = g[g.season >= max(seasons) - 6]
    summarize("last 7 seasons", last7, grade(last7, RJ_MULT, RJ_HFA))

    print("\n--- multiplier sweep (HFA 1.75) ---")
    best = None
    for mult in [round(x, 2) for x in np.arange(0.5, 6.01, 0.25)]:
        res = grade(g, mult, RJ_HFA).dropna()
        pct = res.mean()
        flag = " <-- best" if best is None or pct > best[1] else ""
        if best is None or pct > best[1]:
            best = (mult, pct)
        print(f"  x{mult:<5} {pct:6.2%}  (n={len(res)}){flag}")

    print("\n--- HFA sweep at the best multiplier ---")
    for hfa in [0.0, 1.0, 1.75, 2.5, 3.0]:
        res = grade(g, best[0], hfa).dropna()
        print(f"  +{hfa:<4} {res.mean():6.2%}  (n={len(res)})")

    print("\n--- edge filter (best multiplier, |rule - vegas| >= k) ---")
    pred = g.wt_diff * best[0] + RJ_HFA
    edge = (pred - g.spread_line).abs()
    res = grade(g, best[0], RJ_HFA)
    for k in [0, 1, 2, 3, 4, 6]:
        d = res[edge >= k].dropna()
        if len(d):
            roi = (d.sum() * (100 / 110) - (len(d) - d.sum())) / len(d)
            print(f"  >= {k}: {d.mean():6.2%}  (n={len(d)})  ROI {roi:+6.2%}")

    slices(g, 1.75)

    if args.vs_model:
        vs_model(g, best[0])

    print(f"\nbreakeven at -110: {BREAKEVEN:.2%}")


if __name__ == "__main__":
    main()
