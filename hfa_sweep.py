"""How much is home field actually worth? Sweep the number and measure.

The diary turned up a real problem: across 3,018 games the model expected 1,717
home wins and there were 1,659 -- further off than luck accounts for, and the
same error shows on both sides of the split, over-rating home favourites and
under-rating road favourites by about the same amount. That is the signature of
one number being wrong rather than a modelling gap.

The number is ELO_HFA in features.py, fixed at 52 Elo points since the project
started. Home field has not been worth a constant amount over that period --
home teams won 59.6% of games in 2018 and 50.0% in 2020, when nobody was in the
stands -- so a constant fitted to the old league is now too generous.

This rebuilds the features at each candidate value and scores the result the
same way train.py does: train on the seasons before, predict the season, never
look ahead. Three things are reported, and a value has to win on more than one
of them to be worth adopting:

    right      how often the pick wins
    honesty    how close its stated confidence is to how often it is right
    lean       predicted home wins against actual, across every season

    python hfa_sweep.py --values 30 35 40 45 52
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from train import PARAMS, ELO_BLEND, BACKTEST_SEASONS

PYTHON = ".venv/bin/python"


def rebuild(hfa: float) -> pd.DataFrame:
    env = dict(os.environ, ELO_HFA=str(hfa))
    r = subprocess.run([PYTHON, "features.py"], env=env,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"features.py failed at HFA={hfa}:\n{r.stderr[-800:]}")
    return pd.read_parquet("features.parquet")


def score(feat: pd.DataFrame) -> dict:
    from features import FEATURES
    done = feat[feat.home_win.notna()].copy()
    probs, actual, seasons = [], [], []
    for season in BACKTEST_SEASONS:
        tr, te = done[done.season < season], done[done.season == season]
        if te.empty or tr.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(tr[FEATURES], tr["home_win"])
        p = ((1 - ELO_BLEND) * m.predict_proba(te[FEATURES])[:, 1]
             + ELO_BLEND * te["elo_prob"].to_numpy())
        probs.append(p)
        actual.append(te["home_win"].to_numpy())
        seasons.append(np.full(len(te), season))

    p = np.concatenate(probs)
    y = np.concatenate(actual)
    s = np.concatenate(seasons)

    # Honesty: how often it said it would be right, against how often it was.
    claim = np.maximum(p, 1 - p)
    right = ((p > 0.5) == (y == 1)).astype(float)
    # Lean: predicted home wins against actual, in units of normal swing.
    lean = (y.sum() - p.sum()) / np.sqrt((p * (1 - p)).sum())
    # How many individual seasons lean the same (wrong) way.
    per_season = [(y[s == k].mean() - p[s == k].mean()) for k in np.unique(s)]

    return {
        "right": accuracy_score(y, p > 0.5),
        "honesty": right.mean() - claim.mean(),
        "brier": brier_score_loss(y, p),
        "lean": lean,
        "seasons_biased": int(sum(1 for d in per_season if d < -0.01)),
        "n_seasons": len(per_season),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--values", type=float, nargs="+",
                    default=[25, 30, 35, 40, 45, 52])
    ap.add_argument("--restore", type=float, default=None,
                    help="rebuild with this value at the end")
    args = ap.parse_args()

    rows = []
    for hfa in args.values:
        print(f"building at HFA={hfa:g} ...", file=sys.stderr)
        r = score(rebuild(hfa))
        r["hfa"] = hfa
        rows.append(r)
        print(f"  right {r['right']:.4f}  honesty {r['honesty']:+.4f}  "
              f"lean {r['lean']:+.2f}", file=sys.stderr)

    df = pd.DataFrame(rows)
    print("\n=== What is home field worth? ===")
    print("   right    = how often the pick wins")
    print("   honesty  = how often it is right minus how often it claims "
          "(0 is perfect, minus means it brags)")
    print("   lean     = predicted home wins vs actual, in normal swings "
          "(0 is perfect, minus means it favours home too much)")
    print("   biased   = seasons where it still over-favours home by 1%+\n")
    print(f"   {'HFA':>5} {'right':>8} {'honesty':>9} {'brier':>8} {'lean':>7} "
          f"{'biased':>8}")
    for _, r in df.iterrows():
        mark = "  <- current" if r.hfa == 52 else ""
        print(f"   {r.hfa:5.0f} {r['right']:8.4f} {r['honesty']:+9.4f} "
              f"{r['brier']:8.4f} {r['lean']:+7.2f} "
              f"{int(r['seasons_biased']):3d}/{int(r['n_seasons']):<4d}{mark}")

    best_lean = df.iloc[(df.lean.abs()).argmin()]
    best_brier = df.iloc[df.brier.argmin()]
    print(f"\n   straightest lean at HFA={best_lean.hfa:.0f} "
          f"({best_lean.lean:+.2f})")
    print(f"   most honest percentages at HFA={best_brier.hfa:.0f} "
          f"(brier {best_brier.brier:.4f})")

    if args.restore is not None:
        print(f"\nrebuilding features at HFA={args.restore:g}", file=sys.stderr)
        rebuild(args.restore)


if __name__ == "__main__":
    main()
