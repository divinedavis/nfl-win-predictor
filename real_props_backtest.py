"""THE honest props test: walk-forward projections vs real DraftKings closing
lines archived by fetch_props_history.py. For each season with archived lines,
models train only on prior seasons, price every Over/Under at the book's
actual line and odds, bet flat $1 wherever model probability beats the vigged
implied probability by the edge threshold, and settle against what happened.

    python real_props_backtest.py [--seasons 2025 2024] [--edges 0.03 0.05 0.08]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from features import norm_name
from paper_trade import implied, payout
from props import (STATS, load_player_weeks, game_context, build_stat_table,
                   fit_quantiles, predict_quantiles, prob_over)

LINES_DIR = Path("props_lines_historical")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="*", default=None)
    ap.add_argument("--edges", type=float, nargs="*",
                    default=[0.03, 0.05, 0.08])
    args = ap.parse_args()

    files = sorted(LINES_DIR.glob("*.csv"))
    if args.seasons:
        files = [f for f in files if int(f.stem) in args.seasons]
    if not files:
        raise SystemExit("No archived line files — run fetch_props_history.py")
    lines = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    lines = lines[lines["game_type"] == "REG"].dropna(subset=["line", "odds"])
    lines["norm"] = lines["player"].map(norm_name)
    print(f"{len(lines)} archived outcomes across seasons "
          f"{sorted(lines.season.unique())}")

    ps = load_player_weeks()
    ctx = game_context()

    bets = []
    for stat in STATS:
        table = build_stat_table(ps, ctx, stat)
        table = table[table["played"]]
        table["norm"] = table["player_display_name"].map(norm_name)
        stat_lines = lines[lines["stat"] == stat]
        for season in sorted(stat_lines.season.unique()):
            train = table[table.season < season]
            test = table[table.season == season].copy()
            if test.empty:
                continue
            models = fit_quantiles(train)
            quants = predict_quantiles(models, test)
            test = test.reset_index(drop=True)
            # (week, norm, team) -> row index; team guards name collisions
            idx = {(r.week, r.norm, r.team): i
                   for i, r in enumerate(test.itertuples(index=False))}
            for ln in stat_lines[stat_lines.season == season].itertuples(index=False):
                i = idx.get((ln.week, ln.norm, ln.home)) \
                    or idx.get((ln.week, ln.norm, ln.away))
                if i is None:
                    continue
                p_over = prob_over(quants[i], float(ln.line))
                p_side = p_over if ln.side == "Over" else 1 - p_over
                imp = implied(ln.odds)
                actual = float(test.loc[i, "y"])
                if actual == ln.line:
                    result = "push"
                else:
                    over_hit = actual > ln.line
                    result = str(over_hit if ln.side == "Over" else not over_hit)
                bets.append({
                    "season": season, "week": ln.week, "stat": stat,
                    "side": ln.side, "line": ln.line, "odds": ln.odds,
                    "p_side": p_side, "implied": imp,
                    "edge": p_side - imp, "result": result,
                })

    d = pd.DataFrame(bets)
    matched = len(d) / max(len(lines), 1)
    print(f"priced {len(d)} outcomes ({matched:.0%} of archived lines matched "
          f"to a modeled player)\n")

    print(f"{'edge>=':>7s} {'bets':>6s} {'win%':>6s} {'ROI':>8s}   "
          f"(breakeven at avg juice shown per row)")
    for edge in args.edges:
        b = d[(d.edge >= edge) & (d.result != "push")]
        if b.empty:
            continue
        won = b["result"] == "True"
        pnl = np.where(won, [payout(o) for o in b["odds"]], -1.0)
        be = b["implied"].mean()
        print(f"{edge:>7.2f} {len(b):6d} {100*won.mean():5.1f}% "
              f"{100*pnl.mean():+7.2f}%   (breakeven {100*be:.1f}%)")
    print()
    b = d[(d.edge >= 0.05) & (d.result != "push")]
    if len(b):
        won = b["result"] == "True"
        by = b.assign(w=won).groupby("stat").agg(
            n=("w", "size"), win=("w", "mean"),
            roi=("odds", lambda o: np.mean(np.where(
                b.loc[o.index, "result"] == "True",
                [payout(x) for x in o], -1.0))))
        print("edge>=0.05 by stat:")
        print(by.to_string(float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
