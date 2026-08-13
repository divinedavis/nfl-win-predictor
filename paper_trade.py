"""Paper-trading ledger for player props. No real money — the point is to
find out, on live closing-ish lines, whether the projections clear the vig
BEFORE a dollar is ever staked.

    python paper_trade.py --pick     match newest lines vs projections,
                                     log every edge >= EDGE_MIN to the ledger
    python paper_trade.py --settle   score finished weeks, print the record

Decision rule (agreed before the season): bet nothing real unless the paper
record clears the -114 breakeven (53.3%) with at least ~200 settled picks.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from features import norm_name
from props import QUANTILES, STATS, prob_over

LEDGER = Path("paper_trades.csv")
LINES_DIR = Path("props_lines")
EDGE_MIN = 0.05          # model P(side) must beat implied by 5+ points
BREAKEVEN = 0.533        # win rate needed at typical -114 prop juice


def implied(odds: float) -> float:
    odds = float(odds)
    return -odds / (-odds + 100) if odds < 0 else 100 / (odds + 100)


def payout(odds: float) -> float:
    odds = float(odds)
    return 100 / -odds if odds < 0 else odds / 100


def pick() -> None:
    snaps = sorted(LINES_DIR.glob("*.csv"))
    if not snaps:
        raise SystemExit("No line snapshots in props_lines/ — run fetch_props.py")
    lines = pd.read_csv(snaps[-1])
    proj = pd.read_csv("props_projections.csv")
    proj["norm"] = proj["player"].map(norm_name)
    lines["norm"] = lines["player"].map(norm_name)

    ledger = pd.read_csv(LEDGER) if LEDGER.exists() else pd.DataFrame()
    seen = (set(zip(ledger.week, ledger.stat, ledger.norm, ledger.side))
            if len(ledger) else set())

    rows = []
    for r in lines.itertuples(index=False):
        m = proj[(proj.norm == r.norm) & (proj.stat == r.stat)]
        if m.empty or pd.isna(r.line) or pd.isna(r.odds):
            continue
        m = m.iloc[0]
        quants = [m.p10, m.p25, m.p50, m.p75, m.p90]
        p_over = prob_over(quants, float(r.line))
        p_side = p_over if r.side == "Over" else 1 - p_over
        edge = p_side - implied(r.odds)
        key = (int(m.week), r.stat, r.norm, r.side)
        if edge >= EDGE_MIN and key not in seen:
            seen.add(key)
            rows.append({
                "picked": date.today().isoformat(),
                "season": int(m.season), "week": int(m.week),
                "stat": r.stat, "player": m.player, "norm": r.norm,
                "team": m.team, "opp": m.opp, "side": r.side,
                "line": float(r.line), "odds": float(r.odds),
                "model_p": round(p_side, 3),
                "implied": round(implied(r.odds), 3),
                "edge": round(edge, 3),
                "p50": m.p50, "won": "",
            })
    if rows:
        out = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        out.to_csv(LEDGER, index=False)
    print(f"{len(rows)} new paper picks (edge >= {EDGE_MIN:.0%}) from "
          f"{snaps[-1].name}; ledger now "
          f"{len(ledger) + len(rows)} picks")


def settle() -> None:
    import nflreadpy as nfl

    if not LEDGER.exists():
        raise SystemExit("No ledger yet — run --pick first")
    led = pd.read_csv(LEDGER)
    open_mask = led["won"].isna() | (led["won"] == "")
    if not open_mask.any():
        print("Nothing to settle.")
    else:
        seasons = sorted(led.loc[open_mask, "season"].unique())
        ps = nfl.load_player_stats([int(s) for s in seasons],
                                   summary_level="week").to_pandas()
        ps["norm"] = ps["player_display_name"].map(norm_name)
        for i in led.index[open_mask]:
            r = led.loc[i]
            actual = ps[(ps.season == r.season) & (ps.week == r.week)
                        & (ps.norm == r.norm)]
            if actual.empty:
                continue  # game not played yet (or player inactive: leave open)
            val = float(actual[r.stat].fillna(0).iloc[0])
            if val == r.line:
                led.loc[i, "won"] = "push"
            else:
                over_hit = val > r.line
                led.loc[i, "won"] = str(over_hit if r.side == "Over"
                                        else not over_hit)
            led.loc[i, "actual"] = val
        led.to_csv(LEDGER, index=False)

    done = led[led["won"].isin(["True", "False"])]
    if done.empty:
        print("No settled picks yet.")
        return
    won = done["won"] == "True"
    pnl = np.where(won, [payout(o) for o in done["odds"]], -1.0)
    print(f"\n=== Paper record: {won.sum()}-{(~won).sum()} "
          f"({won.mean():.1%}) over {len(done)} picks ===")
    print(f"breakeven at typical juice: {BREAKEVEN:.1%}  "
          f"({'ABOVE' if won.mean() > BREAKEVEN else 'below'})")
    print(f"flat-stake ROI: {pnl.mean():+.2%}   "
          f"(needs ~200+ picks before it means anything)")
    by_stat = done.assign(w=won).groupby("stat")["w"].agg(["count", "mean"])
    print(by_stat.to_string(float_format=lambda x: f"{x:.1%}"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--pick", action="store_true")
    g.add_argument("--settle", action="store_true")
    args = ap.parse_args()
    pick() if args.pick else settle()
