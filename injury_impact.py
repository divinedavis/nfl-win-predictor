"""What an injury costs, measured drive by drive instead of in one number.

position_impact.py asks the game-level model the same question and gets back a
single win-probability delta, because that model sees an absence once. The world
model sees it on every snap, so the answer comes with a mechanism attached:

    DL   -1.4% win prob   sacks 7.3% -> 6.6%   opponent +0.19 yds/play   +0.8 pts

That is the case for simulating at all. A pass rusher's value is not one number
applied once; it is a slightly worse outcome on sixty dropbacks, compounded
through drive structure into points. A game-level classifier has to price that
in a single coefficient and, as this repo has measured before, Elo tends to
swallow it -- single defensive outs read as roughly zero there.

The counterfactual matches position_impact.py so the two are comparable: one
full-time player ruled Out (weighted outs 1.0) against a fully healthy group.
The player's rating is set to what a real full-time absence at that group
actually looks like in the data, not an invented constant.

    python injury_impact.py                       # every group, sampled games
    python injury_impact.py --group dl --games 40
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from features import POS_GROUPS, OFF_GROUPS, DEF_GROUPS
from simulate import Simulator, game_seed

# Groups the world model can actually respond to, and the columns that carry
# them. OL has no public per-player stats, so it moves on snap weight alone.
GROUP_COLUMNS = {
    **{g: [f"{{side}}_{g}_out_wt", f"{{side}}_{g}_out_epa"] for g in OFF_GROUPS},
    **{g: [f"{{side}}_{g}_out_wt", f"{{side}}_{g}_out_val"] for g in DEF_GROUPS},
    "ol": ["{side}_ol_out_wt"],
}


def typical_absence(feat: pd.DataFrame, grp: str) -> dict[str, float]:
    """What one full-time absence at this group looks like in the real data.

    Rows where the weighted outs sit near 1.0 are, by construction, roughly one
    full-time player missing; the median rating on those rows is the honest
    magnitude to test with.
    """
    out = {}
    for tmpl in GROUP_COLUMNS[grp]:
        col = tmpl.format(side="home")
        if col.endswith("_out_wt"):
            out[tmpl] = 1.0
            continue
        wt = feat[f"home_{grp}_out_wt"]
        near = feat[(wt > 0.85) & (wt < 1.15)]
        vals = near[col]
        vals = vals[vals > 0]
        out[tmpl] = float(vals.median()) if len(vals) >= 20 else 1.0
    return out


def _rates(sim: Simulator, n_games: int) -> dict:
    """Per-play rates from the simulator's own trace."""
    t = sim.trace
    ev = t["events"]
    passes = sum(ev.values()) or 1
    scrim = sum(t["yards"][k][1] for k in ("run", "pass") if k in t["yards"])
    yards = sum(t["yards"][k][0] for k in ("run", "pass") if k in t["yards"])
    return {
        "sack_rate": ev.get("sack", 0) / passes,
        "int_rate": ev.get("interception", 0) / passes,
        "yds_per_play": yards / max(scrim, 1),
    }


def measure(sim: Simulator, games: pd.DataFrame, side: str, grp: str,
            absence: dict[str, float], n: int) -> dict:
    """Simulate each game healthy and hurt, and difference the two."""
    healthy, hurt = [], []
    for _, g in games.iterrows():
        base = g.copy()
        for tmpl in GROUP_COLUMNS[grp]:
            base[tmpl.format(side=side)] = 0.0
        seed = game_seed(g.game_id)
        o = sim.simulate_game(base, n=n, seed=seed)
        healthy.append((o, _rates(sim, 1)))

        alt = base.copy()
        for tmpl, val in absence.items():
            alt[tmpl.format(side=side)] = val
        o = sim.simulate_game(alt, n=n, seed=seed)
        hurt.append((o, _rates(sim, 1)))

    def mean(rows, key, fn):
        return float(np.mean([fn(o) for o, _ in rows])) if key is None else \
            float(np.mean([r[key] for _, r in rows]))

    own = "home_score" if side == "home" else "away_score"
    opp = "away_score" if side == "home" else "home_score"
    wp = lambda o: o["home_win_prob"] if side == "home" else 1 - o["home_win_prob"]
    return {
        "d_win_prob": mean(hurt, None, wp) - mean(healthy, None, wp),
        "d_own_points": mean(hurt, None, lambda o: o[own].mean())
        - mean(healthy, None, lambda o: o[own].mean()),
        "d_opp_points": mean(hurt, None, lambda o: o[opp].mean())
        - mean(healthy, None, lambda o: o[opp].mean()),
        "sack_healthy": mean(healthy, "sack_rate", None),
        "sack_hurt": mean(hurt, "sack_rate", None),
        "d_yds_per_play": mean(hurt, "yds_per_play", None)
        - mean(healthy, "yds_per_play", None),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="world_model.pkl")
    ap.add_argument("--group", nargs="+", default=POS_GROUPS)
    ap.add_argument("--side", default="home", choices=["home", "away"])
    ap.add_argument("--games", type=int, default=24)
    ap.add_argument("--n", type=int, default=800)
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    args = ap.parse_args()

    feat = pd.read_parquet("features.parquet")
    done = feat[feat.home_win.notna()]
    games = done[done.season.isin(args.seasons)].sample(
        min(args.games, len(done)), random_state=42)
    sim = Simulator.load(args.model)

    print(f"Cost of ONE full-time player ruled Out, simulated over "
          f"{len(games)} games at n={args.n}\n")
    print(f"  {'grp':>3}  {'win prob':>9}  {'own pts':>8}  {'opp pts':>8}  "
          f"{'opp sack%':>18}  {'opp yds/play':>12}")
    rows = []
    for grp in args.group:
        absence = typical_absence(done, grp)
        r = measure(sim, games, args.side, grp, absence, args.n)
        rows.append((grp, r))
        # A defensive absence is felt by the opponent's offence, so the sack and
        # yardage columns describe whichever offence the group faces.
        sk = f"{r['sack_healthy']:.1%} -> {r['sack_hurt']:.1%}"
        print(f"  {grp.upper():>3}  {r['d_win_prob']:>+8.1%}  "
              f"{r['d_own_points']:>+8.2f}  {r['d_opp_points']:>+8.2f}  "
              f"{sk:>18}  {r['d_yds_per_play']:>+12.3f}")

    print("\nOffensive groups are measured on their own offence; defensive "
          "groups\nshow the effect on the offence they face. Ratings come from "
          "what a real\nfull-time absence at that group looks like in the data, "
          "not a fixed constant.")


if __name__ == "__main__":
    main()
