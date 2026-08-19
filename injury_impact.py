"""What an injury costs -- simulated, and then checked against what really happened.

READ THE VERDICT COLUMN. This tool was built on the theory that spreading an
absence across every snap would surface value the game-level model absorbs into
Elo. Measured on 2026-08-19, that theory is WRONG, and wrong in a way worth
keeping the tool around to demonstrate:

    group   simulated        actually happened      n
    DL      opp +2.50 pts    opp +0.32 pts        108
    LB      opp +1.96 pts    opp -0.13 pts        235
    TE      own +1.12 pts    own +0.33 pts         66

Compounding cuts both ways. A small per-play error multiplied by 130 plays
becomes a large fake number, so the simulator overstates a defensive lineman by
roughly eight times. The repo's existing finding stands: one defensive starter
out is worth close to nothing, and position_impact.py was right to read it as
about zero.

Two separate problems produce the bad rows:

  * thin evidence -- teams rotate running backs and tight ends, so a genuinely
    full-time one is rare (21 and 66 games in eleven seasons against 250 for
    quarterbacks). The heads learn noise and the simulator reports it with a
    straight face. Watch the n column.
  * amplification -- even where evidence is decent, a per-play bias compounds
    over a full game into points that are not there.

So every simulated number here is printed next to the real-world effect for the
same absence, with the sample size behind it. Trust the simulated figure only
where the two agree and n is large. Do not put these numbers on the site.

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


def observed_effect(feat: pd.DataFrame, grp: str) -> tuple[float, int]:
    """What really happened, from the games themselves.

    An offensive absence is measured against that offence's own scoring average
    for the season; a defensive one against how much that defence normally
    allows. Comparing a team to itself cancels team quality, which is the whole
    confound.
    """
    defensive = grp in DEF_GROUPS
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        pts = feat[f"{opp}_score"] if defensive else feat[f"{side}_score"]
        rows.append(pd.DataFrame({
            "team": feat[f"{side}_team"].values, "season": feat["season"].values,
            "pts": pts.values, "wt": feat[f"{side}_{grp}_out_wt"].values,
        }))
    t = pd.concat(rows, ignore_index=True)
    t["base"] = t.groupby(["team", "season"]).pts.transform("mean")
    t["diff"] = t.pts - t.base
    out = t[(t.wt >= 0.85) & (t.wt < 1.15)]
    healthy = t[t.wt < 0.05]
    if len(out) < 10:
        return float("nan"), len(out)
    return float(out["diff"].mean() - healthy["diff"].mean()), len(out)


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
    print(f"  {'grp':>3}  {'win prob':>9}  {'simulated':>10}  {'REALLY':>9}  "
          f"{'n':>5}  {'verdict':>12}")
    print(f"  {'':>3}  {'':>9}  {'pts':>10}  {'pts':>9}  {'':>5}")
    for grp in args.group:
        absence = typical_absence(done, grp)
        r = measure(sim, games, args.side, grp, absence, args.n)
        real, n = observed_effect(done, grp)
        # The simulated figure to compare is whichever side of the ball the
        # group actually acts on.
        simulated = r["d_opp_points"] if grp in DEF_GROUPS else r["d_own_points"]
        if n < 100:
            verdict = "TOO FEW"
        elif not np.isfinite(real) or abs(real) < 1e-9:
            verdict = "no real effect"
        elif np.sign(simulated) != np.sign(real):
            verdict = "WRONG SIGN"
        elif abs(simulated) > 2.5 * abs(real):
            verdict = f"{abs(simulated / real):.0f}x TOO BIG"
        else:
            verdict = "agrees"
        print(f"  {grp.upper():>3}  {r['d_win_prob']:>+8.1%}  {simulated:>+10.2f}  "
              f"{real:>+9.2f}  {n:>5}  {verdict:>12}")

    print("\n  'REALLY' is what happened in the actual games: a team's scoring "
          "when that\n  group had a starter out, against its own average for "
          "the season, so team\n  quality cancels. Defensive groups are scored "
          "on what the opponent did.\n"
          "\n  Trust a simulated number only where the verdict says it agrees "
          "and n is\n  large. A small per-play error compounds over 130 plays "
          "into points that\n  are not there -- see this module's docstring. "
          "Do not put these on the site.")


if __name__ == "__main__":
    main()
