"""Does the world model actually describe football? Check it against real games.

A simulator that produces plausible-looking numbers is easy; one whose numbers
are *right* is the only kind worth blending into a prediction. This runs the
world model over completed games it was never fitted on and asks four questions:

1. Does it score like the NFL? Simulated points, margin spread and drive
   outcomes are compared against what those seasons actually produced.
2. Is the distribution honest? For each game the actual margin is located inside
   the simulated margin distribution. Done right, those percentile positions are
   uniform -- results land in the tails exactly as often as the tails claim.
   Piled up in the middle means the simulator is too uncertain; piled up in the
   tails means it is overconfident.
3. Do the intervals cover? The real margin should fall inside the central 50%
   of simulations 50% of the time, and inside the central 80% 80% of the time.
4. Is the win probability any good? Accuracy, Brier and log loss against the
   Elo + XGBoost blend already in production, plus a sweep of how much weight
   the simulator deserves in that blend.

    python sim_backtest.py --model world_model_2023.pkl --seasons 2024 2025
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURES
from simulate import Simulator, game_seed
from train import PARAMS, ELO_BLEND

CACHE = "sim_backtest.csv"


def baseline_probs(done: pd.DataFrame, seasons: list[int]) -> pd.Series:
    """The production probability, refit walk-forward so nothing leaks."""
    out = pd.Series(index=done.index, dtype=float)
    for season in seasons:
        train = done[done.season < season]
        test = done[done.season == season]
        if test.empty:
            continue
        model = XGBClassifier(**PARAMS)
        model.fit(train[FEATURES], train["home_win"])
        raw = model.predict_proba(test[FEATURES])[:, 1]
        out.loc[test.index] = (1 - ELO_BLEND) * raw + ELO_BLEND * test["elo_prob"].values
    return out


def run_sims(games: pd.DataFrame, model_path: str, n: int,
             margins_path: str | None = None) -> pd.DataFrame:
    sim = Simulator.load(model_path)
    rows, margins = [], {}
    for i, (_, g) in enumerate(games.iterrows(), 1):
        out = sim.simulate_game(g, n=n, seed=game_seed(g.game_id))
        margin = out["margin"]
        actual = g.home_score - g.away_score
        # Randomised PIT: where the real margin sits in the simulated
        # distribution, with ties broken uniformly so a discrete distribution
        # still gives a uniform score when the model is right.
        below = float((margin < actual).mean())
        equal = float((margin == actual).mean())
        rows.append({
            "game_id": g.game_id, "season": g.season, "week": g.week,
            "home_team": g.home_team, "away_team": g.away_team,
            "actual_margin": actual, "actual_total": g.home_score + g.away_score,
            "home_win": g.home_win,
            "sim_wp": out["home_win_prob"],
            "sim_margin": margin.mean(), "sim_margin_sd": margin.std(),
            "sim_total": out["total"].mean(),
            "sim_home_pts": out["home_score"].mean(),
            "sim_away_pts": out["away_score"].mean(),
            "pit": below + np.random.random() * equal,
            "spread_line": g.get("spread_line", np.nan),
            # Cover probability read straight off the simulated distribution,
            # rather than from a normal fitted around a point estimate.
            "sim_cover_prob": float((margin > g.get("spread_line", np.nan)).mean())
            if pd.notna(g.get("spread_line", np.nan)) else np.nan,
            "q10": np.quantile(margin, 0.10), "q25": np.quantile(margin, 0.25),
            "q75": np.quantile(margin, 0.75), "q90": np.quantile(margin, 0.90),
        })
        if margins_path:
            margins[g.game_id] = margin.astype("int16")
        if i % 20 == 0:
            print(f"  {i}/{len(games)} games", file=sys.stderr)
    if margins_path:
        # Keeping the full simulated margins makes every later distribution
        # question answerable without re-simulating for half an hour.
        np.savez_compressed(margins_path, **margins)
        print(f"wrote {margins_path}", file=sys.stderr)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, base: pd.Series | None) -> None:
    n = len(df)
    print(f"\n=== World model vs {n} real games "
          f"({df.season.min()}-{df.season.max()}) ===")

    print("\n1. Does it score like the NFL?")
    print(f"   points per game     sim {df.sim_total.mean():6.1f}   "
          f"actual {df.actual_total.mean():6.1f}")
    print(f"   home margin         sim {df.sim_margin.mean():+6.2f}   "
          f"actual {df.actual_margin.mean():+6.2f}")
    print(f"   margin spread (sd)  sim {df.sim_margin.std():6.2f} across games, "
          f"{df.sim_margin_sd.mean():5.2f} within a game")
    print(f"                       actual {df.actual_margin.std():6.2f}")

    print("\n2. Is the distribution honest? (percentile of the real result)")
    edges = np.linspace(0, 1, 11)
    counts, _ = np.histogram(df.pit, bins=edges)
    for lo, c in zip(edges[:-1], counts):
        bar = "#" * int(round(c / n * 200))
        print(f"   {lo:.1f}-{lo+0.1:.1f} {c/n:6.1%} {bar}")
    # Kolmogorov-Smirnov against uniform: how far the worst point is off.
    p = np.sort(df.pit.to_numpy())
    ks = np.abs(p - np.arange(1, n + 1) / n).max()
    print(f"   KS distance from uniform {ks:.3f}  "
          f"(5% critical value {1.36 / np.sqrt(n):.3f})")

    print("\n3. Do the intervals cover?")
    c50 = ((df.actual_margin >= df.q25) & (df.actual_margin <= df.q75)).mean()
    c80 = ((df.actual_margin >= df.q10) & (df.actual_margin <= df.q90)).mean()
    print(f"   central 50% covers {c50:.1%}  (target 50%)")
    print(f"   central 80% covers {c80:.1%}  (target 80%)")

    ats = df[df.spread_line.notna() & (df.spread_line != 0)
             & (df.actual_margin != df.spread_line)].copy()
    if len(ats):
        print("\n4. Against the spread -- does the simulated distribution beat "
              "a fitted normal?")
        covered = (ats.actual_margin > ats.spread_line).astype(int)
        # The production approach: one expected margin, one league-wide sigma.
        sigma = 13.9
        from scipy.stats import norm
        normal_p = 1 - norm.cdf(ats.spread_line, loc=ats.sim_margin, scale=sigma)
        for name, p in (("simulated distribution", ats.sim_cover_prob.to_numpy()),
                        ("normal(sim margin, 13.9)", normal_p)):
            pick = p > 0.5
            hit = (pick == covered.to_numpy()).mean()
            print(f"   {name:26s} ATS {hit:.4f}  "
                  f"brier {brier_score_loss(covered, np.clip(p, 1e-6, 1-1e-6)):.4f}")
        print(f"   ({len(ats)} games with a line; break-even 52.4%)")

    print("\n5. Is the win probability any good?")
    y = df.home_win.to_numpy().astype(int)
    print(f"   {'simulator':16s} acc {accuracy_score(y, df.sim_wp > 0.5):.4f}  "
          f"brier {brier_score_loss(y, df.sim_wp):.4f}  "
          f"logloss {log_loss(y, df.sim_wp.clip(1e-6, 1 - 1e-6)):.4f}")
    if base is not None:
        b = base.to_numpy()
        print(f"   {'elo + xgboost':16s} acc {accuracy_score(y, b > 0.5):.4f}  "
              f"brier {brier_score_loss(y, b):.4f}  logloss {log_loss(y, b):.4f}")
        print("\n   blend sweep (w = weight on the simulator)")
        best = None
        for w in np.arange(0, 1.01, 0.1):
            p = (1 - w) * b + w * df.sim_wp.to_numpy()
            br = brier_score_loss(y, p)
            acc = accuracy_score(y, p > 0.5)
            flag = ""
            if best is None or br < best[1]:
                best, flag = (w, br), ""
            print(f"     w={w:.1f}  acc {acc:.4f}  brier {br:.4f}{flag}")
        print(f"   best Brier at w={best[0]:.1f} ({best[1]:.4f})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="world_model_2023.pkl")
    ap.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    ap.add_argument("--n", type=int, default=2000, help="simulations per game")
    ap.add_argument("--limit", type=int, default=None, help="sample this many games")
    ap.add_argument("--cache", default=CACHE)
    ap.add_argument("--reuse", action="store_true", help="skip simulating, read the cache")
    ap.add_argument("--save-margins", default=None,
                    help="also store every game's simulated margins (.npz)")
    args = ap.parse_args()

    feat = pd.read_parquet("features.parquet")
    done = feat[feat.home_win.notna()].copy()
    games = done[done.season.isin(args.seasons)]
    if args.limit:
        games = games.sample(args.limit, random_state=42).sort_values(["season", "week"])

    if args.reuse:
        df = pd.read_csv(args.cache)
    else:
        print(f"simulating {len(games)} games at n={args.n}...", file=sys.stderr)
        df = run_sims(games, args.model, args.n, args.save_margins)
        df.to_csv(args.cache, index=False)
        print(f"wrote {args.cache}", file=sys.stderr)

    base = baseline_probs(done, sorted(args.seasons))
    base = base.loc[games.index].reset_index(drop=True)
    df = df.reset_index(drop=True)
    report(df, base)


if __name__ == "__main__":
    main()
