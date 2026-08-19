"""Is the live win probability any good? Replay real in-game states and check.

live_wp.py can price a game from any state. That is only worth shipping if the
numbers are right, and there is a strong yardstick available: nflfastR ships its
own win-probability model, fitted on the same play-by-play, and every historical
play already carries its answer in `home_wp`.

So this takes states out of real games the world model never trained on, prices
each one by simulating the rest of the game, and scores both against what
actually happened. nflfastR's model is a direct classifier over the game state;
ours reaches its answer by playing the remaining snaps out thousands of times.
If the simulator lands close, the live number is trustworthy.

Note what is *not* claimed: beating nflfastR is not the goal, and its model has
had years of tuning. Matching it is the bar, because matching it means the
simulator's in-game dynamics are sound.

    python ingame_backtest.py --model world_model_2023.pkl --season 2025
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from simulate import Simulator, game_seed

# Only states where the simulator has something to decide: a real snap, with a
# down and a clock, before the game is effectively over.
NEEDED = ["game_id", "play_id", "posteam", "home_team", "away_team", "down",
          "ydstogo", "yardline_100", "half_seconds_remaining", "qtr",
          "total_home_score", "total_away_score", "home_wp", "vegas_home_wp",
          "play_type",
          "posteam_timeouts_remaining", "defteam_timeouts_remaining"]


def sample_states(season: int, n_states: int, seed: int) -> pd.DataFrame:
    pbp = nfl.load_pbp([season]).to_pandas()
    df = pbp[[c for c in NEEDED if c in pbp.columns]].copy()
    df = df[df.play_type.isin(["run", "pass", "punt", "field_goal"])
            & df.down.notna() & df.home_wp.notna() & df.vegas_home_wp.notna()
            & df.yardline_100.notna() & df.half_seconds_remaining.notna()]
    # Regulation only: overtime states are rare and the comparison there is
    # dominated by the coin toss rather than by the dynamics.
    df = df[df.qtr <= 4]
    return df.sample(min(n_states, len(df)), random_state=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="world_model_2023.pkl")
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--states", type=int, default=800)
    ap.add_argument("--n", type=int, default=400, help="simulations per state")
    ap.add_argument("--out", default="ingame_backtest.csv")
    args = ap.parse_args()

    feat = pd.read_parquet("features.parquet")
    done = feat[feat.home_win.notna()]
    outcome = dict(zip(done.game_id, done.home_win))
    rows_by_game = {g.game_id: g for _, g in done.iterrows()}

    states = sample_states(args.season, args.states, seed=42)
    sim = Simulator.load(args.model)

    out = []
    for i, (_, p) in enumerate(states.iterrows(), 1):
        row = rows_by_game.get(p.game_id)
        if row is None or p.game_id not in outcome:
            continue
        half = 1 if p.qtr <= 2 else 2
        res = sim.simulate_from(
            row,
            pos_is_home=(p.posteam == p.home_team),
            yardline_100=float(p.yardline_100),
            down=int(p.down), ydstogo=float(p.ydstogo),
            half_seconds_remaining=float(p.half_seconds_remaining),
            half=half,
            home_score=int(p.total_home_score), away_score=int(p.total_away_score),
            home_timeouts=int(p.get("posteam_timeouts_remaining", 3) or 3)
            if p.posteam == p.home_team else 3,
            away_timeouts=3,
            n=args.n, seed=game_seed(f"{p.game_id}:{p.play_id}"),
        )
        out.append({
            "game_id": p.game_id, "qtr": int(p.qtr),
            "half_seconds_remaining": float(p.half_seconds_remaining),
            "score_diff": int(p.total_home_score - p.total_away_score),
            "play_id": p.play_id,
            "sim_wp": res["home_win_prob"], "nflfastr_wp": float(p.home_wp),
            "vegas_wp": float(p.vegas_home_wp),
            "home_win": int(outcome[p.game_id]),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(states)} states", file=sys.stderr)

    df = pd.DataFrame(out)
    df.to_csv(args.out, index=False)
    y = df.home_win.to_numpy()

    print(f"\n=== In-game win probability, {len(df)} states from {args.season} ===")
    print(f"   {'':16s} {'acc':>7} {'brier':>8} {'logloss':>9}")
    for name, p in (("world model", df.sim_wp),
                    ("nflfastR (state)", df.nflfastr_wp),
                    ("nflfastR + Vegas", df.vegas_wp)):
        pc = p.clip(1e-6, 1 - 1e-6)
        print(f"   {name:16s} {accuracy_score(y, p > 0.5):7.4f} "
              f"{brier_score_loss(y, p):8.4f} {log_loss(y, pc):9.4f}")
    print(f"\n   correlation with nflfastR state model: {df.sim_wp.corr(df.nflfastr_wp):.4f}")
    print(f"   correlation with the Vegas-aware model: {df.sim_wp.corr(df.vegas_wp):.4f}")
    print(f"   mean |difference| vs Vegas-aware:       "
          f"{(df.sim_wp - df.vegas_wp).abs().mean():.4f}")
    print("\n   Note: nflfastR's state model does not know who is playing; the "
          "Vegas-aware\n   one does, and is the like-for-like benchmark since "
          "the world model\n   carries Elo, EPA, quarterback rating and injuries.")

    print("\n   by quarter (mean |difference| vs nflfastR, and each model's Brier)")
    for q, grp in df.groupby("qtr"):
        yq = grp.home_win.to_numpy()
        print(f"     Q{q}  n={len(grp):4d}"
              f"   world {brier_score_loss(yq, grp.sim_wp):.4f}"
              f"   state {brier_score_loss(yq, grp.nflfastr_wp):.4f}"
              f"   vegas {brier_score_loss(yq, grp.vegas_wp):.4f}")

    print("\n   calibration of the world model (predicted vs actual)")
    bins = np.arange(0, 1.01, 0.1)
    df["bucket"] = np.clip(np.digitize(df.sim_wp, bins[1:-1]), 0, 9)
    for b, grp in df.groupby("bucket"):
        print(f"     {bins[b]:.1f}-{bins[b]+0.1:.1f}  n={len(grp):4d}  "
              f"predicted {grp.sim_wp.mean():.3f}  actual {grp.home_win.mean():.3f}")


if __name__ == "__main__":
    main()
