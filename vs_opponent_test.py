"""Does "this player historically does well against that team" help pick winners?

The claim under test: a quarterback, receiver or back has a real edge (or
hole) against a specific opponent, over and above how good he is in general —
and knowing it should make the prediction better.

The feature. Walking every game in order, each player carries two things: how
he has done overall (mean value per game, value = passing + rushing +
receiving EPA) and how he has done against this particular opponent. The
difference between them, shrunk toward zero by how few meetings there have
been, is his "edge vs this team". A game's team-level feature is that edge
averaged over the skill players who actually carry the offense, weighted by
usage. Only games strictly before kickoff are ever read, so nothing leaks.

Then the honest part: add the feature to the model, retrain walk-forward the
way train.py does, and see whether the predictions improve. Run it several
times with different seeds, because one run cannot tell a real effect from
luck, and print the wobble next to every number.

    python vs_opponent_test.py            # the whole thing
    python vs_opponent_test.py --quick    # signal check only, no retraining
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURES, FIRST_SEASON, LAST_SEASON, canon
from train import PARAMS, ELO_BLEND, BACKTEST_SEASONS

SEEDS = [42, 7, 123]

# How many meetings before the "edge vs this team" is taken at face value.
# 2 pseudo-meetings of no-edge mixed in: one game against a team moves the
# number a third of the way, four games move it two thirds.
MEET_SHRINK = 2.0

# Players are only counted if they carry enough of the offense to matter.
MIN_USAGE = 5.0        # attempts + carries + targets, per game, recent average
USAGE_WINDOW = 8       # games of recent usage that decide who "carries" it


def player_weeks() -> pd.DataFrame:
    cols = ["season", "week", "team", "opponent_team", "player_id", "position",
            "attempts", "carries", "targets",
            "passing_epa", "rushing_epa", "receiving_epa"]
    frames = []
    for s in range(FIRST_SEASON, LAST_SEASON + 1):
        try:
            frames.append(nfl.load_player_stats([s], summary_level="week")
                          .select(cols).to_pandas())
        except Exception:
            continue
    ps = pd.concat(frames, ignore_index=True)
    ps = ps[ps["position"].isin(["QB", "RB", "WR", "TE", "FB"])].copy()
    ps["team"] = ps["team"].map(canon)
    ps["opponent_team"] = ps["opponent_team"].map(canon)
    ps["value"] = (ps["passing_epa"].fillna(0) + ps["rushing_epa"].fillna(0)
                   + ps["receiving_epa"].fillna(0))
    ps["usage"] = (ps["attempts"].fillna(0) + ps["carries"].fillna(0)
                   + ps["targets"].fillna(0))
    return ps.sort_values(["season", "week"]).reset_index(drop=True)


def build_edges(ps: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team, opponent) -> usage-weighted edge vs that opponent.

    Everything is accumulated strictly as the season walks forward, so the row
    for a game only ever reflects games already played.
    """
    all_vals: dict = {}          # pid -> [value, ...]
    vs_vals: dict = {}           # (pid, opp) -> [value, ...]
    usage: dict = {}             # pid -> [usage, ...]
    out = []

    for (season, week), chunk in ps.groupby(["season", "week"], sort=True):
        # 1. read: what does the history say about the players in this game
        for (team, opp), side in chunk.groupby(["team", "opponent_team"]):
            num = den = 0.0
            counted = 0
            for r in side.itertuples(index=False):
                use = usage.get(r.player_id, [])
                if len(use) < 3:
                    continue                      # not enough of a track record
                recent_use = float(np.mean(use[-USAGE_WINDOW:]))
                if recent_use < MIN_USAGE:
                    continue                      # not carrying the offense
                hist = all_vals.get(r.player_id, [])
                meetings = vs_vals.get((r.player_id, opp), [])
                if len(hist) < 8 or not meetings:
                    continue                      # no baseline, or never met them
                baseline = float(np.mean(hist))
                edge = float(np.mean(meetings)) - baseline
                edge *= len(meetings) / (len(meetings) + MEET_SHRINK)
                num += recent_use * edge
                den += recent_use
                counted += 1
            if den:
                out.append({"season": season, "week": week, "team": team,
                            "opponent": opp, "vs_edge": num / den,
                            "vs_players": counted})

        # 2. write: only after the whole week has been read
        for r in chunk.itertuples(index=False):
            all_vals.setdefault(r.player_id, []).append(r.value)
            vs_vals.setdefault((r.player_id, r.opponent_team), []).append(r.value)
            usage.setdefault(r.player_id, []).append(r.usage)

    return pd.DataFrame(out)


def attach(df: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    h = edges.rename(columns={"team": "home_team", "opponent": "away_team",
                              "vs_edge": "home_vs_edge",
                              "vs_players": "home_vs_players"})
    a = edges.rename(columns={"team": "away_team", "opponent": "home_team",
                              "vs_edge": "away_vs_edge",
                              "vs_players": "away_vs_players"})
    df = df.merge(h, on=["season", "week", "home_team", "away_team"], how="left")
    df = df.merge(a, on=["season", "week", "home_team", "away_team"], how="left")
    for c in ("home_vs_edge", "away_vs_edge"):
        df[c] = df[c].fillna(0.0)
    df["vs_edge_diff"] = df["home_vs_edge"] - df["away_vs_edge"]
    return df


NEW = ["home_vs_edge", "away_vs_edge", "vs_edge_diff"]


def signal_check(df: pd.DataFrame) -> None:
    d = df[df["home_win"].notna() & (df["season"] >= 2006)].copy()
    have = d[(d["home_vs_players"].notna()) & (d["away_vs_players"].notna())]
    print(f"\ngames with an edge on both sides: {len(have)} of {len(d)} "
          f"({100*len(have)/len(d):.0f}%)")
    print(f"skill players counted per side, median: "
          f"{have[['home_vs_players','away_vs_players']].stack().median():.0f}")
    print(f"vs_edge_diff spread: sd {have.vs_edge_diff.std():.2f} EPA/gm, "
          f"middle 90% {have.vs_edge_diff.quantile(.05):+.1f} to "
          f"{have.vs_edge_diff.quantile(.95):+.1f}")
    print(f"\ncorrelation with the actual result:")
    print(f"  vs_edge_diff  vs  home win      r = {have.vs_edge_diff.corr(have.home_win):+.4f}")
    if "result" in have.columns:
        print(f"  vs_edge_diff  vs  margin        r = {have.vs_edge_diff.corr(have.result):+.4f}")
    print(f"  (for scale) elo_diff vs home win r = {have.elo_diff.corr(have.home_win):+.4f}")

    # Does it say anything the model does not already know?
    resid = have["home_win"] - have["elo_prob"]
    print(f"\n  vs_edge_diff  vs  what Elo MISSES  r = {have.vs_edge_diff.corr(resid):+.4f}")
    top = have[have.vs_edge_diff > have.vs_edge_diff.quantile(0.9)]
    bot = have[have.vs_edge_diff < have.vs_edge_diff.quantile(0.1)]
    print(f"\n  home team's skill players own this opponent (top 10%): "
          f"home wins {top.home_win.mean():.1%} of {len(top)}")
    print(f"  ...and the reverse (bottom 10%):                        "
          f"home wins {bot.home_win.mean():.1%} of {len(bot)}")


def walk_forward(done: pd.DataFrame, feats: list, seed: int) -> dict:
    briers, lls, accs, n = [], [], [], 0
    for season in BACKTEST_SEASONS:
        train = done[done.season < season]
        test = done[done.season == season]
        if test.empty:
            continue
        params = dict(PARAMS, random_state=seed)
        m = XGBClassifier(**params)
        m.fit(train[feats], train["home_win"])
        raw = m.predict_proba(test[feats])[:, 1]
        prob = (1 - ELO_BLEND) * raw + ELO_BLEND * test["elo_prob"].values
        y = test["home_win"].values
        briers.append(brier_score_loss(y, prob) * len(y))
        lls.append(log_loss(y, prob, labels=[0, 1]) * len(y))
        accs.append(accuracy_score(y, prob >= 0.5) * len(y))
        n += len(y)
    return {"brier": sum(briers) / n, "logloss": sum(lls) / n,
            "acc": sum(accs) / n, "n": n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="signal check only")
    ap.add_argument("--pos", default="", help="restrict to one position, e.g. QB")
    args = ap.parse_args()

    df = pd.read_parquet("features.parquet")
    ps = player_weeks()
    if args.pos:
        # A quarterback-only version, in case averaging him in with four
        # receivers is what buries the effect.
        ps = ps[ps["position"] == args.pos.upper()]
        print(f"restricted to {args.pos.upper()}")
    print("building the vs-opponent history (walking every game in order)...")
    edges = build_edges(ps)
    df = attach(df, edges)
    signal_check(df)
    if args.quick:
        return

    done = df[df["home_win"].notna()].copy()
    print(f"\nwalk-forward backtest, {BACKTEST_SEASONS.start}-"
          f"{BACKTEST_SEASONS.stop - 1}, {len(SEEDS)} seeds")
    base, plus = [], []
    for seed in SEEDS:
        base.append(walk_forward(done, FEATURES, seed))
        plus.append(walk_forward(done, FEATURES + NEW, seed))
        print(f"  seed {seed:>3}: brier {base[-1]['brier']:.5f} -> "
              f"{plus[-1]['brier']:.5f}   acc {base[-1]['acc']:.4f} -> "
              f"{plus[-1]['acc']:.4f}")

    print(f"\n{'':10} {'without':>12} {'with':>12} {'change':>12} {'seed wobble':>13}")
    for key, better in (("brier", "lower"), ("logloss", "lower"), ("acc", "higher")):
        b = np.array([r[key] for r in base])
        p = np.array([r[key] for r in plus])
        wobble = max(b.std(), p.std())
        delta = p.mean() - b.mean()
        verdict = "noise" if abs(delta) < wobble else ("better" if
                  ((delta < 0) == (better == "lower")) else "worse")
        print(f"{key:10} {b.mean():>12.5f} {p.mean():>12.5f} "
              f"{delta:>+12.5f} {wobble:>13.5f}   {verdict}")
    print(f"\n{base[0]['n']} games judged. A change smaller than the seed "
          "wobble is not a result.")


if __name__ == "__main__":
    main()
