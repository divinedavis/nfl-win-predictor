"""Do penalties predict anything, or are undisciplined teams just bad teams?

The claim is intuitive: a team that commits twelve penalties a game gives away
yards and drives, so it should lose more. The trouble with intuitive claims is
that the effect can be entirely real and still useless, either because it does
not carry forward or because the model already knows.

Penalties are worth separating rather than counting, because they are not one
thing:

    pre-snap        false start, delay of game, offside, illegal formation.
                    Nobody made a play; the team simply beat itself. This is
                    the closest thing to a pure discipline measurement.
    during-play     offensive holding, pass interference, roughing. These are a
                    by-product of what was being attempted -- a cornerback who
                    interferes was in coverage, a lineman who holds was losing.
                    Partly discipline, partly getting beaten.

The distinction matters for gate 1. Discipline should carry forward if it is a
real trait. Getting beaten should show up as being a bad team, which the model
already measures directly.

    python penalty_ablation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES, LAST_SEASON, canon
from train import PARAMS, ELO_BLEND, BACKTEST_SEASONS

SEEDS = [42, 7, 123, 2024, 99]
ROLL = 8
FIRST = 2006
CACHE = "penalty_team_weeks.csv"

# Penalties where no play happened -- the team beat itself.
PRE_SNAP = {"False Start", "Delay of Game", "Defensive Offside",
            "Neutral Zone Infraction", "Illegal Formation", "Encroachment",
            "Illegal Shift", "Illegal Motion", "Illegal Substitution",
            "Too Many Men on Field", "Defensive Too Many Men on Field"}

NEW = ["home_pen16", "away_pen16", "pen16_diff",
       "home_presnap16", "away_presnap16", "presnap16_diff"]


def build() -> pd.DataFrame:
    frames = []
    for season in range(FIRST, LAST_SEASON + 1):
        try:
            p = nfl.load_pbp([season]).select(
                ["season", "week", "penalty", "penalty_team", "penalty_yards",
                 "penalty_type", "posteam", "defteam"]).to_pandas()
        except Exception:
            continue
        pen = p[p.penalty.fillna(0).gt(0) & p.penalty_team.notna()].copy()
        pen["pre_snap"] = pen.penalty_type.isin(PRE_SNAP).astype(float)
        # Every team that played that week, so a clean game gets a zero rather
        # than a missing row.
        played = pd.concat([
            p[["season", "week", "posteam"]].rename(columns={"posteam": "team"}),
            p[["season", "week", "defteam"]].rename(columns={"defteam": "team"}),
        ]).dropna().drop_duplicates()
        agg = (pen.groupby(["season", "week", "penalty_team"])
               .agg(pens=("penalty", "size"),
                    pen_yards=("penalty_yards", "sum"),
                    pre_snap=("pre_snap", "sum")).reset_index()
               .rename(columns={"penalty_team": "team"}))
        frames.append(played.merge(agg, on=["season", "week", "team"],
                                   how="left").fillna(0))
        print(f"{season}: {len(pen):,} penalties")

    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv(CACHE, index=False)
    return out


def gate1(tw: pd.DataFrame) -> None:
    print("\n=== GATE 1: is being penalised a trait, or just what happened? ===\n")
    rows = []
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        if len(g) < 12:
            continue
        half = len(g) // 2
        a, b = g.iloc[:half], g.iloc[half:]
        rows.append({"all_1": a.pens.mean(), "all_2": b.pens.mean(),
                     "pre_1": a.pre_snap.mean(), "pre_2": b.pre_snap.mean(),
                     "yds_1": a.pen_yards.mean(), "yds_2": b.pen_yards.mean()})
    d = pd.DataFrame(rows)
    for a, b, label in (("all_1", "all_2", "all penalties"),
                        ("pre_1", "pre_2", "pre-snap only"),
                        ("yds_1", "yds_2", "penalty yards")):
        r = d[a].corr(d[b])
        verdict = ("a real, repeatable trait" if r > 0.3 else
                   "weakly repeatable" if r > 0.15 else "mostly luck")
        print(f"   {label:16s} first half vs second:  r = {r:+.3f}   {verdict}")
    print(f"\n   (n = {len(d)} team-seasons. For scale: overall offensive "
          f"quality\n    persists at about +0.39, coaching skill at -0.17.)")


def gate2_raw(tw: pd.DataFrame, feat: pd.DataFrame) -> None:
    """Before modelling: do teams simply win less in games they are penalised in?"""
    print("\n=== Does being penalised more actually lose you games? ===\n")
    res = []
    for side, opp in (("home", "away"), ("away", "home")):
        res.append(pd.DataFrame({
            "season": feat.season, "week": feat.week,
            "team": feat[f"{side}_team"],
            "won": (feat.home_win if side == "home" else 1 - feat.home_win),
            "opp": feat[f"{opp}_team"]}))
    r = pd.concat(res).dropna()
    m = r.merge(tw, on=["season", "week", "team"], how="inner")
    m = m.merge(tw.rename(columns={"team": "opp", "pens": "opp_pens"})
                [["season", "week", "opp", "opp_pens"]],
                on=["season", "week", "opp"], how="inner")
    m["edge"] = m.opp_pens - m.pens          # positive = fewer flags than them
    for lo, hi, label in ((3, 99, "3+ FEWER penalties than the opponent"),
                          (1, 2, "1-2 fewer"), (-0.5, 0.5, "the same"),
                          (-2, -1, "1-2 more"), (-99, -3, "3+ MORE")):
        g = m[m.edge.between(lo, hi)]
        if len(g) >= 200:
            print(f"   {label:38s} won {g.won.mean():.1%}   (n={len(g):,})")
    print("\n   A gap here is real but says nothing about prediction: penalties")
    print("   happen during the game, and cannot be known before it.")


def attach(feat: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    tw = tw.sort_values(["team", "season", "week"]).copy()
    for src, dst in (("pens", "pen_roll"), ("pre_snap", "pre_roll")):
        tw[dst] = (tw.groupby("team")[src]
                   .transform(lambda s: s.shift(1)
                              .rolling(ROLL, min_periods=3).mean()))
    key = tw.groupby(["season", "week", "team"])[["pen_roll", "pre_roll"]].mean()

    out = feat.copy()
    for side, col in (("home", "home_team"), ("away", "away_team")):
        idx = pd.MultiIndex.from_arrays([out.season, out.week, out[col]])
        out[f"{side}_pen16"] = key.pen_roll.reindex(idx).to_numpy()
        out[f"{side}_presnap16"] = key.pre_roll.reindex(idx).to_numpy()
    out["pen16_diff"] = out.home_pen16 - out.away_pen16
    out["presnap16_diff"] = out.home_presnap16 - out.away_presnap16
    for c in NEW:
        out[c] = out[c].fillna(0.0)
    return out


def gate3(feat: pd.DataFrame) -> None:
    print("\n=== GATE 2: does knowing it beforehand improve the model? ===\n")
    done = feat[feat.home_win.notna()].copy()

    def run(cols, seed):
        P, Y = [], []
        for s in BACKTEST_SEASONS:
            tr, te = done[done.season < s], done[done.season == s]
            if tr.empty or te.empty:
                continue
            m = XGBClassifier(**{**PARAMS, "random_state": seed})
            m.fit(tr[cols], tr["home_win"])
            P.append((1 - ELO_BLEND) * m.predict_proba(te[cols])[:, 1]
                     + ELO_BLEND * te.elo_prob.to_numpy())
            Y.append(te.home_win.to_numpy())
        p, y = np.concatenate(P), np.concatenate(Y)
        return brier_score_loss(y, p), accuracy_score(y, p > 0.5)

    variants = {
        "model as it is now": FEATURES,
        "+ penalties per game": FEATURES + ["home_pen16", "away_pen16",
                                            "pen16_diff"],
        "+ pre-snap only": FEATURES + ["home_presnap16", "away_presnap16",
                                       "presnap16_diff"],
        "+ both": FEATURES + NEW,
    }
    base = None
    print(f"   {'':28s} {'brier':>18} {'right':>17}")
    for name, cols in variants.items():
        b, a = zip(*[run(cols, s) for s in SEEDS])
        bm, bs = float(np.mean(b)), float(np.std(b))
        if base is None:
            base = bm
        gap = bm - base
        tag = "" if gap == 0 else ("   BETTER" if gap < -bs else
                                   "   worse" if gap > bs else "   no different")
        print(f"   {name:28s} {bm:.5f} +/-{bs:.5f} {np.mean(a):.4f} "
              f"+/-{np.std(a):.4f}{tag}")


def main() -> None:
    try:
        tw = pd.read_csv(CACHE)
    except FileNotFoundError:
        tw = build()
    feat = pd.read_parquet("features.parquet")
    print(f"\n{len(tw):,} team-games, {int(tw.season.min())}-{int(tw.season.max())}, "
          f"averaging {tw.pens.mean():.1f} penalties "
          f"({tw.pre_snap.mean():.1f} pre-snap) per team per game")
    gate1(tw)
    gate2_raw(tw, feat)
    gate3(attach(feat, tw))


if __name__ == "__main__":
    main()
