"""Two questions the earlier tests did not answer.

DEFENSIVE PENALTIES. penalty_ablation.py grouped by whoever committed the foul
and never split offence from defence, so it could not see this. The two are not
the same thing at all. A false start costs five yards and a down; defensive pass
interference is spotted at the catch point and comes with an automatic first
down, and so does defensive holding. A defence that fouls is handing out drives,
not yards, and that should show up differently from an offence that fouls.

POSITIVE PLAYS. The model measures both sides with EPA, which is an average of
how much each play was worth. An average is dominated by its extremes: one
eighty-yard touchdown outweighs a dozen three-yard gains, so an EPA figure can
swing on a handful of snaps. Success rate asks a blunter question -- what share
of plays gained anything at all -- and blunter measures are often steadier. If
success rate carries forward better than EPA, it may be the better input, and
replacing EPA with it is a substitution rather than an addition, which on this
project is the shape that has actually worked.

Both are tested the same way: does it persist, and does the model improve.
Replacement is tested as well as addition, because two measures of the same
thing usually work better as one than as both.

    python positive_plays_ablation.py
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
ROLL = 8                  # matches the 8-game window features.py already uses
FIRST = 2006
CACHE = "positive_team_weeks.csv"

EPA_COLS = ["home_off_epa8", "away_off_epa8", "off_epa8_diff",
            "home_def_epa8", "away_def_epa8", "def_epa8_diff"]
SUCCESS_COLS = ["home_off_succ8", "away_off_succ8", "off_succ8_diff",
                "home_def_succ8", "away_def_succ8", "def_succ8_diff"]
PEN_COLS = ["home_def_pen8", "away_def_pen8", "def_pen8_diff",
            "home_off_pen8", "away_off_pen8"]


def build() -> pd.DataFrame:
    frames = []
    for season in range(FIRST, LAST_SEASON + 1):
        try:
            p = nfl.load_pbp([season]).select(
                ["season", "week", "posteam", "defteam", "epa", "success",
                 "play_type", "penalty", "penalty_team", "penalty_yards"]
            ).to_pandas()
        except Exception:
            continue

        s = p[p.play_type.isin(["run", "pass"]) & p.success.notna()
              & p.posteam.notna()]
        off = (s.groupby(["season", "week", "posteam"])
               .agg(off_succ=("success", "mean")).reset_index()
               .rename(columns={"posteam": "team"}))
        dfn = (s.groupby(["season", "week", "defteam"])
               .agg(def_succ=("success", "mean")).reset_index()
               .rename(columns={"defteam": "team"}))
        # Flip so a higher number is always better for the team named.
        dfn["def_succ"] = 1 - dfn["def_succ"]

        # Penalties, split by which side of the ball committed them.
        pen = p[p.penalty.fillna(0).gt(0) & p.penalty_team.notna()].copy()
        pen["side"] = np.where(pen.penalty_team == pen.posteam, "off",
                               np.where(pen.penalty_team == pen.defteam,
                                        "def", "other"))
        pen = pen[pen.side != "other"]
        pc = (pen.groupby(["season", "week", "penalty_team", "side"])
              .agg(n=("penalty", "size"), yds=("penalty_yards", "sum"))
              .reset_index().rename(columns={"penalty_team": "team"}))
        pc = pc.pivot_table(index=["season", "week", "team"], columns="side",
                            values=["n", "yds"], fill_value=0).reset_index()
        pc.columns = ["season", "week", "team", "def_pen", "off_pen",
                      "def_pen_yds", "off_pen_yds"]

        m = off.merge(dfn, on=["season", "week", "team"], how="outer")
        m = m.merge(pc, on=["season", "week", "team"], how="left").fillna(
            {"def_pen": 0, "off_pen": 0, "def_pen_yds": 0, "off_pen_yds": 0})
        frames.append(m)
        print(f"{season}: {len(s):,} plays, {len(pen):,} penalties")

    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv(CACHE, index=False)
    return out


def persistence(tw: pd.DataFrame) -> None:
    print("\n=== GATE 1: which of these are real traits? ===")
    print("   Split each season in half; does the first half predict the second?\n")
    cols = [("off_succ", "offence, positive-play rate"),
            ("def_succ", "defence, stopping rate"),
            ("def_pen", "DEFENSIVE penalties per game"),
            ("off_pen", "offensive penalties per game"),
            ("def_pen_yds", "defensive penalty yards")]
    rows = {c: [] for c, _ in cols}
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        if len(g) < 12:
            continue
        half = len(g) // 2
        for c, _ in cols:
            rows[c].append((g.iloc[:half][c].mean(), g.iloc[half:][c].mean()))
    for c, label in cols:
        d = pd.DataFrame(rows[c], columns=["a", "b"])
        r = d.a.corr(d.b)
        verdict = ("a real, repeatable trait" if r > 0.3 else
                   "weakly repeatable" if r > 0.15 else "mostly luck")
        print(f"   {label:32s} r = {r:+.3f}   {verdict}")
    print(f"\n   (n = {len(rows['off_succ'])} team-seasons)")


def raw_effect(tw: pd.DataFrame, feat: pd.DataFrame) -> None:
    print("\n=== Do defensive penalties actually cost you the game? ===\n")
    res = []
    for side, opp in (("home", "away"), ("away", "home")):
        res.append(pd.DataFrame({
            "season": feat.season, "week": feat.week,
            "team": feat[f"{side}_team"], "opp": feat[f"{opp}_team"],
            "won": (feat.home_win if side == "home" else 1 - feat.home_win)}))
    r = pd.concat(res).dropna()
    m = r.merge(tw[["season", "week", "team", "def_pen"]],
                on=["season", "week", "team"], how="inner")
    m = m.merge(tw[["season", "week", "team", "def_pen"]]
                .rename(columns={"team": "opp", "def_pen": "opp_def_pen"}),
                on=["season", "week", "opp"], how="inner")
    m["edge"] = m.opp_def_pen - m.def_pen
    for lo, hi, label in ((2, 99, "2+ FEWER defensive fouls than them"),
                          (1, 1, "1 fewer"), (0, 0, "the same"),
                          (-1, -1, "1 more"), (-99, -2, "2+ MORE")):
        g = m[m.edge.between(lo, hi)]
        if len(g) >= 200:
            print(f"   {label:38s} won {g.won.mean():.1%}   (n={len(g):,})")


def attach(feat: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    tw = tw.sort_values(["team", "season", "week"]).copy()
    for src in ("off_succ", "def_succ", "def_pen", "off_pen"):
        tw[f"{src}_roll"] = (tw.groupby("team")[src]
                             .transform(lambda s: s.shift(1)
                                        .rolling(ROLL, min_periods=3).mean()))
    key = tw.groupby(["season", "week", "team"])[
        [f"{c}_roll" for c in ("off_succ", "def_succ", "def_pen", "off_pen")]].mean()

    out = feat.copy()
    for side, col in (("home", "home_team"), ("away", "away_team")):
        idx = pd.MultiIndex.from_arrays([out.season, out.week, out[col]])
        out[f"{side}_off_succ8"] = key.off_succ_roll.reindex(idx).to_numpy()
        out[f"{side}_def_succ8"] = key.def_succ_roll.reindex(idx).to_numpy()
        out[f"{side}_def_pen8"] = key.def_pen_roll.reindex(idx).to_numpy()
        out[f"{side}_off_pen8"] = key.off_pen_roll.reindex(idx).to_numpy()
    out["off_succ8_diff"] = out.home_off_succ8 - out.away_off_succ8
    out["def_succ8_diff"] = out.home_def_succ8 - out.away_def_succ8
    out["def_pen8_diff"] = out.home_def_pen8 - out.away_def_pen8
    for c in SUCCESS_COLS + PEN_COLS:
        out[c] = out[c].fillna(out[c].median() if out[c].notna().any() else 0.0)
    return out


def gate2(feat: pd.DataFrame) -> None:
    print("\n=== GATE 2: does the model get better? ===\n")
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

    without_epa = [c for c in FEATURES if c not in EPA_COLS]
    variants = {
        "model as it is now": FEATURES,
        "+ defensive penalties": FEATURES + PEN_COLS,
        "+ positive-play rate": FEATURES + SUCCESS_COLS,
        "positive-play rate INSTEAD of EPA": without_epa + SUCCESS_COLS,
        "+ both new groups": FEATURES + SUCCESS_COLS + PEN_COLS,
    }
    base = None
    print(f"   {'':36s} {'brier':>18} {'right':>17}")
    for name, cols in variants.items():
        b, a = zip(*[run(cols, s) for s in SEEDS])
        bm, bs = float(np.mean(b)), float(np.std(b))
        if base is None:
            base = bm
        gap = bm - base
        tag = "" if gap == 0 else ("   BETTER" if gap < -bs else
                                   "   worse" if gap > bs else "   no different")
        print(f"   {name:36s} {bm:.5f} +/-{bs:.5f} {np.mean(a):.4f} "
              f"+/-{np.std(a):.4f}{tag}")


def main() -> None:
    try:
        tw = pd.read_csv(CACHE)
    except FileNotFoundError:
        tw = build()
    feat = pd.read_parquet("features.parquet")
    print(f"\n{len(tw):,} team-games, {int(tw.season.min())}-{int(tw.season.max())}")
    print(f"   defences average {tw.def_pen.mean():.1f} penalties a game, "
          f"offences {tw.off_pen.mean():.1f}")
    persistence(tw)
    raw_effect(tw, feat)
    gate2(attach(feat, tw))


if __name__ == "__main__":
    main()
