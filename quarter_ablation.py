"""Do fast starters and strong finishers exist, and do they win more?

Quarters sum to the final score, and Elo is built from final scores, so a team
that plays well in the first quarter has already had that counted. Asking
whether good first-quarter teams win more is therefore the wrong question --
they do, and Elo knows.

The right question is whether the SHAPE carries anything the total does not.
Two teams with identical season-long EPA, one of which is consistently strong
early and fades, the other slow to start and dangerous late: is that a real
difference that persists, and does knowing it help?

So every quarter number here is measured against that team's own average.
A team at +0.05 in the first quarter is +0.05 better in the first quarter than
it is overall -- not better than the league. That subtraction is the whole
point: it removes exactly the part Elo already has.

Three things get tested:

    raw          does first-quarter EPA persist (it will -- good teams are good)
    shape        does first-quarter-ABOVE-OWN-AVERAGE persist (the real test)
    halftime     does third-quarter-above-own-average persist, which is the
                 coaching claim about adjustments made at the break

    python quarter_ablation.py
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
ROLL = 16
FIRST = 2006
CACHE = "quarter_team_weeks.csv"
QUARTERS = [1, 2, 3, 4]

NEW = ([f"{s}_q{q}_off" for s in ("home", "away") for q in QUARTERS]
       + [f"{s}_q{q}_def" for s in ("home", "away") for q in QUARTERS])


def build() -> pd.DataFrame:
    frames = []
    for season in range(FIRST, LAST_SEASON + 1):
        try:
            p = nfl.load_pbp([season]).select(
                ["season", "week", "qtr", "posteam", "defteam", "epa",
                 "play_type"]).to_pandas()
        except Exception:
            continue
        p = p[p.play_type.isin(["run", "pass"]) & p.epa.notna()
              & p.posteam.notna() & p.qtr.isin(QUARTERS)]
        off = (p.groupby(["season", "week", "posteam", "qtr"]).epa
               .agg(off_epa="mean", off_n="size").reset_index()
               .rename(columns={"posteam": "team"}))
        dfn = (p.groupby(["season", "week", "defteam", "qtr"]).epa
               .agg(def_epa="mean", def_n="size").reset_index()
               .rename(columns={"defteam": "team"}))
        dfn["def_epa"] = -dfn["def_epa"]      # positive is always good
        frames.append(off.merge(dfn, on=["season", "week", "team", "qtr"],
                                how="outer"))
        print(f"{season}: {len(p):,} plays")

    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week", "team", "qtr"]).reset_index(drop=True)
    out.to_csv(CACHE, index=False)
    return out


def team_game_overall(tw: pd.DataFrame) -> pd.DataFrame:
    """Each team-game's EPA across all four quarters, to measure shape against.

    Deliberately NOT the season average. Subtracting a whole-season mean is a
    leak twice over: it uses games that have not happened yet, and when the two
    halves of a season are both measured against their shared mean they are
    forced to point in opposite directions, which manufactures a negative
    correlation out of arithmetic.
    """
    w = tw.copy()
    for side in ("off", "def"):
        w[f"{side}_w"] = w[f"{side}_epa"] * w[f"{side}_n"].fillna(0)
    g = (w.groupby(["season", "week", "team"])
         .agg(off_w=("off_w", "sum"), off_n=("off_n", "sum"),
              def_w=("def_w", "sum"), def_n=("def_n", "sum")).reset_index())
    g["off_game"] = g.off_w / g.off_n.replace(0, np.nan)
    g["def_game"] = g.def_w / g.def_n.replace(0, np.nan)
    return g[["season", "week", "team", "off_game", "def_game"]]


def shapes(tw: pd.DataFrame) -> pd.DataFrame:
    """Attach each team-game's overall EPA alongside its per-quarter EPA."""
    return tw.merge(team_game_overall(tw), on=["season", "week", "team"],
                    how="left")


def gate1(d: pd.DataFrame) -> None:
    print("\n=== GATE 1: is a quarter profile a real trait? ===")
    print("   Split each season in half; does the first half predict the second?\n")
    print(f"   {'':26s} {'raw':>9} {'shape':>19}")
    for side, label, ncol in (("off", "offence", "off_n"), ("def", "defence", "def_n")):
        for q in QUARTERS:
            raw, shape = [], []
            for (season, team), g in d[d.qtr == q].groupby(["season", "team"]):
                g = g.sort_values("week")
                g = g[g[ncol].fillna(0) >= 8]
                if len(g) < 12:
                    continue
                half = len(g) // 2
                a, b = g.iloc[:half], g.iloc[half:]
                raw.append((a[f"{side}_epa"].mean(), b[f"{side}_epa"].mean()))
                # Each half measured against ITS OWN overall EPA, never a mean
                # the two halves share.
                shape.append((a[f"{side}_epa"].mean() - a[f"{side}_game"].mean(),
                              b[f"{side}_epa"].mean() - b[f"{side}_game"].mean()))
            r1 = pd.DataFrame(raw, columns=["a", "b"])
            r2 = pd.DataFrame(shape, columns=["a", "b"])
            print(f"   quarter {q} {label:12s} {r1.a.corr(r1.b):+9.3f} "
                  f"{r2.a.corr(r2.b):+19.3f}")
    print("\n   The 'raw' column is expected to be positive -- good teams are good")
    print("   in every quarter, which Elo already knows. The column that matters")
    print("   is the second: whether a team is reliably better in one quarter")
    print("   than it is in the others. Above about +0.15 there is a shape worth")
    print("   knowing; near zero there is only the total, already counted.")


def gate_halftime(d: pd.DataFrame) -> None:
    print("\n=== Do halftime adjustments exist? ===")
    print("   Third quarter against that team's own first half, one season to")
    print("   the next and within a season.\n")
    adj = []
    for (season, team), g in d.groupby(["season", "team"]):
        first = g[g.qtr.isin([1, 2])]
        third = g[g.qtr == 3]
        if len(first) < 16 or len(third) < 8:
            continue
        adj.append({"season": season, "team": team,
                    "adj": third.off_epa.mean() - first.off_epa.mean()})  # both measured within the same season, no shared subtraction
    a = pd.DataFrame(adj)
    nxt = a.copy()
    nxt["season"] = nxt.season - 1
    j = a.merge(nxt, on=["season", "team"], suffixes=("_now", "_next"))
    r = j.adj_now.corr(j.adj_next)
    print(f"   a team's third-quarter jump, this season vs next:  r = {r:+.3f}"
          f"   (n={len(j)})")
    print(f"   {'a real coaching trait' if r > 0.2 else 'no evidence a staff reliably adjusts better than another'}")


def attach(feat: pd.DataFrame, d: pd.DataFrame) -> pd.DataFrame:
    """Roll each team's quarter shape forward, using only earlier games."""
    out = feat.copy()
    for q in QUARTERS:
        g = d[d.qtr == q].sort_values(["team", "season", "week"]).copy()
        for side in ("off", "def"):
            # Roll the quarter and the whole game separately, both shifted so a
            # game never sees itself, then difference the two rolled values.
            # Subtracting first and rolling after would smuggle the current
            # game's own overall EPA into its own feature.
            q_roll = (g.groupby("team")[f"{side}_epa"]
                      .transform(lambda s: s.shift(1)
                                 .rolling(ROLL, min_periods=6).mean()))
            all_roll = (g.groupby("team")[f"{side}_game"]
                        .transform(lambda s: s.shift(1)
                                   .rolling(ROLL, min_periods=6).mean()))
            g[f"{side}_roll"] = q_roll - all_roll
        key = g.groupby(["season", "week", "team"])[["off_roll", "def_roll"]].mean()
        for side_name, col in (("home", "home_team"), ("away", "away_team")):
            idx = pd.MultiIndex.from_arrays(
                [out.season, out.week, out[col]])
            out[f"{side_name}_q{q}_off"] = key.off_roll.reindex(idx).to_numpy()
            out[f"{side_name}_q{q}_def"] = key.def_roll.reindex(idx).to_numpy()
    for c in NEW:
        out[c] = out[c].fillna(0.0)
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

    q1 = [f"{s}_q1_{t}" for s in ("home", "away") for t in ("off", "def")]
    q3 = [f"{s}_q3_{t}" for s in ("home", "away") for t in ("off", "def")]
    variants = {
        "model as it is now": FEATURES,
        "+ first-quarter shape": FEATURES + q1,
        "+ third-quarter shape": FEATURES + q3,
        "+ all four quarters": FEATURES + NEW,
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
    d = shapes(tw)
    feat = pd.read_parquet("features.parquet")
    print(f"\n{len(tw):,} team-game-quarters, "
          f"{int(tw.season.min())}-{int(tw.season.max())}")
    gate1(d)
    gate_halftime(d)
    gate2(attach(feat, d))


if __name__ == "__main__":
    main()
