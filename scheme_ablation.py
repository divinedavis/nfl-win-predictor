"""Does the scheme matchup matter? Three gates, all of which have to hold.

The claim being tested is the one every broadcast makes: this offence struggles
against man coverage, and it is about to play a defence that plays a lot of man,
so expect trouble. For that to be predictable rather than a story told
afterwards, three separate things have to be true at once.

GATE 1  The defence has a stable preference. If a team's man-coverage rate
        bounces around week to week, knowing last year's rate tells you nothing
        about Sunday.

GATE 2  The offence has a stable strength against one coverage over the other.
        This is the one that usually kills matchup claims. An offence can be
        +0.15 EPA better against man over half a season purely by luck, and if
        that edge does not carry into the second half there is no trait to
        match up against -- only a number that already happened.

GATE 3  Knowing both improves the prediction. Even if the first two hold, the
        model already carries offensive and defensive EPA, so the matchup only
        earns a place by adding something those cannot have absorbed.

Gate 2 failing is decisive on its own: without a persistent edge there is
nothing for a defensive tendency to interact with, and gate 3 cannot be real
even if it looks like it is.

    python scheme_ablation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, ELO_BLEND

SEEDS = [42, 7, 123, 2024, 99]
ROLL = 16
# Coverage data starts in 2018, so the backtest starts once there is enough
# history to roll over.
SCHEME_SEASONS = range(2020, 2026)
NEW = ["home_def_man", "away_def_man", "home_man_adv", "away_man_adv",
       "home_scheme_edge", "away_scheme_edge"]


def gate1_defence(tw: pd.DataFrame) -> None:
    print("\n=== GATE 1: does a defence stick to its preference? ===\n")
    rows = []
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        g = g[g.def_cov_snaps.fillna(0) >= 10]
        if len(g) < 12:
            continue
        half = len(g) // 2
        rows.append({"first": g.iloc[:half].def_man_rate.mean(),
                     "second": g.iloc[half:].def_man_rate.mean()})
    d = pd.DataFrame(rows)
    r = d["first"].corr(d["second"])
    print(f"   man-coverage rate, first half vs second half:  r = {r:+.3f}")
    print(f"   {'PASSES -- defences do have a stable identity' if r > 0.3 else 'FAILS -- too unstable to predict from'}"
          f"   (n = {len(d)} team-seasons)")


def gate2_offence(tw: pd.DataFrame) -> None:
    print("\n=== GATE 2: is an offence reliably better against one coverage? ===\n")
    rows = []
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        g = g[(g.off_snaps_man.fillna(0) >= 5) & (g.off_snaps_zone.fillna(0) >= 5)]
        if len(g) < 12:
            continue
        half = len(g) // 2
        a, b = g.iloc[:half], g.iloc[half:]
        rows.append({
            "first": (a.off_epa_man.mean() - a.off_epa_zone.mean()),
            "second": (b.off_epa_man.mean() - b.off_epa_zone.mean()),
            "epa_first": a.off_epa_zone.mean(), "epa_second": b.off_epa_zone.mean(),
        })
    d = pd.DataFrame(rows)
    r = d["first"].corr(d["second"])
    # A control: overall offensive quality obviously persists, so this shows
    # what a real persistent trait looks like on the same games.
    ctrl = d["epa_first"].corr(d["epa_second"])
    print(f"   advantage vs man over zone, first half vs second:  r = {r:+.3f}")
    print(f"   overall offensive quality, same split (control):   r = {ctrl:+.3f}")
    print()
    if r > 0.3:
        print("   PASSES -- the edge carries forward")
    elif r > 0.15:
        print("   WEAK -- carries forward a little")
    else:
        print("   FAILS -- the edge does not carry forward. An offence that "
              "looked\n   better against man in the first half is no more "
              "likely than chance\n   to be better against it in the second. "
              "There is no trait here for a\n   defensive tendency to interact "
              "with.")
    print(f"   (n = {len(d)} team-seasons)")


def attach(feat: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Roll each team's scheme numbers forward, using only earlier games."""
    tw = tw.sort_values(["team", "season", "week"]).copy()
    tw["man_adv"] = tw.off_epa_man - tw.off_epa_zone
    for col, src in (("man_roll", "def_man_rate"), ("adv_roll", "man_adv")):
        tw[col] = (tw.groupby("team")[src]
                   .transform(lambda s: s.shift(1)
                              .rolling(ROLL, min_periods=6).mean()))
    # The outer merges can leave a team twice in a week (a formation row with
    # no coverage row, say); collapse to one before indexing.
    key = (tw.groupby(["season", "week", "team"])[["man_roll", "adv_roll"]]
           .mean())

    out = feat.copy()
    for side, col in (("home", "home_team"), ("away", "away_team")):
        idx = pd.MultiIndex.from_arrays([out.season, out.week, out[col]])
        out[f"{side}_def_man"] = key.man_roll.reindex(idx).to_numpy()
        out[f"{side}_man_adv"] = key.adv_roll.reindex(idx).to_numpy()
    # The matchup itself: an offence's edge against man, weighted by how much
    # man the defence it faces actually plays.
    out["home_scheme_edge"] = out.home_man_adv * out.away_def_man
    out["away_scheme_edge"] = out.away_man_adv * out.home_def_man
    for c in NEW:
        out[c] = out[c].fillna(0.0)
    return out


def gate3_model(feat: pd.DataFrame) -> None:
    print("\n=== GATE 3: does the model get better? ===\n")
    done = feat[feat.home_win.notna()].copy()

    def run(cols, seed):
        P, Y = [], []
        for s in SCHEME_SEASONS:
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
        "+ how much man each plays": FEATURES + ["home_def_man", "away_def_man"],
        "+ the matchup itself": FEATURES + NEW,
    }
    base = None
    print(f"   {'':32s} {'brier':>18} {'right':>17}")
    for name, cols in variants.items():
        b, a = zip(*[run(cols, s) for s in SEEDS])
        bm, bs = float(np.mean(b)), float(np.std(b))
        if base is None:
            base = bm
        gap = bm - base
        tag = "" if gap == 0 else ("   BETTER" if gap < -bs else
                                   "   worse" if gap > bs else "   no different")
        print(f"   {name:32s} {bm:.5f} +/-{bs:.5f} {np.mean(a):.4f} "
              f"+/-{np.std(a):.4f}{tag}")
    print(f"\n   ({int(min(SCHEME_SEASONS))}-{int(max(SCHEME_SEASONS))} only -- "
          "coverage data starts in 2018 and needs\n   history to roll over, so "
          "this is a shorter backtest than usual.)")


def main() -> None:
    feat = pd.read_parquet("features.parquet")
    tw = pd.read_csv("scheme_team_weeks.csv")
    gate1_defence(tw)
    gate2_offence(tw)
    gate3_model(attach(feat, tw))


if __name__ == "__main__":
    main()
