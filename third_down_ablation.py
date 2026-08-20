"""Does third-down defence carry information the model does not already have?

Two gates, the same pair the clutch features had to pass, and both have to be
cleared before anything is promoted. They ask different things and a feature can
easily pass one and fail the other.

GATE 1 -- is it a real trait, or just what happened?
    Split each team's season in half and compare the two halves. If teams that
    stop third downs early in a season keep doing it later, the number is
    measuring a skill. If the halves are unrelated, it is measuring luck, and no
    amount of modelling turns luck into a prediction. This is where the coach
    hypothesis died: coaching skill beyond Elo had a split-half correlation of
    -0.17, meaning the "signal" was noise.

GATE 2 -- does the model get better?
    Add the columns, rebuild, and score it the way train.py does -- trained only
    on earlier seasons. Run it several times with different random draws,
    because the offensive-line idea looked like a winner on one draw and turned
    out to be nothing across eight.

The trap this is built to avoid: third-down defence could be entirely real and
still be worthless here, because the model already carries defensive EPA over
the last eight games. If stopping third downs is just what good defences do,
the information is already in. A feature only earns its place by carrying
something Elo and EPA cannot have absorbed through the scoreboard.

    python third_down_ablation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, ELO_BLEND, BACKTEST_SEASONS

SEEDS = [42, 7, 123, 2024, 99]
ROLL = 16          # games in the rolling average, matching the clutch features
NEW = ["home_td_def16", "away_td_def16", "home_td_off16", "away_td_off16",
       "td_net_diff"]


def gate1_persistence(tw: pd.DataFrame) -> None:
    """Do teams that stop third downs early keep stopping them later?"""
    print("\n=== GATE 1: is it a real trait? ===")
    print("   Split each team's season in half; compare the halves.\n")
    rows = []
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        if len(g) < 12:
            continue
        half = len(g) // 2
        a, b = g.iloc[:half], g.iloc[half:]
        if a.def_3d_att.sum() < 30 or b.def_3d_att.sum() < 30:
            continue
        rows.append({
            "def_first": a.def_3d_above.sum() / a.def_3d_att.sum(),
            "def_second": b.def_3d_above.sum() / b.def_3d_att.sum(),
            "off_first": a.off_3d_above.sum() / a.off_3d_att.sum(),
            "off_second": b.off_3d_above.sum() / b.off_3d_att.sum(),
        })
    d = pd.DataFrame(rows)
    for side in ("def", "off"):
        r = d[f"{side}_first"].corr(d[f"{side}_second"])
        label = "defence" if side == "def" else "offence"
        verdict = ("a real, repeatable skill" if r > 0.3 else
                   "weakly repeatable" if r > 0.15 else
                   "mostly luck -- does not carry forward")
        print(f"   {label:8s} first half vs second half:  r = {r:+.3f}   "
              f"{verdict}")
    print(f"\n   (n = {len(d)} team-seasons; for reference, clutch scored about "
          f"+0.08\n    and was promoted anyway on gate 2, while coaching scored "
          f"-0.17 and was not)")


def attach(feat: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    """Roll each team's third-down record forward, using only earlier games."""
    tw = tw.sort_values(["team", "season", "week"]).copy()
    for side in ("def", "off"):
        rate = tw[f"{side}_3d_above"] / tw[f"{side}_3d_att"].replace(0, np.nan)
        tw[f"{side}_rate"] = rate
        # shift(1) so a game never sees its own result.
        tw[f"{side}_roll"] = (tw.groupby("team")[f"{side}_rate"]
                              .transform(lambda s: s.shift(1)
                                         .rolling(ROLL, min_periods=4).mean()))
    key = tw.set_index(["season", "week", "team"])

    out = feat.copy()
    for prefix, col in (("home", "home_team"), ("away", "away_team")):
        idx = pd.MultiIndex.from_arrays(
            [out.season, out.week, out[col]])
        out[f"{prefix}_td_def16"] = key.def_roll.reindex(idx).to_numpy()
        out[f"{prefix}_td_off16"] = key.off_roll.reindex(idx).to_numpy()
    out["td_net_diff"] = ((out.home_td_off16 - out.away_td_def16)
                          - (out.away_td_off16 - out.home_td_def16))
    for c in NEW:
        out[c] = out[c].fillna(0.0)
    return out


def gate2_model(feat: pd.DataFrame) -> None:
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

    variants = {
        "model as it is now": FEATURES,
        "+ third-down defence only": FEATURES + ["home_td_def16", "away_td_def16"],
        "+ third down, both sides": FEATURES + NEW,
    }
    base = None
    print(f"   {'':30s} {'brier':>18} {'right':>17}")
    for name, cols in variants.items():
        b, a = zip(*[run(cols, s) for s in SEEDS])
        bm, bs = float(np.mean(b)), float(np.std(b))
        if base is None:
            base = bm
        gap = bm - base
        tag = "" if gap == 0 else (
            "   BETTER" if gap < -bs else
            "   worse" if gap > bs else "   no different")
        print(f"   {name:30s} {bm:.5f} +/-{bs:.5f} {np.mean(a):.4f} "
              f"+/-{np.std(a):.4f}{tag}")
    print("\n   A variant has to beat the current model by more than its own "
          "wobble\n   to be worth promoting. Anything else is noise.")


def main() -> None:
    feat = pd.read_parquet("features.parquet")
    tw = pd.read_csv("third_down_team_weeks.csv")
    gate1_persistence(tw)
    gate2_model(attach(feat, tw))


if __name__ == "__main__":
    main()
