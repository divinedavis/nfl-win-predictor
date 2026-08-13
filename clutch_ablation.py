"""Two-gate test of 'finishing ability' (last-5-min one-score EPA) features.

Gate 1 — persistence: does first-half-of-season clutch EPA predict
second-half clutch EPA, and does it add anything beyond overall offensive
EPA? If the clutch residual doesn't persist, it's luck wearing a narrative.

Gate 2 — model value: walk-forward game-model backtest with rolling clutch
features; scored overall AND on the one-score final-margin bucket.

    python clutch_ablation.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND

ROLL_GAMES = 16
MIN_PLAYS = 40

CLUTCH_FEATS = ["home_clutch_off16", "home_clutch_def16",
                "away_clutch_off16", "away_clutch_def16", "clutch_net_diff"]


def rolling_clutch() -> pd.DataFrame:
    c = pd.read_csv("clutch_team_weeks.csv").sort_values(["season", "week"])
    g = c.groupby("team")
    for side in ("off", "def"):
        c[f"{side}_epa_r"] = g[f"{side}_clutch_epa"].transform(
            lambda s: s.shift(1).rolling(ROLL_GAMES, min_periods=8).sum())
        c[f"{side}_plays_r"] = g[f"{side}_clutch_plays"].transform(
            lambda s: s.shift(1).rolling(ROLL_GAMES, min_periods=8).sum())
        rate = c[f"{side}_epa_r"] / c[f"{side}_plays_r"].replace(0, np.nan)
        c[f"clutch_{side}16"] = rate.where(c[f"{side}_plays_r"] >= MIN_PLAYS)
    return c[["season", "week", "team", "clutch_off16", "clutch_def16"]]


def gate1_persistence() -> None:
    c = pd.read_csv("clutch_team_weeks.csv")
    halves = []
    for (season, team), grp in c.groupby([c.season, c.team]):
        reg = grp[grp.week <= 18]
        h1, h2 = reg[reg.week <= 9], reg[reg.week > 9]
        if h1.off_clutch_plays.sum() < 30 or h2.off_clutch_plays.sum() < 30:
            continue
        halves.append({
            "off1": h1.off_clutch_epa.sum() / h1.off_clutch_plays.sum(),
            "off2": h2.off_clutch_epa.sum() / h2.off_clutch_plays.sum(),
            "def1": h1.def_clutch_epa.sum() / h1.def_clutch_plays.sum(),
            "def2": h2.def_clutch_epa.sum() / h2.def_clutch_plays.sum(),
        })
    h = pd.DataFrame(halves)
    print(f"=== Gate 1: within-season persistence ({len(h)} team-seasons) ===")
    print(f"clutch OFF 1st half -> 2nd half: r = {h.off1.corr(h.off2):+.3f}")
    print(f"clutch DEF 1st half -> 2nd half: r = {h.def1.corr(h.def2):+.3f}")

    # Does clutch-specific info survive once overall offense is known?
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna() & (df.game_type == "REG")]
    overall = pd.concat([
        done[["season", "week", "home_team", "home_off_epa8"]].rename(
            columns={"home_team": "team", "home_off_epa8": "off8"}),
        done[["season", "week", "away_team", "away_off_epa8"]].rename(
            columns={"away_team": "team", "away_off_epa8": "off8"}),
    ])
    o1 = (overall[overall.week <= 9].groupby(["season", "team"])["off8"]
          .mean().rename("overall1"))
    halves2 = []
    for (season, team), grp in c.groupby([c.season, c.team]):
        reg = grp[grp.week <= 18]
        h1, h2 = reg[reg.week <= 9], reg[reg.week > 9]
        if h1.off_clutch_plays.sum() < 30 or h2.off_clutch_plays.sum() < 30:
            continue
        ov = o1.get((season, team))
        if ov is None or pd.isna(ov):
            continue
        halves2.append({
            "clutch1": h1.off_clutch_epa.sum() / h1.off_clutch_plays.sum(),
            "clutch2": h2.off_clutch_epa.sum() / h2.off_clutch_plays.sum(),
            "overall1": ov,
        })
    h2df = pd.DataFrame(halves2)
    import numpy.linalg as la
    X = np.column_stack([np.ones(len(h2df)), h2df.overall1])
    resid = h2df.clutch1 - X @ la.lstsq(X, h2df.clutch1, rcond=None)[0]
    print(f"overall OFF 1st half -> clutch 2nd half: r = "
          f"{h2df.overall1.corr(h2df.clutch2):+.3f}")
    print(f"CLUTCH RESIDUAL (clutch beyond overall) -> 2nd half: r = "
          f"{pd.Series(resid).corr(h2df.clutch2):+.3f}   <- the signal test\n")


def gate2_model() -> None:
    roll = rolling_clutch()
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()].copy()
    for side in ("home", "away"):
        done = done.merge(
            roll.rename(columns={
                "team": f"{side}_team",
                "clutch_off16": f"{side}_clutch_off16",
                "clutch_def16": f"{side}_clutch_def16"}),
            on=["season", "week", f"{side}_team"], how="left")
    done["clutch_net_diff"] = (
        (done.home_clutch_off16.fillna(0) - done.home_clutch_def16.fillna(0))
        - (done.away_clutch_off16.fillna(0) - done.away_clutch_def16.fillna(0)))
    done["margin"] = (done.home_score - done.away_score).abs()

    results = {}
    for name, feats in [("baseline", FEATURES),
                        ("+clutch", FEATURES + CLUTCH_FEATS)]:
        probs, idx = [], []
        for season in BACKTEST_SEASONS:
            tr, te = done[done.season < season], done[done.season == season]
            if te.empty:
                continue
            m = XGBClassifier(**PARAMS)
            m.fit(tr[feats], tr["home_win"])
            raw = m.predict_proba(te[feats])[:, 1]
            probs.append((1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].values)
            idx.append(te.index.values)
        p = np.concatenate(probs)
        sub = done.loc[np.concatenate(idx)]
        y = sub["home_win"].values
        hit = (p > .5) == (y == 1)
        one = sub["margin"] <= 8
        results[name] = (accuracy_score(y, p > .5), brier_score_loss(y, p),
                         hit[one.values].mean(), int(one.sum()))
        print(f"{name:9s} acc {results[name][0]:.4f}  brier {results[name][1]:.4f}  "
              f"one-score acc {results[name][2]:.4f} (n={results[name][3]})")


if __name__ == "__main__":
    gate1_persistence()
    gate2_model()
