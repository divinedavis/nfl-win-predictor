"""Field position, and whether WHEN something happens changes what it is worth.

Two questions, tested together because both need the same play-by-play sweep.

FIELD POSITION. Where an offence starts its drives is the hidden half of
football -- a team starting at its own 32 all afternoon is playing a different
game from one starting at its own 21. It is produced by punting, coverage,
returns and takeaways, none of which the model measures directly.

TIMING. The one feature this project has ever promoted is clutch, which works
precisely because it is situational: how a team plays in the last five minutes
of a one-score game is invisible in a season-long record. So the same question
is worth asking of the two traits that just failed. Defensive penalties and
positive plays are both real and both already priced as season totals -- but a
holding call on a fourth-quarter drive is not the same event as one in the
first, and a season average blends the two into nothing.

Every quarter is measured on its own terms here, as a rate, and rolled forward
so a game never sees itself. The earlier quarter test failed by measuring
quarters against a whole-season average that included future games; this one
measures each quarter directly and differences only rolled values.

    python field_and_timing_ablation.py
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
CACHE = "field_timing_team_weeks.csv"

FIELD_COLS = ["home_start8", "away_start8", "start8_diff"]
TIMING_COLS = ["home_q4_pen8", "away_q4_pen8", "q4_pen8_diff",
               "home_q4_succ8", "away_q4_succ8", "q4_succ8_diff"]


def build() -> pd.DataFrame:
    frames = []
    for season in range(FIRST, LAST_SEASON + 1):
        try:
            p = nfl.load_pbp([season]).select(
                ["season", "week", "posteam", "defteam", "epa", "success",
                 "play_type", "penalty", "penalty_team", "qtr", "fixed_drive",
                 "yardline_100", "down", "game_id"]).to_pandas()
        except Exception:
            continue

        scrim = p[p.play_type.isin(["run", "pass"]) & p.posteam.notna()]

        # --- where drives start ------------------------------------------
        # The first snap of each drive is where the offence took over.
        starts = (scrim[scrim.yardline_100.notna()]
                  .sort_values(["game_id", "fixed_drive"])
                  .groupby(["game_id", "fixed_drive", "season", "week",
                            "posteam"], as_index=False)
                  .yardline_100.first())
        fp = (starts.groupby(["season", "week", "posteam"])
              .agg(avg_start=("yardline_100", "mean"),
                   drives=("yardline_100", "size")).reset_index()
              .rename(columns={"posteam": "team"}))

        # --- positive plays, by quarter ------------------------------------
        s = scrim[scrim.success.notna()]
        succ = (s.assign(q4=s.qtr.eq(4))
                .groupby(["season", "week", "posteam", "q4"])
                .agg(rate=("success", "mean"), n=("success", "size"))
                .reset_index().rename(columns={"posteam": "team"}))
        succ = succ.pivot_table(index=["season", "week", "team"], columns="q4",
                                values=["rate", "n"]).reset_index()
        succ.columns = ["season", "week", "team", "succ_early", "succ_q4",
                        "n_early", "n_q4"]

        # --- defensive penalties, by quarter --------------------------------
        pen = p[p.penalty.fillna(0).gt(0) & p.penalty_team.notna()
                & p.defteam.notna()].copy()
        pen = pen[pen.penalty_team == pen.defteam]
        dsnaps = (scrim.assign(q4=scrim.qtr.eq(4))
                  .groupby(["season", "week", "defteam", "q4"])
                  .size().reset_index(name="snaps")
                  .rename(columns={"defteam": "team"}))
        dpen = (pen.assign(q4=pen.qtr.eq(4))
                .groupby(["season", "week", "penalty_team", "q4"])
                .size().reset_index(name="pens")
                .rename(columns={"penalty_team": "team"}))
        dd = dsnaps.merge(dpen, on=["season", "week", "team", "q4"],
                          how="left").fillna({"pens": 0})
        # Per snap, so a long fourth quarter does not look undisciplined.
        dd["pen_rate"] = dd.pens / dd.snaps.replace(0, np.nan)
        dd = dd.pivot_table(index=["season", "week", "team"], columns="q4",
                            values="pen_rate").reset_index()
        dd.columns = ["season", "week", "team", "pen_early", "pen_q4"]

        m = fp.merge(succ, on=["season", "week", "team"], how="outer")
        m = m.merge(dd, on=["season", "week", "team"], how="outer")
        frames.append(m)
        print(f"{season}: {len(starts):,} drives, {len(pen):,} defensive fouls")

    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv(CACHE, index=False)
    return out


def persistence(tw: pd.DataFrame) -> None:
    print("\n=== GATE 1: which of these are real traits? ===\n")
    cols = [("avg_start", "average drive start"),
            ("succ_q4", "positive plays, 4th quarter"),
            ("succ_early", "positive plays, 1st-3rd"),
            ("pen_q4", "defensive fouls per snap, 4th"),
            ("pen_early", "defensive fouls per snap, 1st-3rd")]
    rows = {c: [] for c, _ in cols}
    for (season, team), g in tw.groupby(["season", "team"]):
        g = g.sort_values("week")
        if len(g) < 12:
            continue
        half = len(g) // 2
        for c, _ in cols:
            a, b = g.iloc[:half][c].mean(), g.iloc[half:][c].mean()
            if np.isfinite(a) and np.isfinite(b):
                rows[c].append((a, b))
    for c, label in cols:
        d = pd.DataFrame(rows[c], columns=["a", "b"])
        r = d.a.corr(d.b)
        verdict = ("a real, repeatable trait" if r > 0.3 else
                   "weakly repeatable" if r > 0.15 else "mostly luck")
        print(f"   {label:36s} r = {r:+.3f}   {verdict}   (n={len(d)})")


def raw_effect(tw: pd.DataFrame, feat: pd.DataFrame) -> None:
    print("\n=== Does starting field position decide games? ===\n")
    res = []
    for side, opp in (("home", "away"), ("away", "home")):
        res.append(pd.DataFrame({
            "season": feat.season, "week": feat.week,
            "team": feat[f"{side}_team"], "opp": feat[f"{opp}_team"],
            "won": (feat.home_win if side == "home" else 1 - feat.home_win)}))
    r = pd.concat(res).dropna()
    m = r.merge(tw[["season", "week", "team", "avg_start"]],
                on=["season", "week", "team"], how="inner")
    m = m.merge(tw[["season", "week", "team", "avg_start"]]
                .rename(columns={"team": "opp", "avg_start": "opp_start"}),
                on=["season", "week", "opp"], how="inner")
    # yardline_100 is yards to the end zone, so LOWER is better field position.
    m["edge"] = m.opp_start - m.avg_start
    for lo, hi, label in ((8, 99, "starting 8+ yards closer than them"),
                          (3, 8, "3-8 yards closer"), (-3, 3, "about level"),
                          (-8, -3, "3-8 yards further back"),
                          (-99, -8, "8+ yards further back")):
        g = m[m.edge.between(lo, hi)]
        if len(g) >= 200:
            print(f"   {label:38s} won {g.won.mean():.1%}   (n={len(g):,})")
    print(f"\n   league average drive start: own "
          f"{100 - tw.avg_start.mean():.1f} yard line")


def attach(feat: pd.DataFrame, tw: pd.DataFrame) -> pd.DataFrame:
    tw = tw.sort_values(["team", "season", "week"]).copy()
    for src in ("avg_start", "succ_q4", "pen_q4"):
        tw[f"{src}_roll"] = (tw.groupby("team")[src]
                             .transform(lambda s: s.shift(1)
                                        .rolling(ROLL, min_periods=3).mean()))
    key = tw.groupby(["season", "week", "team"])[
        ["avg_start_roll", "succ_q4_roll", "pen_q4_roll"]].mean()

    out = feat.copy()
    for side, col in (("home", "home_team"), ("away", "away_team")):
        idx = pd.MultiIndex.from_arrays([out.season, out.week, out[col]])
        out[f"{side}_start8"] = key.avg_start_roll.reindex(idx).to_numpy()
        out[f"{side}_q4_succ8"] = key.succ_q4_roll.reindex(idx).to_numpy()
        out[f"{side}_q4_pen8"] = key.pen_q4_roll.reindex(idx).to_numpy()
    out["start8_diff"] = out.home_start8 - out.away_start8
    out["q4_succ8_diff"] = out.home_q4_succ8 - out.away_q4_succ8
    out["q4_pen8_diff"] = out.home_q4_pen8 - out.away_q4_pen8
    for c in FIELD_COLS + TIMING_COLS:
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

    variants = {
        "model as it is now": FEATURES,
        "+ average drive start": FEATURES + FIELD_COLS,
        "+ 4th-quarter timing": FEATURES + TIMING_COLS,
        "+ both": FEATURES + FIELD_COLS + TIMING_COLS,
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
    print(f"\n{len(tw):,} team-games, {int(tw.season.min())}-{int(tw.season.max())}")
    persistence(tw)
    raw_effect(tw, feat)
    gate2(attach(feat, tw))


if __name__ == "__main__":
    main()
