"""Three-gate test of coaching effects.

Residual = (won game) - (pregame Elo win probability): what a coach's team
did beyond what team strength predicted. Rolled per coach (32 games) and per
ordered coach-pair (head-to-head history), always shifted/leak-free.

Gate 1a  coach skill: split-half persistence of career residual per coach
Gate 1b  matchup effect: does prior H2H residual vs THIS opponent coach
         predict the next meeting, beyond Elo?
Gate 2   model: walk-forward backtest with coach features, per-season Brier.

    python coach_ablation.py
"""

import numpy as np
import pandas as pd
import nflreadpy as nfl
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND

COACH_ROLL = 32
COACH_FEATS = ["home_coach_resid32", "away_coach_resid32", "coach_resid_diff",
               "h2h_prior_resid", "h2h_meetings"]


def build_coach_table() -> pd.DataFrame:
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()].copy()
    sched = nfl.load_schedules(list(range(1999, 2027))).to_pandas()[
        ["game_id", "home_coach", "away_coach"]]
    done = done.merge(sched, on="game_id", how="left")
    done = done[done.home_coach.notna() & done.away_coach.notna()]
    done = done.sort_values(["gameday", "game_id"]).reset_index(drop=True)

    coach_hist: dict = {}     # coach -> list of residuals (newest last)
    pair_hist: dict = {}      # (coach, opp_coach) -> [residuals]
    rows = []
    for r in done.itertuples(index=False):
        hc, ac, p = r.home_coach, r.away_coach, r.elo_prob
        h_res = coach_hist.get(hc, [])
        a_res = coach_hist.get(ac, [])
        h2h = pair_hist.get((hc, ac), [])
        rows.append({
            "home_coach_resid32": np.mean(h_res[-COACH_ROLL:]) if len(h_res) >= 8 else np.nan,
            "away_coach_resid32": np.mean(a_res[-COACH_ROLL:]) if len(a_res) >= 8 else np.nan,
            "h2h_prior_resid": float(np.sum(h2h)) if h2h else 0.0,
            "h2h_meetings": len(h2h),
            "home_games": len(h_res), "away_games": len(a_res),
        })
        # update AFTER recording pregame values (leak-free)
        home_resid = float(r.home_win) - p
        coach_hist.setdefault(hc, []).append(home_resid)
        coach_hist.setdefault(ac, []).append(-home_resid)
        pair_hist.setdefault((hc, ac), []).append(home_resid)
        pair_hist.setdefault((ac, hc), []).append(-home_resid)
    extra = pd.DataFrame(rows, index=done.index)
    out = pd.concat([done, extra], axis=1)
    out["coach_resid_diff"] = (out.home_coach_resid32.fillna(0)
                               - out.away_coach_resid32.fillna(0))
    return out, coach_hist


def main() -> None:
    done, coach_hist = build_coach_table()

    # --- Gate 1a: coach skill persistence (split-half) ---
    per = []
    for coach, res in coach_hist.items():
        if len(res) < 64:
            continue
        res = np.array(res)
        per.append({"coach": coach, "odd": res[::2].mean(),
                    "even": res[1::2].mean(), "n": len(res)})
    per = pd.DataFrame(per).sort_values("even", ascending=False)
    print(f"=== Gate 1a: coach skill persistence ({len(per)} coaches, 64+ games) ===")
    print(f"odd-game residual vs even-game residual: r = {per.odd.corr(per.even):+.3f}")
    full = per.assign(career=(per.odd + per.even) / 2).sort_values("career", ascending=False)
    print("best career resid/game:",
          ", ".join(f"{r.coach} {r.career:+.3f}" for r in full.head(3).itertuples()))
    print("worst:",
          ", ".join(f"{r.coach} {r.career:+.3f}" for r in full.tail(3).itertuples()), "\n")

    # --- Gate 1b: H2H matchup effect beyond coach skill ---
    m = done[(done.h2h_meetings >= 3)
             & done.home_coach_resid32.notna() & done.away_coach_resid32.notna()].copy()
    m["h2h_rate"] = m.h2h_prior_resid / m.h2h_meetings
    m["outcome_resid"] = m.home_win - m.elo_prob
    r_h2h = m.h2h_rate.corr(m.outcome_resid)
    # partial: residualize h2h_rate on the two coaches' general skill first
    X = np.column_stack([np.ones(len(m)), m.home_coach_resid32, m.away_coach_resid32])
    beta = np.linalg.lstsq(X, m.h2h_rate, rcond=None)[0]
    h2h_specific = m.h2h_rate - X @ beta
    r_partial = pd.Series(h2h_specific.values).corr(m.outcome_resid.reset_index(drop=True))
    print(f"=== Gate 1b: head-to-head effect ({len(m)} meetings w/ 3+ prior) ===")
    print(f"prior H2H rate -> next-meeting residual:              r = {r_h2h:+.3f}")
    print(f"H2H beyond both coaches' general skill (the matchup): r = {r_partial:+.3f}\n")

    # --- Gate 2: model backtest ---
    print("=== Gate 2: walk-forward backtest ===")
    results = {}
    for name, feats in [("baseline", FEATURES),
                        ("+coach", FEATURES + COACH_FEATS)]:
        probs, idx, per_season = [], [], {}
        for season in BACKTEST_SEASONS:
            tr, te = done[done.season < season], done[done.season == season]
            if te.empty:
                continue
            mdl = XGBClassifier(**PARAMS)
            mdl.fit(tr[feats], tr["home_win"])
            raw = mdl.predict_proba(te[feats])[:, 1]
            p = (1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].values
            probs.append(p); idx.append(te.index.values)
            per_season[season] = brier_score_loss(te["home_win"].values, p)
        p = np.concatenate(probs)
        s = done.loc[np.concatenate(idx)]
        y = s.home_win.values
        results[name] = per_season
        print(f"{name:9s} acc {accuracy_score(y, p > .5):.4f}  "
              f"brier {brier_score_loss(y, p):.4f}")
    deltas = [results["+coach"][s] - results["baseline"][s]
              for s in results["baseline"]]
    print(f"coach features better (brier) in {sum(d < 0 for d in deltas)}/{len(deltas)} seasons")


if __name__ == "__main__":
    main()
