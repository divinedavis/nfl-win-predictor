"""Which of the 71 things the model looks at actually help, and which hurt?

Two questions that sound the same and are not:

  what does it LEAN on    Scramble one column so it becomes meaningless, and
                          see how much worse the predictions get. Cheap -- no
                          retraining, just re-predicting -- so every column can
                          be checked many times over. Answers "how much of the
                          answer comes from this".
  what actually HELPS     Delete the column and rebuild the model without it.
                          Expensive, but it is the only way to know whether the
                          model would be BETTER off without something, which
                          leaning cannot tell you: a column the model leans on
                          heavily can still be leading it astray.

Both are run several times with different random draws, because a single run
cannot tell a real effect from luck. Every number is printed next to how much
it wobbles, and anything smaller than its own wobble is marked as noise --
which, on this project, most things are.

A warning that matters. Checking 71 columns means 71 chances to be fooled: if
none of them mattered at all, three or four would still look like winners by
luck alone. Nothing here should be acted on because it appears at the top of
the list once. Treat it as a shortlist to test properly, not a verdict.

    python ablate.py --lean               # fast: what does it lean on
    python ablate.py --helps              # slow: what actually helps
    python ablate.py --helps --groups     # slow: by group rather than column
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES, POS_GROUPS
from train import PARAMS, ELO_BLEND, BACKTEST_SEASONS

SEEDS = [42, 7, 123]
SHUFFLES = 5          # re-scrambles per column, per seed

# Columns grouped the way a person thinks about them, for the slower test.
GROUPS = {
    "elo rating": ["elo_diff", "elo_home", "elo_away", "elo_prob"],
    "rest days": ["home_rest", "away_rest", "rest_diff"],
    "schedule": ["div_game", "is_dome", "week"],
    "weather": ["temp", "wind"],
    "injury counts": ["home_n_out", "away_n_out", "home_n_quest", "away_n_quest"],
    "quarterback change": ["home_qb_changed", "away_qb_changed"],
    "injuries by position": [f"{s}_{g}_out_wt" for s in ("home", "away")
                             for g in POS_GROUPS],
    "value of who is out": ([f"{s}_{g}_out_epa" for s in ("home", "away")
                             for g in ("qb", "rb", "wr", "te")]
                            + [f"{s}_{g}_out_val" for s in ("home", "away")
                               for g in ("dl", "lb", "db")]),
    "quarterback rating": ["home_qb_val", "away_qb_val", "qb_val_diff"],
    "recent form": ["home_pdiff8", "away_pdiff8", "pdiff8_diff",
                    "home_winrate8", "away_winrate8",
                    "home_pf8", "home_pa8", "away_pf8", "away_pa8"],
    "offence and defence": ["home_off_epa8", "away_off_epa8", "off_epa8_diff",
                            "home_def_epa8", "away_def_epa8", "def_epa8_diff"],
    "late-game play": ["home_clutch_off16", "home_clutch_def16",
                       "away_clutch_off16", "away_clutch_def16",
                       "clutch_net_diff"],
}


def fit_predict(done: pd.DataFrame, cols: list[str], seed: int):
    """Train on earlier seasons, predict the next, all the way through."""
    P, Y, X = [], [], []
    for season in BACKTEST_SEASONS:
        tr, te = done[done.season < season], done[done.season == season]
        if tr.empty or te.empty:
            continue
        m = XGBClassifier(**{**PARAMS, "random_state": seed})
        m.fit(tr[cols], tr["home_win"])
        raw = m.predict_proba(te[cols])[:, 1]
        P.append((1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].to_numpy())
        Y.append(te["home_win"].to_numpy())
        X.append((m, te))
    return np.concatenate(P), np.concatenate(Y), X


def baseline(done: pd.DataFrame) -> tuple[float, float]:
    scores = []
    for s in SEEDS:
        p, y, _ = fit_predict(done, FEATURES, s)
        scores.append(brier_score_loss(y, p))
    return float(np.mean(scores)), float(np.std(scores))


def lean(done: pd.DataFrame) -> pd.DataFrame:
    """Scramble each column and see how much worse the predictions get."""
    rows = []
    fitted = {s: fit_predict(done, FEATURES, s) for s in SEEDS}
    base = {s: brier_score_loss(fitted[s][1], fitted[s][0]) for s in SEEDS}

    for i, col in enumerate(FEATURES, 1):
        drops = []
        for s in SEEDS:
            rng = np.random.default_rng(1000 + s)
            for _ in range(SHUFFLES):
                P, Y = [], []
                for m, te in fitted[s][2]:
                    x = te[FEATURES].copy()
                    x[col] = rng.permutation(x[col].to_numpy())
                    raw = m.predict_proba(x)[:, 1]
                    P.append((1 - ELO_BLEND) * raw
                             + ELO_BLEND * te["elo_prob"].to_numpy())
                    Y.append(te["home_win"].to_numpy())
                drops.append(brier_score_loss(np.concatenate(Y),
                                              np.concatenate(P)) - base[s])
        rows.append({"column": col, "leans_on": float(np.mean(drops)),
                     "wobble": float(np.std(drops))})
        if i % 10 == 0:
            print(f"  {i}/{len(FEATURES)} columns", file=sys.stderr)
    return pd.DataFrame(rows).sort_values("leans_on", ascending=False)


def helps(done: pd.DataFrame, groups: bool) -> pd.DataFrame:
    """Remove each column (or group) and rebuild without it."""
    base_mean, base_wobble = baseline(done)
    print(f"  model with everything: brier {base_mean:.4f} "
          f"(wobbles +/-{base_wobble:.4f})", file=sys.stderr)

    targets = GROUPS if groups else {c: [c] for c in FEATURES}
    rows = []
    for i, (name, cols) in enumerate(targets.items(), 1):
        keep = [c for c in FEATURES if c not in cols]
        if not keep:
            continue
        scores = []
        for s in SEEDS:
            p, y, _ = fit_predict(done, keep, s)
            scores.append(brier_score_loss(y, p))
        # Positive = removing it made things worse, so it was helping.
        rows.append({"what": name, "n_cols": len(cols),
                     "helps_by": float(np.mean(scores)) - base_mean,
                     "wobble": float(np.std(scores))})
        print(f"  {i}/{len(targets)}  {name}", file=sys.stderr)
    return pd.DataFrame(rows).sort_values("helps_by", ascending=False)


def show(df: pd.DataFrame, value: str, label: str, top: int) -> None:
    print(f"\n   {'':34s} {label:>10} {'wobble':>9}   verdict")
    for _, r in df.head(top).iterrows():
        name = r.get("column", r.get("what"))
        v, w = r[value], r["wobble"]
        if w > 0 and abs(v) < w:
            verdict = "noise"
        elif v > 0:
            verdict = "helping" if value == "helps_by" else "leans on it"
        else:
            verdict = "HURTING -- drop it" if value == "helps_by" else "ignores it"
        print(f"   {name:34s} {v:+10.5f} {w:9.5f}   {verdict}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--helps", action="store_true")
    ap.add_argument("--groups", action="store_true")
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()

    feat = pd.read_parquet("features.parquet")
    done = feat[feat.home_win.notna()].copy()

    if args.lean or not args.helps:
        print("Scrambling each column to see what the model leans on...",
              file=sys.stderr)
        df = lean(done)
        df.to_csv("ablate_lean.csv", index=False)
        print("\n=== What the model leans on ===")
        print("   Bigger number = predictions get worse when this column is "
              "scrambled,\n   so more of the answer comes from it.")
        show(df, "leans_on", "leans on", args.top)
        print("\n   bottom of the list -- barely used at all:")
        show(df.tail(8).sort_values("leans_on"), "leans_on", "leans on", 8)

    if args.helps:
        print("Rebuilding the model without each piece...", file=sys.stderr)
        df = helps(done, args.groups)
        df.to_csv("ablate_helps.csv", index=False)
        print("\n=== What actually helps ===")
        print("   Plus = the model got worse without it, so it was helping.")
        print("   Minus = the model got BETTER without it, so it is hurting.")
        show(df, "helps_by", "helps by", args.top)

    print("\n   Reminder: 71 columns means 71 chances to be fooled. If none of "
          "them\n   mattered, three or four would still look like winners by "
          "luck. Anything\n   here is a shortlist to test properly, not a "
          "verdict.")


if __name__ == "__main__":
    main()
