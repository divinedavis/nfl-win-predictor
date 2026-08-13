"""Ablation backtest for candidate upset-indicator feature groups.

Runs the same walk-forward protocol as train.py (train on all prior seasons,
test on each season 2015-2025) once with the baseline FEATURES, once per
candidate group added, and once with all groups — so each group's marginal
value is measured under identical conditions. Reports blended (production)
metrics and XGBoost-only metrics; the blend's 60% Elo share mutes feature
effects, so xgb-only Brier is the sharper signal of whether a group helps.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURES, CANDIDATE_GROUPS
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND


def backtest(done: pd.DataFrame, feats: list) -> dict:
    probs, raws, ys = [], [], []
    for season in BACKTEST_SEASONS:
        train = done[done.season < season]
        test = done[done.season == season]
        if test.empty:
            continue
        model = XGBClassifier(**PARAMS)
        model.fit(train[feats], train["home_win"])
        raw = model.predict_proba(test[feats])[:, 1]
        raws.append(raw)
        probs.append((1 - ELO_BLEND) * raw + ELO_BLEND * test["elo_prob"].values)
        ys.append(test["home_win"].values)
    y = np.concatenate(ys)
    prob = np.concatenate(probs)
    raw = np.concatenate(raws)
    return {
        "games": len(y),
        "acc": accuracy_score(y, prob > 0.5),
        "brier": brier_score_loss(y, prob),
        "logloss": log_loss(y, prob),
        "xgb_acc": accuracy_score(y, raw > 0.5),
        "xgb_brier": brier_score_loss(y, raw),
    }


def main() -> None:
    df = pd.read_parquet("features.parquet")
    done = df[df["home_win"].notna()].copy()

    configs = {"baseline": FEATURES}
    for name, cols in CANDIDATE_GROUPS.items():
        configs[f"+{name}"] = FEATURES + cols
    configs["+all"] = FEATURES + [c for cols in CANDIDATE_GROUPS.values()
                                  for c in cols]

    rows = []
    base = None
    for name, feats in configs.items():
        r = backtest(done, feats)
        if name == "baseline":
            base = r
        rows.append({
            "config": name, "n_feats": len(feats), **r,
            "d_acc": r["acc"] - base["acc"],
            "d_brier": r["brier"] - base["brier"],
            "d_xgb_brier": r["xgb_brier"] - base["xgb_brier"],
        })
        print(f"{name:12s} acc {r['acc']:.4f}  brier {r['brier']:.4f}  "
              f"xgb_acc {r['xgb_acc']:.4f}  xgb_brier {r['xgb_brier']:.4f}")

    res = pd.DataFrame(rows)
    print("\n=== Ablation over", base["games"], "games (2015-2025) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:+.4f}"
                        if abs(x) < 1 else f"{x:.4f}"))


if __name__ == "__main__":
    main()
