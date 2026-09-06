"""Walk-forward backtest + final model for FBS games: cfb_model.json.

Same shape as the NFL trainer: for each test season, train on every prior
season and score that season blind. The Elo blend weight is swept on Brier
each run and written to cfb_model_meta.json so the exporter uses the number
the backtest actually chose rather than a constant someone typed.

    python -m cfb.train
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from cfb.features import FEATURES

PARAMS = dict(
    n_estimators=400, learning_rate=0.03, max_depth=3, min_child_weight=10,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
    objective="binary:logistic", eval_metric="logloss", tree_method="hist",
    random_state=42,
)
BACKTEST_SEASONS = range(2015, 2026)
BLENDS = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
MODEL = "cfb_model.json"
META = "cfb_model_meta.json"


def load_blend() -> float:
    try:
        return float(json.load(open(META))["elo_blend"])
    except Exception:
        return 0.4


def main() -> None:
    df = pd.read_parquet("cfb_features.parquet")
    done = df[df["home_win"].notna()].copy()

    raws, elos, ys, seasons, spreads = [], [], [], [], []
    for season in BACKTEST_SEASONS:
        train = done[done.season < season]
        test = done[done.season == season]
        if test.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(train[FEATURES], train["home_win"])
        raws.append(m.predict_proba(test[FEATURES])[:, 1])
        elos.append(test["elo_prob"].values)
        ys.append(test["home_win"].values)
        seasons.append(np.full(len(test), season))
        spreads.append(test["spread_line"].values)
    raw, elo, y = np.concatenate(raws), np.concatenate(elos), np.concatenate(ys)
    season_v, spread = np.concatenate(seasons), np.concatenate(spreads)

    print("\n=== Elo blend sweep (walk-forward 2015-2025) ===")
    best = None
    for b in BLENDS:
        p = (1 - b) * raw + b * elo
        brier = brier_score_loss(y, p)
        print(f"  blend {b:.1f}: acc {accuracy_score(y, p > 0.5):.4f}  brier {brier:.4f}")
        if best is None or brier < best[1]:
            best = (b, brier)
    blend = best[0]
    p = (1 - blend) * raw + blend * elo

    rows = []
    for s in sorted(set(season_v)):
        k = season_v == s
        has = ~np.isnan(spread[k]) & (spread[k] != 0)
        rows.append({
            "season": int(s), "games": int(k.sum()),
            "model_acc": accuracy_score(y[k], p[k] > 0.5),
            "elo_acc": accuracy_score(y[k], elo[k] > 0.5),
            "home_acc": y[k].mean(),
            "vegas_acc": accuracy_score(y[k][has], (spread[k][has] > 0)) if has.any() else np.nan,
            "vegas_n": int(has.sum()),
            "model_brier": brier_score_loss(y[k], p[k]),
            "elo_brier": brier_score_loss(y[k], elo[k]),
        })
    res = pd.DataFrame(rows)
    print(f"\n=== Walk-forward backtest, blend {blend:.1f} ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    w = res.games
    print("\n=== Weighted over", int(w.sum()), "games ===")
    for c in ["model_acc", "elo_acc", "home_acc", "model_brier", "elo_brier"]:
        print(f"  {c:12s} {np.average(res[c], weights=w):.4f}")
    hv = res.vegas_n > 0
    print(f"  vegas_acc    {np.average(res.vegas_acc[hv], weights=res.vegas_n[hv]):.4f} "
          f"(on {int(res.vegas_n.sum())} games with a spread)")
    # Confidence tiers, the page's star threshold.
    conf = np.maximum(p, 1 - p)
    for t in (0.65, 0.75, 0.85):
        k = conf >= t
        print(f"  conf>={t:.2f}: {k.mean():.1%} of games, acc {accuracy_score(y[k], p[k] > 0.5):.4f}")

    final = XGBClassifier(**PARAMS)
    final.fit(done[FEATURES], done["home_win"])
    final.save_model(MODEL)
    tiers = {str(t): float(accuracy_score(y[conf >= t], p[conf >= t] > 0.5))
             for t in (0.65, 0.75, 0.85)}
    json.dump({"elo_blend": blend, "backtest_acc": float(accuracy_score(y, p > 0.5)),
               "backtest_brier": float(best[1]), "tier_acc": tiers,
               "games": int(len(y)), "seasons": "2015-2025",
               "features": FEATURES}, open(META, "w"), indent=1)
    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== Feature importance ===")
    print(imp.head(10).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved {MODEL} + {META} (trained on {len(done)} games)")


if __name__ == "__main__":
    main()
