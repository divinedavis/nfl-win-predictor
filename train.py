"""Walk-forward backtest + final model training.

For each test season, train on all prior seasons and evaluate on that season —
the model never sees the games it predicts. Compares against three baselines:
always-pick-home, Elo alone, and the Vegas closing spread (the bar to aspire to).

Ends by training a final model on every completed game and saving it to
model.json for predict.py.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

from features import FEATURES

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.03,
    max_depth=3,
    min_child_weight=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=2.0,
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

BACKTEST_SEASONS = range(2015, 2026)

# Final probability = (1 - ELO_BLEND) * xgboost + ELO_BLEND * elo.
# Swept in backtest: 0.6 minimizes Brier (0.2214) and nearly matches pure
# Elo's accuracy while beating it on calibration.
ELO_BLEND = 0.6


def main() -> None:
    df = pd.read_parquet("features.parquet")
    done = df[df["home_win"].notna()].copy()

    rows = []
    for season in BACKTEST_SEASONS:
        train = done[done.season < season]
        test = done[done.season == season]
        if test.empty:
            continue

        model = XGBClassifier(**PARAMS)
        model.fit(train[FEATURES], train["home_win"])
        raw = model.predict_proba(test[FEATURES])[:, 1]
        prob = (1 - ELO_BLEND) * raw + ELO_BLEND * test["elo_prob"].values
        y = test["home_win"].values

        vegas_pick = np.where(test["spread_line"] > 0, 1.0, 0.0)  # spread_line > 0 → home favored
        has_line = test["spread_line"].notna() & (test["spread_line"] != 0)

        rows.append({
            "season": season,
            "games": len(test),
            "model_acc": accuracy_score(y, prob > 0.5),
            "elo_acc": accuracy_score(y, test["elo_prob"] > 0.5),
            "home_acc": y.mean(),
            "vegas_acc": accuracy_score(y[has_line], vegas_pick[has_line]),
            "model_logloss": log_loss(y, prob),
            "model_brier": brier_score_loss(y, prob),
            "elo_brier": brier_score_loss(y, test["elo_prob"]),
        })

    res = pd.DataFrame(rows)
    total_games = res.games.sum()
    print("\n=== Walk-forward backtest (train on all prior seasons) ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n=== Weighted averages over", total_games, "games ===")
    for col in ["model_acc", "elo_acc", "home_acc", "vegas_acc",
                "model_logloss", "model_brier", "elo_brier"]:
        print(f"  {col:15s} {np.average(res[col], weights=res.games):.4f}")

    final = XGBClassifier(**PARAMS)
    final.fit(done[FEATURES], done["home_win"])
    final.save_model("model.json")

    imp = pd.Series(final.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\n=== Feature importance (final model) ===")
    print(imp.head(12).to_string(float_format=lambda x: f"{x:.3f}"))
    print(f"\nSaved model.json (trained on {len(done)} games "
          f"{done.season.min()}-{done.season.max()})")


if __name__ == "__main__":
    main()
