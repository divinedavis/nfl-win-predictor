"""Which position's injuries actually move win probability?

Counterfactual on the trained model: for each position group, take every
2015-2025 game, set the home team's weighted-outs at that group to "one
full-time player out" (+1.0) vs fully healthy (0), and measure the average
drop in predicted home win probability. Model trained walk-forward as usual.
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES, POS_GROUPS
from train import ELO_BLEND, PARAMS


def main() -> None:
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()]
    train = done[done.season < 2020]
    test = done[(done.season >= 2020) & (done.season <= 2025)].copy()

    model = XGBClassifier(**PARAMS)
    model.fit(train[FEATURES], train["home_win"])

    print("Win-probability cost of ONE full-time player ruled Out "
          "(counterfactual, 2020-2025 games):\n")
    impacts = {}
    for grp in POS_GROUPS:
        healthy, hurt = test.copy(), test.copy()
        healthy[f"home_{grp}_out_wt"] = 0.0
        hurt[f"home_{grp}_out_wt"] = 1.0
        p0 = model.predict_proba(healthy[FEATURES])[:, 1]
        p1 = model.predict_proba(hurt[FEATURES])[:, 1]
        # Only the XGBoost share of the blend responds to these features.
        impacts[grp] = (1 - ELO_BLEND) * float(np.mean(p1 - p0))

    for grp, delta in sorted(impacts.items(), key=lambda kv: kv[1]):
        bar = "#" * max(1, int(abs(delta) * 400))
        print(f"  {grp.upper():>3}  {delta:+.1%}  {bar}")

    print("\nNote: player impact here = snap share x position sensitivity. "
          "A 100%-snap QB out moves the needle by the QB row; a 40%-snap "
          "rotational DL moves 40% of the DL row. Skill differences between "
          "players (elite vs average starter) are not modeled yet.")


if __name__ == "__main__":
    main()
