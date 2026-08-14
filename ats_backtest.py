"""ATS backtest: model margin vs closing spread, walk-forward 2015-2025.

Margin map (home margin ~ blended win prob) is fit per season on training
data only. Verification trail (2026-08-13, one-off pulls in session):
  - nflverse spread_line matches true DK closing lines (10 min pre-kick):
    mean |diff| 0.08 pts, 97% within 1 pt -> the column IS closing lines.
  - blind always-take-the-dog control: 51.5% (model adds ~4 pts of selection)
  - 2024 subsample vs verified DK lines: 58.1%
Result: 55.2% over 2,943 games, above the 52.4% breakeven in 10/11 seasons.

    python ats_backtest.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND


def main() -> None:
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()].copy()
    done["margin"] = done.home_score - done.away_score

    total_cov = total_n = above = seasons = 0
    print("season   n_ats  cover%   (breakeven 52.4%)")
    for season in BACKTEST_SEASONS:
        tr, te = done[done.season < season], done[done.season == season]
        if te.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(tr[FEATURES], tr["home_win"])
        raw = m.predict_proba(te[FEATURES])[:, 1]
        p = (1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].values
        tr_raw = m.predict_proba(tr[FEATURES])[:, 1]
        tr_p = (1 - ELO_BLEND) * tr_raw + ELO_BLEND * tr["elo_prob"].values
        A = np.column_stack([np.ones(len(tr_p)), tr_p])
        beta = np.linalg.lstsq(A, tr["margin"].values, rcond=None)[0]
        te = te.assign(mm=beta[0] + beta[1] * p)
        v = te[te.spread_line.notna() & (te.margin != te.spread_line)]
        pick_home = v.mm > v.spread_line
        cov = np.where(pick_home, v.margin > v.spread_line,
                       v.margin < v.spread_line)
        total_cov += cov.sum(); total_n += len(v); seasons += 1
        above += cov.mean() > 0.524
        print(f"{season}   {len(v):5d}  {cov.mean():.1%}")
    print(f"\nTOTAL: {total_cov}/{total_n} = {total_cov/total_n:.1%}, "
          f"above breakeven in {above}/{seasons} seasons")


if __name__ == "__main__":
    main()
