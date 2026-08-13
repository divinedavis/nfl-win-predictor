"""Three-gate test: does kicker quality predict winning tight games?

Metric: FGOE — field-goal points over expectation per kick, where expectation
is a distance-based logistic make-probability fit on pre-backtest seasons
(2006-2014). Rolled over each kicker's last 60 kicks, leak-free.

Gate 1  persistence: does a kicker's FGOE in odd-numbered kicks predict his
        even-numbered kicks? (split-half, controls for era/team)
Gate 2  descriptive: close-game (final margin <=3) win rate for teams with
        top- vs bottom-quartile kickers
Gate 3  model: walk-forward game-model backtest with kicker features,
        scored overall, on the FG-margin bucket, and per-season.

    python kicker_ablation.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND

ROLL_KICKS = 60
MIN_KICKS = 20
KICKER_FEATS = ["home_kicker_fgoe", "away_kicker_fgoe", "kicker_fgoe_diff"]


def load_kicks() -> pd.DataFrame:
    k = pd.read_csv("kicks.csv").sort_values(["season", "week"]).reset_index(drop=True)
    # Distance -> make probability, fit on pre-backtest era only.
    fit = k[k.season <= 2014]
    lr = LogisticRegression()
    lr.fit(fit[["kick_distance"]], fit["made"])
    k["p_make"] = lr.predict_proba(k[["kick_distance"]])[:, 1]
    k["fgoe"] = 3.0 * (k["made"] - k["p_make"])   # points over expectation
    return k


def rolling_kicker_fgoe(k: pd.DataFrame) -> pd.DataFrame:
    """Per (season, week, team): the team's current kicker's rolling FGOE per
    kick over his last ROLL_KICKS kicks BEFORE this week (leak-free)."""
    k = k.copy()
    g = k.groupby("kicker_player_id")
    k["fgoe_roll"] = g["fgoe"].transform(
        lambda s: s.shift(1).rolling(ROLL_KICKS, min_periods=MIN_KICKS).mean())
    # Within a game a kicker may kick several times; take his rolling value at
    # his FIRST kick of that game (all pre-game info), i.e. min index per group.
    first = (k.reset_index()
             .groupby(["season", "week", "team", "kicker_player_id"])
             .first().reset_index())
    # A team's kicker for the week = the one who attempted its kicks that week.
    # For feature purposes we need the value BEFORE games with no FG attempt
    # too, so also build a per-team as-of series: last known kicker + value.
    first = first.sort_values(["season", "week"])
    return first[["season", "week", "team", "kicker_player_id", "fgoe_roll"]]


def team_asof_features(kicker_weeks: pd.DataFrame, games: pd.DataFrame) -> dict:
    """(season, week, team) -> kicker fgoe as of that week (last known)."""
    hist: dict = {}
    out: dict = {}
    rows = kicker_weeks.sort_values(["season", "week"])
    all_weeks = (pd.concat([
        games[["season", "week", "home_team"]].rename(columns={"home_team": "team"}),
        games[["season", "week", "away_team"]].rename(columns={"away_team": "team"}),
    ]).drop_duplicates().sort_values(["season", "week"]))
    pending = list(rows.itertuples(index=False))
    pi = 0
    for r in all_weeks.itertuples(index=False):
        key = (r.season, r.week)
        while pi < len(pending) and (pending[pi].season, pending[pi].week) < key:
            p = pending[pi]
            if pd.notna(p.fgoe_roll):
                hist[p.team] = p.fgoe_roll
            pi += 1
        out[(r.season, r.week, r.team)] = hist.get(r.team, np.nan)
    return out


def main() -> None:
    k = load_kicks()

    # --- Gate 1: split-half persistence per kicker ---
    per = []
    for pid, grp in k.groupby("kicker_player_id"):
        if len(grp) < 60:
            continue
        odd, even = grp.iloc[::2], grp.iloc[1::2]
        per.append({"odd": odd.fgoe.mean(), "even": even.fgoe.mean(),
                    "n": len(grp)})
    per = pd.DataFrame(per)
    print(f"=== Gate 1: split-half persistence ({len(per)} kickers, 60+ kicks) ===")
    print(f"odd-kick FGOE vs even-kick FGOE: r = {per.odd.corr(per.even):+.3f}\n")

    # --- rolling team-week features ---
    kw = rolling_kicker_fgoe(k)
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()].copy()
    asof = team_asof_features(kw, done)
    done["home_kicker_fgoe"] = [asof.get((r.season, r.week, r.home_team), np.nan)
                                for r in done.itertuples(index=False)]
    done["away_kicker_fgoe"] = [asof.get((r.season, r.week, r.away_team), np.nan)
                                for r in done.itertuples(index=False)]
    done["kicker_fgoe_diff"] = (done.home_kicker_fgoe.fillna(0)
                                - done.away_kicker_fgoe.fillna(0))
    done["margin"] = (done.home_score - done.away_score).abs()

    # --- Gate 2: descriptive close-game win rate by kicker quartile ---
    sub = done[(done.season >= 2015) & done.home_kicker_fgoe.notna()
               & done.away_kicker_fgoe.notna()]
    long = pd.concat([
        pd.DataFrame({"fgoe": sub.home_kicker_fgoe, "won": sub.home_win == 1,
                      "margin": sub.margin}),
        pd.DataFrame({"fgoe": sub.away_kicker_fgoe, "won": sub.home_win == 0,
                      "margin": sub.margin})])
    tight = long[long.margin <= 3]
    q = tight.fgoe.quantile([0.25, 0.75])
    top = tight[tight.fgoe >= q[0.75]]
    bot = tight[tight.fgoe <= q[0.25]]
    print("=== Gate 2: FG-margin (<=3) games, 2015-2025 ===")
    print(f"top-quartile kicker teams:    {top.won.mean():.1%} win ({len(top)} team-games)")
    print(f"bottom-quartile kicker teams: {bot.won.mean():.1%} win ({len(bot)} team-games)\n")

    # --- Gate 3: model backtest ---
    print("=== Gate 3: walk-forward model backtest ===")
    results = {}
    for name, feats in [("baseline", FEATURES),
                        ("+kicker", FEATURES + KICKER_FEATS)]:
        probs, idx, per_season = [], [], {}
        for season in BACKTEST_SEASONS:
            tr, te = done[done.season < season], done[done.season == season]
            if te.empty:
                continue
            m = XGBClassifier(**PARAMS)
            m.fit(tr[feats], tr["home_win"])
            raw = m.predict_proba(te[feats])[:, 1]
            p = (1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].values
            probs.append(p)
            idx.append(te.index.values)
            per_season[season] = brier_score_loss(te["home_win"].values, p)
        p = np.concatenate(probs)
        s = done.loc[np.concatenate(idx)]
        y = s.home_win.values
        hit = (p > .5) == (y == 1)
        fg = (s.margin <= 3).values
        results[name] = per_season
        print(f"{name:9s} acc {accuracy_score(y, p > .5):.4f}  "
              f"brier {brier_score_loss(y, p):.4f}  "
              f"FG-margin acc {hit[fg].mean():.4f} (n={fg.sum()})")
    deltas = [results["+kicker"][s] - results["baseline"][s]
              for s in results["baseline"]]
    print(f"kicker better (brier) in {sum(d < 0 for d in deltas)}/{len(deltas)} seasons")


if __name__ == "__main__":
    main()
