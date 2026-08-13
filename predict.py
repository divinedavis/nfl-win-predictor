"""Print win probabilities for a week of NFL games.

Usage:
    python predict.py                 # next week with unplayed games
    python predict.py --week 3        # specific week of the default season
    python predict.py --season 2026 --week 1

Run features.py first (fresh data), then train.py (fresh model), then this.
"""

import argparse

import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES, LAST_SEASON
from train import ELO_BLEND


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=LAST_SEASON)
    ap.add_argument("--week", type=int, default=None)
    args = ap.parse_args()

    df = pd.read_parquet("features.parquet")
    season_df = df[df.season == args.season]
    if season_df.empty:
        raise SystemExit(f"No games found for season {args.season}")

    week = args.week
    if week is None:
        unplayed = season_df[season_df.home_score.isna()]
        if unplayed.empty:
            raise SystemExit(f"Season {args.season} is complete; pass --week")
        week = int(unplayed.week.min())

    games = season_df[season_df.week == week].copy()
    if games.empty:
        raise SystemExit(f"No games for season {args.season} week {week}")

    model = XGBClassifier()
    model.load_model("model.json")
    raw = model.predict_proba(games[FEATURES])[:, 1]
    games["home_prob"] = (1 - ELO_BLEND) * raw + ELO_BLEND * games["elo_prob"]
    games["pick"] = games.apply(
        lambda r: r.home_team if r.home_prob >= 0.5 else r.away_team, axis=1
    )
    games["confidence"] = games.home_prob.where(
        games.home_prob >= 0.5, 1 - games.home_prob
    )
    games = games.sort_values("confidence", ascending=False)

    print(f"\nNFL {args.season} — Week {week} picks "
          f"(model trained through last completed week)\n")
    for r in games.itertuples(index=False):
        date = pd.Timestamp(r.gameday).strftime("%a %m/%d")
        line = (f"  {date}  {r.away_team:>3} @ {r.home_team:<3}  "
                f"pick: {r.pick:<3} {r.confidence:.0%}  "
                f"(home {r.home_prob:.0%}, elo {r.elo_prob:.0%})")
        if pd.notna(r.home_score):
            actual = r.home_team if r.home_score > r.away_score else r.away_team
            hit = "HIT " if actual == r.pick else "MISS"
            line += f"  final {int(r.away_score)}-{int(r.home_score)} {hit}"
        print(line)

    played = games[games.home_score.notna()]
    if len(played):
        correct = sum(
            (r.home_team if r.home_score > r.away_score else r.away_team) == r.pick
            for r in played.itertuples(index=False)
        )
        print(f"\n  Week record: {correct}-{len(played) - correct}")


if __name__ == "__main__":
    main()
