"""Per-team-game third-down performance, adjusted for how far there was to go.

Raw third-down rates are misleading. Converting 60% of third downs is ordinary
if they are all third-and-one and remarkable if they are all third-and-eight,
and a defence that keeps offences in third-and-long will look worse on the raw
number than one that gives up first downs on first and second down. League
rates make the point:

    3rd and 1-3   converted 59.6%
    3rd and 4-6   converted 45.0%
    3rd and 7+    converted 25.0%

So each third down is scored against what a league-average team would do from
that distance, and a team's number is the total it beat or missed that
expectation by. A defence at -2.0 for a game stopped two more third downs than
the distances it faced would predict.

Distance expectations are measured once over the whole window rather than per
season, so a team is not judged against a bar that its own results moved.

    python build_third_down.py [--first 2006] [--update]
"""

import argparse

import numpy as np
import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon

# Distances beyond this are lumped together; third-and-25 and third-and-30 are
# the same proposition and splitting them just thins the sample.
MAX_DISTANCE = 20


def expected_rates(df: pd.DataFrame) -> pd.Series:
    """League conversion rate at each distance to go."""
    d = df.copy()
    d["togo"] = d.ydstogo.clip(1, MAX_DISTANCE)
    return d.groupby("togo").converted.mean()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=2006)
    ap.add_argument("--update", action="store_true",
                    help="refresh only the current season into the cache")
    args = ap.parse_args()

    cached, first = None, args.first
    if args.update:
        try:
            cached = pd.read_csv("third_down_team_weeks.csv")
            first = LAST_SEASON
        except FileNotFoundError:
            pass

    plays = []
    for season in range(first, LAST_SEASON + 1):
        try:
            pbp = nfl.load_pbp([season]).select(
                ["season", "week", "posteam", "defteam", "down", "ydstogo",
                 "play_type", "third_down_converted", "epa"]).to_pandas()
        except Exception:
            continue  # season not published yet
        t = pbp[pbp.down.eq(3) & pbp.play_type.isin(["run", "pass"])
                & pbp.posteam.notna() & pbp.ydstogo.notna()].copy()
        t["converted"] = t.third_down_converted.fillna(0)
        plays.append(t)
        print(f"{season}: {len(t):,} third downs")

    if not plays:
        raise SystemExit("no play-by-play available")
    df = pd.concat(plays, ignore_index=True)

    # What a league-average team does from each distance.
    rates = expected_rates(df)
    df["expected"] = df.ydstogo.clip(1, MAX_DISTANCE).map(rates)
    df["above"] = df.converted - df.expected

    # Offence: how much better than expected the team converted.
    off = (df.groupby(["season", "week", "posteam"])
           .agg(off_3d_att=("converted", "size"),
                off_3d_above=("above", "sum"),
                off_3d_epa=("epa", "sum"))
           .reset_index().rename(columns={"posteam": "team"}))
    # Defence: the same sum from the other side, negated, so that a positive
    # number always means the team did well.
    dfn = (df.groupby(["season", "week", "defteam"])
           .agg(def_3d_att=("converted", "size"),
                def_3d_above=("above", "sum"),
                def_3d_epa=("epa", "sum"))
           .reset_index().rename(columns={"defteam": "team"}))
    dfn["def_3d_above"] = -dfn["def_3d_above"]
    dfn["def_3d_epa"] = -dfn["def_3d_epa"]

    out = off.merge(dfn, on=["season", "week", "team"], how="outer").fillna(0)
    out["team"] = out["team"].map(canon)
    if cached is not None:
        out = pd.concat([cached[cached.season < LAST_SEASON], out],
                        ignore_index=True)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv("third_down_team_weeks.csv", index=False)

    print(f"\nWrote third_down_team_weeks.csv ({len(out):,} team-games, "
          f"{int(out.season.min())}-{int(out.season.max())})")
    print("\nExpected conversion by distance to go:")
    for togo in (1, 2, 3, 5, 7, 10, 15):
        if togo in rates.index:
            print(f"   3rd and {togo:<2d}  {rates[togo]:.1%}")
    best = out.groupby(["season", "team"]).def_3d_above.sum().nlargest(3)
    print("\nBest third-down defences on record (stops above expectation):")
    for (season, team), v in best.items():
        print(f"   {int(season)} {team}  +{v:.1f}")


if __name__ == "__main__":
    main()
