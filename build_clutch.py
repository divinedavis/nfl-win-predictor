"""Build per-team-game 'finishing' aggregates from play-by-play: EPA per play
on offense and defense in the last 5 minutes of one-score games. Cached to
clutch_team_weeks.csv so the heavy pbp download happens once.

    python build_clutch.py [--first 2006]
"""

import argparse

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon

CLUTCH_SECONDS = 300      # last 5 minutes of the game
CLUTCH_MARGIN = 8         # one-score game at the time of the snap


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=2006)
    ap.add_argument("--update", action="store_true",
                    help="refresh only the current season into the cache "
                         "(daily droplet mode — history is static)")
    args = ap.parse_args()

    frames = []
    first = args.first
    if args.update:
        try:
            prev = pd.read_csv("clutch_team_weeks.csv")
            frames.append(prev[prev.season < LAST_SEASON])
            first = LAST_SEASON
        except FileNotFoundError:
            pass  # no cache yet: fall through to a full build
    for season in range(first, LAST_SEASON + 1):
        try:
            pbp = nfl.load_pbp([season]).select(
                ["season", "week", "posteam", "defteam", "epa",
                 "game_seconds_remaining", "score_differential"]).to_pandas()
        except Exception:
            continue  # season not published yet
        c = pbp[(pbp.game_seconds_remaining <= CLUTCH_SECONDS)
                & (pbp.score_differential.abs() <= CLUTCH_MARGIN)
                & pbp.epa.notna() & pbp.posteam.notna()]
        off = (c.groupby(["season", "week", "posteam"])["epa"]
               .agg(["sum", "count"]).reset_index()
               .rename(columns={"posteam": "team", "sum": "off_clutch_epa",
                                "count": "off_clutch_plays"}))
        def_ = (c.groupby(["season", "week", "defteam"])["epa"]
                .agg(["sum", "count"]).reset_index()
                .rename(columns={"defteam": "team", "sum": "def_clutch_epa",
                                 "count": "def_clutch_plays"}))
        merged = off.merge(def_, on=["season", "week", "team"], how="outer")
        # Every team that played that week gets a row, even one with nothing in
        # it. A blowout produces no last-five-minute one-score snaps at all, and
        # without a zero row that team-week simply vanished from the table --
        # which left the rolling window with no row to attach a value to and
        # blanked the model's strongest input on 41% of played games.
        played = pd.concat([
            pbp[["season", "week", "posteam"]].rename(columns={"posteam": "team"}),
            pbp[["season", "week", "defteam"]].rename(columns={"defteam": "team"}),
        ]).dropna().drop_duplicates()
        merged = played.merge(merged, on=["season", "week", "team"], how="left")
        frames.append(merged)
        print(f"{season}: {len(c)} clutch plays, {len(merged)} team-games")

    out = pd.concat(frames, ignore_index=True).fillna(0)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv("clutch_team_weeks.csv", index=False)
    print(f"Wrote clutch_team_weeks.csv ({len(out)} team-games, "
          f"{out.season.min()}-{out.season.max()})")


if __name__ == "__main__":
    main()
