"""Cache every field-goal attempt (2006+) from play-by-play into kicks.csv:
season, week, team, kicker, distance, made. One heavy pbp pass, tiny output —
the raw material for kicker-quality features.

    python build_kicks.py [--update]   (--update refreshes current season only)
"""

import argparse

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon

FIRST = 2006


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--update", action="store_true")
    args = ap.parse_args()

    frames = []
    first = FIRST
    if args.update:
        try:
            prev = pd.read_csv("kicks.csv")
            frames.append(prev[prev.season < LAST_SEASON])
            first = LAST_SEASON
        except FileNotFoundError:
            pass
    for season in range(first, LAST_SEASON + 1):
        try:
            pbp = nfl.load_pbp([season]).select(
                ["season", "week", "posteam", "field_goal_result",
                 "kick_distance", "kicker_player_id",
                 "kicker_player_name"]).to_pandas()
        except Exception:
            continue
        fg = pbp[pbp.field_goal_result.isin(["made", "missed", "blocked"])
                 & pbp.kick_distance.notna()
                 & pbp.kicker_player_id.notna()].copy()
        fg["team"] = fg["posteam"].map(canon)
        fg["made"] = (fg.field_goal_result == "made").astype(int)
        frames.append(fg[["season", "week", "team", "kicker_player_id",
                          "kicker_player_name", "kick_distance", "made"]])
        print(f"{season}: {len(fg)} FG attempts")

    out = pd.concat(frames, ignore_index=True).sort_values(["season", "week"])
    out.to_csv("kicks.csv", index=False)
    print(f"Wrote kicks.csv ({len(out)} attempts, "
          f"{out.season.min()}-{out.season.max()})")


if __name__ == "__main__":
    main()
