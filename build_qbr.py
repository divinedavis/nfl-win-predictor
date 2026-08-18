"""Cache ESPN Total QBR (week level and season level) from the nflverse data
release, trimmed to the seasons the dashboard shows.

QBR is ESPN's own 0-100 measure — it weights each play by win-probability
leverage and splits credit between the passer and everyone else — so it says
something the EPA-per-dropback grades in qb_grades.py do not, and the two
sitting side by side is the point.

    python build_qbr.py [--first 2021]
"""

import argparse

import pandas as pd

from build_qb_splits import FIRST_SEASON
from features import canon

BASE = ("https://github.com/nflverse/nflverse-data/releases/download/espn_data/"
        "qbr_{}_level.csv")
WEEK_CSV = "qbr_week.csv"
SEASON_CSV = "qbr_season.csv"
# ESPN spells two franchises differently from nflverse
ESPN_TEAM = {"WSH": "WAS", "LAR": "LA"}
KEEP = ["season", "season_type", "team", "player_id", "qb", "rank", "qbr",
        "plays", "pts_added", "epa_total"]


def tidy(d: pd.DataFrame, first: int) -> pd.DataFrame:
    d = d[d.season >= first].copy()
    d["team"] = d.team_abb.map(lambda t: canon(ESPN_TEAM.get(t, t)))
    d["qb"] = d.name_display.fillna(d.name_short)
    d = d.rename(columns={"qbr_total": "qbr", "qb_plays": "plays"})
    cols = KEEP + [c for c in ("week_num",) if c in d.columns]
    return d[cols].sort_values(["season", "season_type"] +
                               (["week_num"] if "week_num" in cols else []) +
                               ["rank"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=FIRST_SEASON)
    args = ap.parse_args()

    for level, path in (("week", WEEK_CSV), ("season", SEASON_CSV)):
        raw = pd.read_csv(BASE.format(level))
        out = tidy(raw, args.first)
        out.to_csv(path, index=False)
        print(f"Wrote {path} ({len(out)} rows, {out.season.min()}-"
              f"{out.season.max()})")


if __name__ == "__main__":
    main()
