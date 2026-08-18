"""Build per-quarter quarterback dropback aggregates from play-by-play, so the
dashboard can grade every passer by quarter and by late-game situation.

One row per (season, season_type, quarterback, team, bucket) with the raw sums
the grading step needs — nothing is normalized here, so the grade formula can
change without re-downloading a decade of play-by-play.

Buckets overlap on purpose: a snap with 3:00 left in a tied fourth quarter
lands in q4, late and clutch alike.

    python build_qb_splits.py [--first 2021] [--update]
"""

import argparse

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon

FIRST_SEASON = 2021        # five completed seasons through 2025
LATE_SECONDS = 300         # "under five minutes" — of the fourth quarter or OT
CLUTCH_MARGIN = 8          # one-score game at the snap

CSV = "qb_splits.csv"
DEF_CSV = "qb_splits_def.csv"

COLS = ["season", "season_type", "game_id", "posteam", "qtr",
        "quarter_seconds_remaining", "score_differential", "qb_dropback",
        "qb_scramble", "sack", "interception", "pass_touchdown",
        "complete_pass", "pass_attempt", "epa", "success", "cpoe", "defteam",
        "passer_id", "passer", "rusher_id", "rusher"]


def buckets(d: pd.DataFrame) -> dict:
    """Boolean mask per bucket. OT counts as fourth-quarter football for the
    late/clutch splits but keeps its own row so it can be read separately."""
    late = (d.qtr >= 4) & (d.quarter_seconds_remaining <= LATE_SECONDS)
    return {
        "all": pd.Series(True, index=d.index),
        "q1": d.qtr == 1, "q2": d.qtr == 2, "q3": d.qtr == 3, "q4": d.qtr == 4,
        "ot": d.qtr >= 5,
        "late": late,
        # one-score plays, whenever they happen: the control group that says
        # how much of any late-game drop is situation rather than the passer
        "close": d.score_differential.abs() <= CLUTCH_MARGIN,
        "clutch": late & (d.score_differential.abs() <= CLUTCH_MARGIN),
    }


def aggregate_def(d: pd.DataFrame, bucket: str) -> pd.DataFrame:
    """Same buckets from the defense's side: EPA allowed per dropback faced.
    The playoff backtest uses it to size how much better postseason defenses
    are, so a QB's January dip isn't blamed on January alone."""
    out = d.groupby(["season", "season_type", "defteam"], dropna=False).agg(
        n=("epa", "size"), epa=("epa", "sum"), succ=("success", "sum"),
        sacks=("sack", "sum"), ints=("interception", "sum"),
    ).reset_index().rename(columns={"defteam": "team"})
    out.insert(2, "bucket", bucket)
    return out


def aggregate(d: pd.DataFrame, bucket: str) -> pd.DataFrame:
    g = d.groupby(["season", "season_type", "qb_id", "qb", "team"], dropna=False)
    out = g.agg(
        games=("game_id", "nunique"),
        n=("epa", "size"),
        epa=("epa", "sum"),
        # sum of squares: the grading step needs play-level variance to size
        # its empirical-Bayes shrinkage per bucket
        epa2=("epa", lambda x: float((x ** 2).sum())),
        succ=("success", "sum"),
        cpoe=("cpoe", "sum"),
        cpoe_n=("cpoe", "count"),
        sacks=("sack", "sum"),
        ints=("interception", "sum"),
        tds=("pass_touchdown", "sum"),
        att=("pass_attempt", "sum"),
        comp=("complete_pass", "sum"),
    ).reset_index()
    out.insert(2, "bucket", bucket)
    return out


def season_rows(season: int):
    try:
        pbp = nfl.load_pbp([season]).select(COLS).to_pandas()
    except Exception:
        return None                      # season not published yet
    d = pbp[(pbp.qb_dropback == 1) & pbp.epa.notna() & pbp.posteam.notna()].copy()
    d["qb_id"] = d.passer_id.fillna(d.rusher_id)
    d["qb"] = d.passer.fillna(d.rusher)
    d = d[d.qb_id.notna()]               # a handful of trick plays have no passer
    d["team"] = d.posteam.map(canon)
    d["success"] = d.success.fillna(0)
    masks = {name: mask for name, mask in buckets(d).items() if mask.any()}
    off = pd.concat([aggregate(d[m], name) for name, m in masks.items()],
                    ignore_index=True)
    dfn = d[d.defteam.notna()].copy()
    dfn["defteam"] = dfn.defteam.map(canon)
    dmasks = buckets(dfn)
    def_ = pd.concat([aggregate_def(dfn[dmasks[name]], name) for name in masks],
                     ignore_index=True)
    return off, def_


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--first", type=int, default=FIRST_SEASON)
    ap.add_argument("--update", action="store_true",
                    help="refresh only the current season into the cache "
                         "(daily droplet mode — history is static)")
    args = ap.parse_args()

    frames, def_frames = [], []
    first = args.first
    if args.update:
        try:
            prev = pd.read_csv(CSV)
            prev_def = pd.read_csv(DEF_CSV)
            frames.append(prev[prev.season < LAST_SEASON])
            def_frames.append(prev_def[prev_def.season < LAST_SEASON])
            first = LAST_SEASON
        except FileNotFoundError:
            frames, def_frames = [], []   # no cache yet: fall through to a full build

    for season in range(first, LAST_SEASON + 1):
        rows = season_rows(season)
        if rows is None:
            print(f"{season}: not published yet, skipped")
            continue
        off, def_ = rows
        frames.append(off)
        def_frames.append(def_)
        drops = int(off.loc[off.bucket == "all", "n"].sum())
        print(f"{season}: {drops} dropbacks, {off.qb_id.nunique()} passers")

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["season", "season_type", "bucket", "n"],
                          ascending=[True, True, True, False]).reset_index(drop=True)
    out.to_csv(CSV, index=False)
    dout = pd.concat(def_frames, ignore_index=True).sort_values(
        ["season", "season_type", "bucket", "team"]).reset_index(drop=True)
    dout.to_csv(DEF_CSV, index=False)
    print(f"Wrote {CSV} ({len(out)} rows, {out.season.min()}-{out.season.max()}) "
          f"and {DEF_CSV} ({len(dout)} rows)")


if __name__ == "__main__":
    main()
