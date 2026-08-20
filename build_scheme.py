"""Scheme identity per team-week, including man versus zone coverage.

The participation feed carries `defense_man_zone_type` and
`defense_coverage_type` on every pass play from 2018 on -- man against zone,
and which coverage inside that (Cover 1, Cover 2, Cover 3, 2-Man, Cover 4,
Cover 0). No subscription, no key. So the scheme question can be asked properly
rather than guessed at:

    what a defence DOES     how much man it plays, how much it blitzes, how
                            many defenders it puts in the box, nickel or dime
    what an offence FACES   its EPA per play against man, and against zone,
                            kept separate

The second one is the point. A matchup effect needs three things to hold at
once: an offence is better against one coverage than the other by more than
luck, that difference carries forward into later games, and the defence it is
about to play has a stable preference. Any one of them failing means there is
nothing to predict with.

Two limits worth knowing before reading any result. Coverage is only recorded
on pass plays, so about half of all snaps carry no label. And the feed is
published after the postseason rather than during it, so these are always a
team's tendencies from earlier seasons, never this week's -- useful only if
scheme identity persists year to year, which the ablation checks first.

    python build_scheme.py [--first 2018]
"""

import argparse
import re

import numpy as np
import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon

# Positions grouped into what actually matters for reading a formation.
DB = {"CB", "FS", "SS", "DB", "S"}
COUNT = re.compile(r"(\d+)\s+([A-Z]+)")
# nflverse game ids look like 2024_01_ARI_BUF.
GAME_ID = re.compile(r"^(\d{4})_(\d{2})_")


def parse(personnel: str) -> dict:
    """'3 CB, 2 DE, 2 DT, 1 FS, 2 ILB, 1 SS' -> {'CB': 3, 'DE': 2, ...}"""
    if not isinstance(personnel, str):
        return {}
    return {pos: int(n) for n, pos in COUNT.findall(personnel)}


def season_week(game_ids: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Participation carries no season or week column; the game id has both."""
    m = game_ids.astype(str).str.extract(GAME_ID)
    return (pd.to_numeric(m[0], errors="coerce"),
            pd.to_numeric(m[1], errors="coerce"))


def build_season(season: int) -> pd.DataFrame | None:
    try:
        p = nfl.load_participation([season]).to_pandas()
    except Exception:
        return None
    if p.empty or "defense_man_zone_type" not in p.columns:
        return None

    p = p[p.possession_team.notna()].copy()
    p["season"], p["week"] = season_week(p.nflverse_game_id)
    p = p[p.week.notna()]

    # --- what each side lines up in -------------------------------------
    dfn = p.defense_personnel.map(parse)
    off = p.offense_personnel.map(parse)
    p["n_db"] = [sum(v for k, v in d.items() if k in DB) for d in dfn]
    p["n_wr"] = [d.get("WR", 0) for d in off]
    p["n_te"] = [d.get("TE", 0) for d in off]
    p["shotgun"] = p.offense_formation.eq("SHOTGUN").astype(float)
    p["box"] = pd.to_numeric(p.defenders_in_box, errors="coerce")

    # The defence's own snaps are the ones it is defending, so they key off
    # whoever did NOT have the ball. Participation only names the possession
    # team, so the defence is recovered from the schedule below.
    d = (p[p.n_db > 0]
         .assign(nickel=lambda x: (x.n_db == 5).astype(float),
                 dime=lambda x: (x.n_db >= 6).astype(float))
         .groupby(["season", "week", "nflverse_game_id", "possession_team"])
         .agg(off_vs_nickel=("nickel", "mean"), off_vs_dime=("dime", "mean"),
              off_vs_box=("box", "mean")).reset_index())

    o = (p[p.n_wr > 0]
         .assign(three_wr=lambda x: (x.n_wr >= 3).astype(float),
                 heavy_te=lambda x: (x.n_te >= 2).astype(float))
         .groupby(["season", "week", "nflverse_game_id", "possession_team"])
         .agg(off_three_wr=("three_wr", "mean"),
              off_heavy_te=("heavy_te", "mean"),
              off_shotgun=("shotgun", "mean")).reset_index())

    # --- coverage, joined to EPA ----------------------------------------
    pbp = nfl.load_pbp([season]).select(
        ["game_id", "play_id", "season", "week", "posteam", "defteam", "epa"]
    ).to_pandas()
    cov = p[["nflverse_game_id", "play_id", "defense_man_zone_type"]].copy()
    cov.columns = ["game_id", "play_id", "cover"]
    cov = cov[cov.cover.astype(str).str.contains("MAN|ZONE", na=False)].copy()
    cov["is_man"] = cov.cover.str.startswith("MAN").astype(float)
    j = pbp.merge(cov, on=["game_id", "play_id"], how="inner")
    j = j[j.epa.notna() & j.posteam.notna() & j.defteam.notna()]

    # What the defence chose to play, and how it fared in each.
    dm = (j.groupby(["season", "week", "defteam"])
          .agg(def_man_rate=("is_man", "mean"),
               def_cov_snaps=("is_man", "size")).reset_index()
          .rename(columns={"defteam": "team"}))

    # How the offence did against each coverage, kept separate.
    om = (j.groupby(["season", "week", "posteam", "is_man"])["epa"]
          .agg(epa="mean", snaps="size").reset_index())
    om = om.pivot_table(index=["season", "week", "posteam"], columns="is_man",
                        values=["epa", "snaps"]).reset_index()
    om.columns = ["season", "week", "team", "off_epa_zone", "off_epa_man",
                  "off_snaps_zone", "off_snaps_man"]

    # Formation columns are keyed on the team with the ball.
    o = o.rename(columns={"possession_team": "team"}).drop(columns="nflverse_game_id")
    d = d.rename(columns={"possession_team": "team"}).drop(columns="nflverse_game_id")
    out = o.merge(d, on=["season", "week", "team"], how="outer")
    out = out.merge(om, on=["season", "week", "team"], how="outer")
    out = out.merge(dm, on=["season", "week", "team"], how="outer")
    print(f"{season}: {len(p):,} plays, {len(j):,} with coverage, "
          f"{len(out)} team-games")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first", type=int, default=2018)
    args = ap.parse_args()

    frames = [f for f in (build_season(s)
                          for s in range(args.first, LAST_SEASON + 1))
              if f is not None]
    if not frames:
        raise SystemExit("no participation data available")

    out = pd.concat(frames, ignore_index=True)
    out["team"] = out["team"].map(canon)
    out = out.sort_values(["season", "week"]).reset_index(drop=True)
    out.to_csv("scheme_team_weeks.csv", index=False)

    yr = int(out.season.max())
    print(f"\nWrote scheme_team_weeks.csv ({len(out):,} team-games, "
          f"{int(out.season.min())}-{yr})")
    last = out[out.season == yr]
    print(f"\nMost man-heavy defences of {yr}:")
    for team, v in last.groupby("team").def_man_rate.mean().nlargest(4).items():
        print(f"   {team}  {v:.0%} man coverage")
    print(f"\nMost zone-heavy defences of {yr}:")
    for team, v in last.groupby("team").def_man_rate.mean().nsmallest(4).items():
        print(f"   {team}  {v:.0%} man coverage")
    league = out.groupby("season").def_man_rate.mean()
    print("\nLeague-wide man coverage rate by season:")
    for s, v in league.items():
        print(f"   {int(s)}  {v:.1%}")


if __name__ == "__main__":
    main()
