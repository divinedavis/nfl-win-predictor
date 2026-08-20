"""Rate defenders by what happens while they are on the field, not by their stats.

features.py values a defender with a box score: sacks 2.0, interceptions 4.0,
tackles 0.1, and so on. That is the only option the public data offered, and it
has a known hole this repo already documents -- a nose tackle who occupies two
blockers creates the sack that the linebacker behind him records, and rates
near zero for it. DEF_SNAP_VALUE_FLOOR exists to patch exactly that.

The participation feed closes the hole. It names all eleven defenders on every
snap, so a defender can be measured the way offensive players already are: by
what actually happened while he was out there.

Two confounds have to be handled or the number is worthless.

    situation   a pass rusher plays third and long, a run stuffer plays first
                and ten, and third and long yields more EPA for reasons that
                have nothing to do with either of them. So every play is scored
                against the league average for its down and distance first.
    teammates   a mediocre player on a great defence looks great. So a
                defender is measured against HIS OWN TEAM's average over the
                same season, not against the league.

Whether this beats the box score is then one question: which rating tells you
more about next season? A rating that does not carry forward cannot help
predict anything, however principled its construction.

    python defender_ratings.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import nflreadpy as nfl

from features import DEF_VALUE_WEIGHTS, LAST_SEASON

FIRST = 2018          # participation carries defenders from here
MIN_SNAPS = 250       # a season's worth of meaningful playing time
CACHE = "defender_ratings.csv"


def situation_baseline(pbp: pd.DataFrame) -> pd.Series:
    """League average EPA for each down and distance bucket."""
    b = pbp.copy()
    b["bucket"] = (b.down.astype("Int64").astype(str) + "&"
                   + pd.cut(b.ydstogo, [0, 3, 6, 10, 99],
                            labels=["1-3", "4-6", "7-10", "11+"]).astype(str))
    return b.groupby("bucket").epa.mean(), b["bucket"]


def build() -> pd.DataFrame:
    rows = []
    for season in range(FIRST, LAST_SEASON + 1):
        try:
            part = nfl.load_participation([season]).to_pandas()
            pbp = nfl.load_pbp([season]).select(
                ["game_id", "play_id", "epa", "down", "ydstogo", "defteam",
                 "play_type"]).to_pandas()
        except Exception:
            continue
        if "defense_players" not in part.columns:
            continue

        pbp = pbp[pbp.play_type.isin(["run", "pass"]) & pbp.epa.notna()
                  & pbp.down.notna() & pbp.defteam.notna()]
        base, bucket = situation_baseline(pbp)
        pbp = pbp.assign(bucket=bucket)
        pbp["above"] = pbp.epa - pbp.bucket.map(base)

        d = part[["nflverse_game_id", "play_id", "defense_players"]].copy()
        d.columns = ["game_id", "play_id", "players"]
        d = d[d.players.notna()]
        j = pbp.merge(d, on=["game_id", "play_id"], how="inner")
        if j.empty:
            continue

        # Explode eleven defenders per snap into eleven rows.
        j = j.assign(player=j.players.str.split(";")).explode("player")
        j = j[j.player.str.len() > 3]

        team_avg = j.groupby("defteam").above.mean()
        g = (j.groupby(["player", "defteam"])
             .agg(snaps=("above", "size"), on_field=("above", "mean"))
             .reset_index())
        g = g[g.snaps >= MIN_SNAPS].copy()
        # Negated so positive is always a good defender, and measured against
        # what his own team allowed.
        g["rating"] = -(g.on_field - g.defteam.map(team_avg))
        g["season"] = season
        rows.append(g[["season", "player", "defteam", "snaps", "rating"]])
        print(f"{season}: {len(j):,} defender-snaps, {len(g)} rated defenders")

    out = pd.concat(rows, ignore_index=True)
    out.to_csv(CACHE, index=False)
    return out


def box_score_ratings(seasons: range) -> pd.DataFrame:
    """The current method, for comparison: value from the stat line."""
    rows = []
    for season in seasons:
        try:
            s = nfl.load_player_stats([season]).to_pandas()
        except Exception:
            continue
        cols = [c for c in DEF_VALUE_WEIGHTS if c in s.columns]
        if not cols:
            continue
        s = s[s.season_type.eq("REG")] if "season_type" in s.columns else s
        val = sum(s[c].fillna(0) * DEF_VALUE_WEIGHTS[c] for c in cols)
        g = (s.assign(val=val).groupby(["season", "player_id"])
             .agg(box=("val", "sum"), games=("val", "size")).reset_index())
        g = g[g.games >= 8]
        g["box"] = g.box / g.games          # per game, as features.py uses it
        rows.append(g.rename(columns={"player_id": "player"}))
        print(f"{season}: {len(g)} players with a defensive stat line")
    return pd.concat(rows, ignore_index=True)


def persistence(df: pd.DataFrame, col: str, label: str) -> float:
    """Does this season's rating predict next season's?"""
    nxt = df[["season", "player", col]].copy()
    nxt["season"] -= 1
    j = df.merge(nxt, on=["season", "player"], suffixes=("_now", "_next"))
    r = j[f"{col}_now"].corr(j[f"{col}_next"])
    print(f"   {label:34s} r = {r:+.3f}   (n = {len(j):,} player-seasons)")
    return r


def main() -> None:
    try:
        on = pd.read_csv(CACHE)
    except FileNotFoundError:
        on = build()
    box = box_score_ratings(range(FIRST, LAST_SEASON + 1))

    print(f"\n=== On-field ratings, {int(on.season.min())}-{int(on.season.max())} ===")
    print(f"   {len(on):,} defender-seasons with {MIN_SNAPS}+ snaps\n")
    last = on[on.season == on.season.max()].nlargest(8, "rating")
    ids = nfl.load_players().to_pandas()[["gsis_id", "display_name", "position"]]
    named = last.merge(ids, left_on="player", right_on="gsis_id", how="left")
    print(f"   Best of {int(on.season.max())} by what happened on their snaps:")
    for _, r in named.iterrows():
        print(f"      {str(r.display_name):24s} {str(r.position):4s} {r.defteam:4s} "
              f"{r.rating:+.4f} EPA/play better than his own defence "
              f"({int(r.snaps)} snaps)")

    print("\n=== Which rating tells you more about next season? ===\n")
    r_on = persistence(on, "rating", "on-field (participation)")
    r_box = persistence(box, "box", "box score (what we use now)")
    print()
    if r_on > r_box + 0.05:
        print("   The on-field rating carries forward better. Worth testing in "
              "the model.")
    elif r_box > r_on + 0.05:
        print("   The BOX SCORE carries forward better. The on-field number is "
              "noisier,\n   which is the usual fate of on/off measurements: a "
              "full-time starter is\n   on the field for almost every snap, so "
              "there is barely any 'off' to\n   compare against and the rating "
              "mostly measures his teammates.")
    else:
        print("   Neither is clearly better. Given the box score is already "
              "built and\n   already in the model, that is not a reason to "
              "switch.")


if __name__ == "__main__":
    main()
