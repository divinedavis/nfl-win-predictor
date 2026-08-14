"""Audit every team's projected starting QB against the official depth chart
and last season's starts leader. Run before each week ships — a projected
name that isn't the depth-chart QB1 means stale expectations somewhere.

    python qb_audit.py
"""

import pandas as pd
import nflreadpy as nfl

from features import LAST_SEASON, canon, norm_name, load_depth_qb1


def main() -> None:
    df = pd.read_parquet("features.parquet")
    wk = df[(df.season == LAST_SEASON) & (df.game_type == "REG")].sort_values(
        ["week", "gameday"])
    proj = {}
    for r in wk.itertuples(index=False):
        proj.setdefault(r.home_team, (r.home_qb_pred_name, r.home_qb_val))
        proj.setdefault(r.away_team, (r.away_qb_pred_name, r.away_qb_val))

    dc = nfl.load_depth_charts([LAST_SEASON]).to_pandas()
    qb1 = (dc[(dc.pos_abb == "QB") & (dc.pos_rank == 1)]
           .sort_values("dt").groupby("team").tail(1))
    qb1_name = {canon(r.team): r.player_name for r in qb1.itertuples(index=False)}

    issues = 0
    for team in sorted(proj):
        name, val = proj[team]
        d = qb1_name.get(team, "?")
        flag = ("EMPTY NAME" if not name else
                "MISMATCH vs depth chart" if norm_name(name) != norm_name(d)
                else "")
        if flag:
            issues += 1
        mark = f"  <-- {flag}" if flag else ""
        print(f"{team:5s} {str(name)[:24]:24s} ({val:+.1f})  depth: {d}{mark}")
    print(f"\n{issues} issue(s) across {len(proj)} teams")
    raise SystemExit(1 if issues else 0)


if __name__ == "__main__":
    main()
