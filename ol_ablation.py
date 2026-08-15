"""Three-gate test of the multi-starter offensive-line injury hypothesis.

The claim (user, 2026-08-15): losing TWO OR MORE starting linemen is not twice
one — the line has to shuffle assignments around a backup, so it should hurt
team scoring, and drag QB / RB / WR production with it.

The existing model already carries {side}_ol_out_wt, a snap-weighted sum of
absent linemen. That is linear by construction: it cannot tell two half-time
backups from two full-time starters. features.py now also builds
{side}_ol_starters_out (bodies, starters only) and {side}_ol_multi_out (the
>=2 threshold), and this script decides whether they earn a place.

Gate 1 — team output: do teams score less, and move the ball worse, than their
own recent form implies when the line is missing multiple starters?
Gate 2 — player output: does the same absence show up in QB / RB / WR box
scores, measured against each player's own trailing form?
Gate 3 — model value: walk-forward game-model backtest, baseline vs +oline.
Only a Brier improvement promotes the group into FEATURES.
Gate 4 — props value: does the line's state sharpen the quantile projections?

    python ol_ablation.py          # all gates
    python ol_ablation.py 1 1b     # just the descriptive ones

VERDICT (2026-08-15): REAL IN RAW NUMBERS, ALREADY PRICED. Do not promote.

Teams missing 2+ starting linemen score 21.6 ppg and win 44%, against 23.4 ppg
and 52% for teams with a whole line — a big, real-looking gap. It is almost
entirely selection: bad teams get hurt and stay hurt, and eight games of
rolling form plus Elo have already absorbed it. Against a team's OWN recent
form, 2+ starters out is worth +0.16 points (z +0.51) and +0.06pp of sack rate
allowed (z +0.38); residualized against the linear snap weight the model
already carries, the threshold adds -0.023 win probability (z -1.44). QB, RB
and WR box scores versus each player's own trailing-4 form move by less than a
yard in every direction, and QB sacks taken by 0.04. Walk-forward, the group
adds nothing (Brier .2174 vs .2173, better in 5/11 seasons); swapping the
counts IN FOR the snap weight looked better on one seed (acc .6514) but over 8
seeds it is -0.0001 Brier +/- 0.0002, 5/8 — noise. Props: pinball flat to
worse in all four markets.

The one split that survives is a warning about the framing, not a feature:
lines that have been broken for WEEKS underperform Elo (-0.053, n=485) while
freshly broken lines do not (+0.007, n=379) — i.e. what is mispriced is slow
Elo, not the injury shock. Same shape as the kicker finding: a real effect the
ratings already own. Fits the project's pattern — a feature only wins when it
carries information Elo/EPA cannot have absorbed through game results.

Kept as a display-only factor: the dashboard's why-panel names a patched-up
line, since a lineman never makes the key-absence list (no stat line, no
rating) and it is the first injury anyone asks about.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

import nflreadpy as nfl

from features import FEATURES, CANDIDATE_GROUPS, canon, SNAPS_FIRST_SEASON
from train import PARAMS, BACKTEST_SEASONS, ELO_BLEND

OL_FEATS = CANDIDATE_GROUPS["oline"]
BUCKETS = [("0 out", 0.0, 0.5), ("1 out", 0.5, 1.5), ("2+ out", 1.5, 99.0)]
# Positions whose box scores the line is supposed to move.
SKILL = {"QB": ["passing_yards", "attempts", "sacks_suffered"],
         "RB": ["rushing_yards", "carries"],
         "WR": ["receiving_yards", "targets"],
         "TE": ["receiving_yards", "targets"]}


def team_weeks() -> pd.DataFrame:
    """One row per team-game: the line's pregame state, the team's pregame
    rolling form, and what it actually did."""
    df = pd.read_parquet("features.parquet")
    df = df[df.home_win.notna() & (df.season >= SNAPS_FIRST_SEASON)]
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "season": df.season, "week": df.week,
            "team": df[f"{side}_team"], "opp": df[f"{opp}_team"],
            "ol_out": df[f"{side}_ol_starters_out"],
            "ol_wt": df[f"{side}_ol_out_wt"],
            "points": df[f"{side}_score"],
            "pf8": df[f"{side}_pf8"],
            "off_epa": np.nan,  # filled from the opponent's allowed column
            "off_epa8": df[f"{side}_off_epa8"],
            "elo_prob": df.elo_prob if side == "home" else 1 - df.elo_prob,
            "won": df.home_win if side == "home" else 1 - df.home_win,
        }))
    tw = pd.concat(rows, ignore_index=True)
    return tw[tw.ol_out.notna()].copy()


def bucket(series: pd.Series) -> pd.Series:
    out = pd.Series(index=series.index, dtype=object)
    for name, lo, hi in BUCKETS:
        out[(series >= lo) & (series < hi)] = name
    return out


def gate1_team() -> None:
    tw = team_weeks()
    tw["b"] = bucket(tw.ol_out)
    tw["pts_resid"] = tw.points - tw.pf8          # vs own last-8 scoring
    tw["win_resid"] = tw.won - tw.elo_prob        # vs pregame Elo expectation
    print(f"=== Gate 1: team output by starting linemen out "
          f"({len(tw)} team-games, {SNAPS_FIRST_SEASON}+) ===")
    print(f"{'line state':10s} {'n':>6s} {'points':>8s} {'vs form':>9s} "
          f"{'win%':>7s} {'vs Elo':>8s} {'ol_out_wt':>10s}")
    for name, _, _ in BUCKETS:
        g = tw[tw.b == name]
        if g.empty:
            continue
        print(f"{name:10s} {len(g):6d} {g.points.mean():8.2f} "
              f"{g.pts_resid.mean():+9.2f} {g.won.mean():7.3f} "
              f"{g.win_resid.mean():+8.3f} {g.ol_wt.mean():10.2f}")
    two = tw[tw.b == "2+ out"]
    for label, col in (("points vs own form", "pts_resid"),
                       ("wins vs Elo", "win_resid")):
        se = two[col].std() / np.sqrt(len(two))
        print(f"2+ out, {label}: {two[col].mean():+.3f} +/- {se:.3f} "
              f"(z {two[col].mean() / se:+.2f}, n={len(two)})")

    # The model already carries ol_out_wt, the LINEAR snap weight of absent
    # linemen. The hypothesis is specifically that the threshold adds
    # something linear weight misses — so residualize against it first.
    fit = tw[tw.ol_wt.notna()]
    X = np.column_stack([np.ones(len(fit)), fit.ol_wt])
    for col in ("pts_resid", "win_resid"):
        beta = np.linalg.lstsq(X, fit[col].to_numpy(), rcond=None)[0]
        r = fit[col].to_numpy() - X @ beta
        hi = r[(fit.ol_out >= 2).to_numpy()]
        se = hi.std() / np.sqrt(len(hi))
        print(f"beyond linear ol_out_wt, 2+ out {col}: {hi.mean():+.3f} "
              f"+/- {se:.3f} (z {hi.mean() / se:+.2f})   <- the threshold test")

    # A line that just broke has not had a week to rebuild protection calls;
    # one that has been patched for a month has. Split on it.
    tw = tw.sort_values(["season", "team", "week"])
    prev = tw.groupby(["season", "team"])["ol_out"].shift(1).fillna(0)
    fresh = (tw.ol_out >= 2) & (prev < 2)
    stale = (tw.ol_out >= 2) & (prev >= 2)
    for name, mask in (("newly broken", fresh), ("already patched", stale)):
        g = tw[mask]
        print(f"{name:16s} n={len(g):4d}  points vs form {g.pts_resid.mean():+.2f}"
              f"  wins vs Elo {g.win_resid.mean():+.3f}")
    print()


def gate1b_pressure() -> None:
    """The mechanism the hypothesis rests on: a patched line should get its
    quarterback hit. Team sack rate allowed, against the team's own trailing
    form, by line state — and split by how good the opposing rush is."""
    tw = team_weeks()[["season", "week", "team", "opp", "ol_out", "ol_wt"]]
    frames = []
    for season in range(SNAPS_FIRST_SEASON, 2026):
        try:
            ps = nfl.load_player_stats([season], summary_level="week").to_pandas()
        except Exception:
            continue
        cols = [c for c in ["season", "week", "team", "sacks_suffered",
                            "attempts", "carries"] if c in ps.columns]
        frames.append(ps[cols])
    team_game = (pd.concat(frames, ignore_index=True)
                 .assign(team=lambda x: x["team"].map(canon))
                 .groupby(["season", "week", "team"], as_index=False).sum())
    team_game["dropbacks"] = team_game.attempts + team_game.sacks_suffered
    team_game["sack_rate"] = (team_game.sacks_suffered
                              / team_game.dropbacks.replace(0, np.nan))
    team_game = team_game.sort_values(["team", "season", "week"])
    team_game["sack_rate8"] = team_game.groupby("team")["sack_rate"].transform(
        lambda s: s.shift(1).rolling(8, min_periods=4).mean())
    d = tw.merge(team_game, on=["season", "week", "team"], how="inner")
    d = d[d.sack_rate8.notna()]
    d["resid"] = d.sack_rate - d.sack_rate8
    d["b"] = bucket(d.ol_out)
    print("=== Gate 1b: sack rate allowed (the mechanism) ===")
    print(f"{'line state':12s} {'n':>6s} {'sack rate':>10s} {'vs own form':>12s}")
    for name, _, _ in BUCKETS:
        g = d[d.b == name]
        se = g.resid.std() / np.sqrt(len(g))
        print(f"{name:12s} {len(g):6d} {g.sack_rate.mean():10.4f} "
              f"{g.resid.mean():+12.4f} (+/-{se:.4f})")
    print()


def gate2_players() -> None:
    """Each player-game's production against his own trailing-4 mean, split by
    how many of his linemen were out. Within-player, so team quality and role
    cancel; what is left is the line's effect on his afternoon."""
    tw = team_weeks()[["season", "week", "team", "ol_out"]]
    cols = ["season", "week", "team", "player_id", "position",
            "passing_yards", "attempts", "sacks_suffered", "rushing_yards",
            "carries", "receiving_yards", "targets"]
    frames = []
    for season in range(SNAPS_FIRST_SEASON, 2026):
        try:
            ps = nfl.load_player_stats([season], summary_level="week").to_pandas()
        except Exception:
            continue
        have = [c for c in cols if c in ps.columns]
        frames.append(ps[have])
    ps = pd.concat(frames, ignore_index=True)
    ps["team"] = ps["team"].map(canon)
    ps = ps.sort_values(["player_id", "season", "week"])

    ps["yds_per_att"] = np.nan
    ps = ps.merge(tw, on=["season", "week", "team"], how="inner")
    ps["b"] = bucket(ps.ol_out)

    print("=== Gate 2: player production vs own trailing-4 form ===")
    print(f"{'position':9s} {'stat':16s} " +
          "  ".join(f"{n:>12s}" for n, _, _ in BUCKETS))
    for pos, stats in SKILL.items():
        sub = ps[ps.position == pos].copy()
        for stat in stats:
            if stat not in sub.columns:
                continue
            s = sub.sort_values(["player_id", "season", "week"]).copy()
            s[stat] = s[stat].fillna(0)
            base = s.groupby("player_id")[stat].transform(
                lambda x: x.shift(1).rolling(4, min_periods=3).mean())
            s["resid"] = s[stat] - base
            s = s[base.notna() & (base > 0)]
            cells = []
            for name, _, _ in BUCKETS:
                g = s[s.b == name]
                cells.append(f"{g.resid.mean():+7.2f} (n={len(g)})"
                             if len(g) else f"{'-':>12s}")
            print(f"{pos:9s} {stat:16s} " + "  ".join(f"{c:>12s}" for c in cells))
    print()


def add_streak(done: pd.DataFrame) -> tuple:
    """Weeks in a row (through this one) with 2+ starters out. Gate 1 found
    the underperformance sits in lines broken for a WHILE, not freshly broken
    ones — form absorbs it but slow-moving Elo may not, so test it."""
    long = pd.concat([
        done[["season", "week", f"{s}_team", f"{s}_ol_starters_out"]].rename(
            columns={f"{s}_team": "team", f"{s}_ol_starters_out": "n"})
        for s in ("home", "away")]).sort_values(["season", "team", "week"])
    streaks, run, key = [], 0, None
    for r in long.itertuples(index=False):
        if (r.season, r.team) != key:
            key, run = (r.season, r.team), 0
        run = run + 1 if r.n >= 2 else 0
        streaks.append(run)
    long["streak"] = streaks
    for side in ("home", "away"):
        done = done.merge(
            long[["season", "week", "team", "streak"]].rename(
                columns={"team": f"{side}_team", "streak": f"{side}_ol_streak"}),
            on=["season", "week", f"{side}_team"], how="left")
    done["ol_streak_diff"] = (done.home_ol_streak.fillna(0)
                              - done.away_ol_streak.fillna(0))
    return done, ["home_ol_streak", "away_ol_streak", "ol_streak_diff"]


def gate3_model() -> None:
    df = pd.read_parquet("features.parquet")
    done = df[df.home_win.notna()].copy()
    done, streak_feats = add_streak(done)
    swap = [f for f in FEATURES
            if f not in ("home_ol_out_wt", "away_ol_out_wt")] + OL_FEATS
    print("=== Gate 3: walk-forward game model (2015-2025) ===")
    base = None
    for name, feats in [("baseline", FEATURES),
                        ("+oline", FEATURES + OL_FEATS),
                        ("+streak", FEATURES + OL_FEATS + streak_feats),
                        ("swap wt", swap)]:
        probs, ys, seasons = [], [], []
        for season in BACKTEST_SEASONS:
            tr, te = done[done.season < season], done[done.season == season]
            if te.empty:
                continue
            m = XGBClassifier(**PARAMS)
            m.fit(tr[feats], tr["home_win"])
            raw = m.predict_proba(te[feats])[:, 1]
            probs.append((1 - ELO_BLEND) * raw + ELO_BLEND * te["elo_prob"].values)
            ys.append(te["home_win"].values)
            seasons.append(np.full(len(te), season))
        p, y, s = (np.concatenate(probs), np.concatenate(ys),
                   np.concatenate(seasons))
        per_season = {sea: brier_score_loss(y[s == sea], p[s == sea])
                      for sea in np.unique(s)}
        print(f"{name:9s} acc {accuracy_score(y, p > .5):.4f}  "
              f"brier {brier_score_loss(y, p):.4f}  "
              f"logloss {log_loss(y, p):.4f}")
        if base is None:
            base = per_season
        else:
            better = sum(per_season[k] < base[k] for k in base)
            print(f"          brier better in {better}/{len(base)} seasons "
                  f"(promotion needs a clear majority)")


def gate4_props() -> None:
    """Same question for the player-props quantile models: does knowing the
    line's state sharpen passing / rushing / receiving projections? Scored in
    pinball loss, the promotion criterion props.py already uses."""
    import props

    ps = props.load_player_weeks()
    ctx = props.game_context()
    ngs = props.load_ngs()
    injuries = props.load_injury_reports()
    print("=== Gate 4: props quantile models (pinball, lower is better) ===")
    print(f"{'stat':16s} {'n':>7s} {'v2':>8s} {'v2+ol':>8s} {'delta':>8s}")
    for stat in props.STATS:
        table = props.build_stat_table_v2(ps, ctx, stat, ngs, injuries)
        table = table[table["played"] & table["ol_out"].notna()]
        tot = {"v2": 0.0, "v2+ol": 0.0}
        n = 0
        for season in props.VALIDATE_SEASONS:
            train, test = table[table.season < season], table[table.season == season]
            if test.empty or train.empty:
                continue
            y = test["y"].to_numpy()
            for name, feats in [("v2", props.FEATS_V2),
                                ("v2+ol", props.FEATS_V2 + props.OL_FEATS)]:
                models = props.fit_quantiles(train, feats)
                pred = props.predict_quantiles(models, test, feats)
                tot[name] += props.pinball(y, pred) * len(y)
            n += len(y)
        if not n:
            continue
        v2, v2ol = tot["v2"] / n, tot["v2+ol"] / n
        print(f"{stat:16s} {n:7d} {v2:8.3f} {v2ol:8.3f} {v2ol - v2:+8.3f}")


if __name__ == "__main__":
    import sys

    gates = sys.argv[1:] or ["1", "1b", "2", "3", "4"]
    if "1" in gates:
        gate1_team()
    if "1b" in gates:
        gate1b_pressure()
    if "2" in gates:
        gate2_players()
    if "3" in gates:
        gate3_model()
    if "4" in gates:
        gate4_props()
