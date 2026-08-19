"""A permanent record of what the model said, and an alarm when it goes wrong.

The model is rebuilt from scratch every morning, so what it believed yesterday
is gone. That makes a whole class of question unanswerable: what did we say
about this game before the injury news broke, are we worse on road underdogs
than on home favourites, has anything quietly got worse since the rule change.
You cannot learn from mistakes you never wrote down.

So this keeps a diary. Every run appends what the model currently thinks about
each upcoming game, stamped with the date. The same game gets a new row each day,
which is the point -- the line moving is itself information. Once a game is
played, its rows get the real result written next to the prediction.

Two things then become possible:

  the track record   how often it is right, broken out by the situations that
                     matter -- home or away, favourite or underdog, weather,
                     how confident it claimed to be
  the alarm          whether recent games are going worse than luck explains,
                     which is how you find out a rule change or a shift in the
                     league has quietly broken something, in October rather
                     than in March

    python diary.py --record     # log today's view of every upcoming game
    python diary.py --settle     # write results in next to finished games
    python diary.py --report     # track record + the alarm
    python diary.py --backfill   # build history from past seasons, once
"""

from __future__ import annotations

import argparse
import os
from datetime import date

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES
from train import PARAMS, ELO_BLEND

DIARY = "diary.csv"

# How many recent games the alarm looks at. Small enough to notice a problem
# inside a season, big enough that a couple of upsets do not trip it.
ALARM_WINDOW = 80
# How far off the alarm tolerates before shouting. At 2.5, a healthy model
# false-alarms roughly once in a hundred checks.
ALARM_THRESHOLD = 2.5

COLUMNS = ["recorded", "game_id", "season", "week", "home_team", "away_team",
           "home_prob", "elo_prob", "spread_line", "home_qb_val", "away_qb_val",
           "home_n_out", "away_n_out", "temp", "wind",
           "home_score", "away_score", "home_win", "correct"]


def load_diary() -> pd.DataFrame:
    if os.path.exists(DIARY):
        return pd.read_csv(DIARY)
    return pd.DataFrame(columns=COLUMNS)


def save_diary(df: pd.DataFrame) -> None:
    df.to_csv(DIARY, index=False)


def current_model(done: pd.DataFrame) -> XGBClassifier:
    m = XGBClassifier(**PARAMS)
    m.fit(done[FEATURES], done["home_win"])
    return m


def blend(raw: np.ndarray, elo: np.ndarray) -> np.ndarray:
    return (1 - ELO_BLEND) * raw + ELO_BLEND * elo


def snapshot(games: pd.DataFrame, probs: np.ndarray, stamp: str) -> pd.DataFrame:
    out = pd.DataFrame({
        "recorded": stamp,
        "game_id": games.game_id.values,
        "season": games.season.values, "week": games.week.values,
        "home_team": games.home_team.values, "away_team": games.away_team.values,
        "home_prob": np.round(probs, 4),
        "elo_prob": np.round(games.elo_prob.values, 4),
        "spread_line": games.get("spread_line", pd.Series(np.nan, index=games.index)).values,
        "home_qb_val": np.round(games.home_qb_val.values, 3),
        "away_qb_val": np.round(games.away_qb_val.values, 3),
        "home_n_out": games.home_n_out.values, "away_n_out": games.away_n_out.values,
        "temp": games.temp.values, "wind": games.wind.values,
    })
    for c in ("home_score", "away_score", "home_win", "correct"):
        out[c] = np.nan
    return out[COLUMNS]


def record(feat: pd.DataFrame, stamp: str) -> None:
    """Append today's view of every game that has not been played yet."""
    done = feat[feat.home_win.notna()]
    upcoming = feat[feat.home_score.isna()].copy()
    if upcoming.empty:
        print("no unplayed games to record")
        return

    model = current_model(done)
    probs = blend(model.predict_proba(upcoming[FEATURES])[:, 1],
                  upcoming.elo_prob.to_numpy())

    diary = load_diary()
    # One row per game per day; re-running the morning job must not double up.
    already = set(zip(diary.recorded.astype(str), diary.game_id.astype(str))) \
        if len(diary) else set()
    fresh = snapshot(upcoming, probs, stamp)
    fresh = fresh[[(stamp, g) not in already for g in fresh.game_id.astype(str)]]
    if fresh.empty:
        print(f"already recorded {stamp}")
        return
    save_diary(pd.concat([diary, fresh], ignore_index=True))
    print(f"recorded {len(fresh)} predictions for {stamp}")


def settle(feat: pd.DataFrame) -> None:
    """Write the real result next to every prediction whose game has finished."""
    diary = load_diary()
    if diary.empty:
        print("diary is empty")
        return
    results = feat[feat.home_win.notna()].set_index("game_id")
    open_rows = diary.home_win.isna()
    if not open_rows.any():
        print("nothing left to settle")
        return

    for i in diary.index[open_rows]:
        gid = diary.at[i, "game_id"]
        if gid not in results.index:
            continue
        r = results.loc[gid]
        diary.at[i, "home_score"] = r.home_score
        diary.at[i, "away_score"] = r.away_score
        diary.at[i, "home_win"] = r.home_win
        # The pick is whichever side the model gave more than an even chance.
        picked_home = diary.at[i, "home_prob"] > 0.5
        diary.at[i, "correct"] = float(picked_home == bool(r.home_win))
    save_diary(diary)
    print(f"settled {int(diary.home_win.notna().sum() - (~open_rows).sum())} rows; "
          f"{int(diary.home_win.notna().sum())} of {len(diary)} now have results")


def backfill(feat: pd.DataFrame, first_season: int) -> None:
    """Reconstruct the diary for past seasons, so the record is useful today.

    Each season is predicted by a model trained only on the seasons before it,
    so these are the numbers the model would genuinely have produced at the
    time -- it never sees the games it is judged on.
    """
    done = feat[feat.home_win.notna()].copy()
    rows = []
    for season in sorted(done[done.season >= first_season].season.unique()):
        train, test = done[done.season < season], done[done.season == season]
        if train.empty or test.empty:
            continue
        m = XGBClassifier(**PARAMS)
        m.fit(train[FEATURES], train["home_win"])
        probs = blend(m.predict_proba(test[FEATURES])[:, 1], test.elo_prob.to_numpy())
        snap = snapshot(test, probs, f"backfill-{season}")
        snap["home_score"] = test.home_score.values
        snap["away_score"] = test.away_score.values
        snap["home_win"] = test.home_win.values
        snap["correct"] = ((snap.home_prob > 0.5) == (snap.home_win == 1)).astype(float)
        rows.append(snap)
        print(f"  {season}: {len(snap)} games")

    diary = load_diary()
    diary = diary[~diary.recorded.astype(str).str.startswith("backfill-")] \
        if len(diary) else diary
    save_diary(pd.concat([diary, *rows], ignore_index=True))
    print(f"backfilled {sum(len(r) for r in rows)} games")


def _slice_table(df: pd.DataFrame, name: str, groups: dict) -> None:
    print(f"\n   by {name}")
    for label, mask in groups.items():
        g = df[mask]
        if len(g) < 25:
            continue
        expected = g.home_prob.where(g.home_prob > 0.5, 1 - g.home_prob).mean()
        print(f"     {label:26s} n={len(g):5d}   right {g.correct.mean():.1%}"
              f"   it expected {expected:.1%}")


def report(feat: pd.DataFrame) -> None:
    diary = load_diary()
    settled = diary[diary.home_win.notna()].copy()
    if len(settled) < 30:
        print(f"only {len(settled)} settled predictions -- not enough to report on yet. "
              "Run --backfill to build history from past seasons.")
        return
    # One row per game: the newest prediction is the one that stands.
    settled = settled.sort_values("recorded").drop_duplicates("game_id", keep="last")

    print(f"\n=== Track record: {len(settled)} games, "
          f"{int(settled.season.min())}-{int(settled.season.max())} ===")
    print(f"\n   right on {settled.correct.mean():.1%} of games")
    # "Expected" is how often it claimed it would be right; comparing the two is
    # the real test -- being right 65% of the time while claiming 75% is worse
    # than being right 60% while claiming 60%.
    claimed = settled.home_prob.where(settled.home_prob > 0.5,
                                      1 - settled.home_prob).mean()
    print(f"   it claimed it would be right {claimed:.1%} of the time")

    fav = settled.home_prob > 0.5
    _slice_table(settled, "who was favoured", {
        "home team favoured": fav, "away team favoured": ~fav})
    _slice_table(settled, "how confident it was", {
        "barely sure (50-60%)": settled.home_prob.between(0.4, 0.6),
        "fairly sure (60-70%)": (settled.home_prob.between(0.6, 0.7)
                                 | settled.home_prob.between(0.3, 0.4)),
        "very sure (70%+)": (settled.home_prob > 0.7) | (settled.home_prob < 0.3)})
    _slice_table(settled, "conditions", {
        "cold (under 40F)": settled.temp < 40,
        "windy (15mph+)": settled.wind >= 15,
        "indoors or mild": (settled.temp >= 40) | settled.temp.isna()})
    _slice_table(settled, "injuries", {
        "home missing 3+": settled.home_n_out >= 3,
        "away missing 3+": settled.away_n_out >= 3,
        "both mostly healthy": (settled.home_n_out <= 1) & (settled.away_n_out <= 1)})

    alarm(settled)


def alarm(settled: pd.DataFrame) -> None:
    """Has the model been doing worse lately than luck explains?

    Two separate questions, because one of them alone misses real breakage.

    Is it right as often as it claims? Every prediction is a claim about the
    model's own hit rate -- "70% home" means "I expect to be right here 7 times
    in 10". Add those claims up over the last stretch and compare to how often
    it actually was right. This is the check that catches the model simply
    getting worse, including a catastrophic wiring error like home and away
    being swapped upstream.

    Is it leaning the wrong way? Add up the home-win chances and compare to how
    many home teams actually won. This catches a one-sided bias -- a broken
    weather feed flattering home teams, say -- that leaves the hit rate intact.

    Neither ever matches exactly, because football is random. The question is
    whether the gap is bigger than randomness accounts for.
    """
    recent = settled.sort_values("recorded").tail(ALARM_WINDOW)
    if len(recent) < 40:
        print(f"\n   alarm: only {len(recent)} recent games, need 40+ to judge")
        return

    p = recent.home_prob.to_numpy()
    print(f"\n=== Alarm: the last {len(recent)} games ===")
    tripped = []

    # 1. Right as often as it claims?
    claim = np.maximum(p, 1 - p)              # its own stated chance of being right
    picked_home = p > 0.5
    was_right = (picked_home == (recent.home_win.to_numpy() == 1)).astype(float)
    exp_right, act_right = claim.sum(), was_right.sum()
    spread_right = np.sqrt((claim * (1 - claim)).sum())
    off_right = (act_right - exp_right) / spread_right
    print(f"   it expected to be right {exp_right:.1f} times, it was right "
          f"{act_right:.0f}  (normal swing {spread_right:.1f})")
    if off_right < -ALARM_THRESHOLD:
        tripped.append("It is being right far less often than it claims it will be. "
                       "That is the model itself getting worse, or something wired "
                       "up wrong -- check that home and away have not been swapped.")

    # 2. Leaning the wrong way?
    exp_home, act_home = p.sum(), recent.home_win.sum()
    spread_home = np.sqrt((p * (1 - p)).sum())
    off_home = (act_home - exp_home) / spread_home
    print(f"   it expected {exp_home:.1f} home wins, there were {act_home:.0f}"
          f"  (normal swing {spread_home:.1f})")
    if abs(off_home) > ALARM_THRESHOLD:
        side = "too much" if off_home < 0 else "not enough"
        tripped.append(f"It is favouring home teams {side}. A one-sided lean like "
                       "this usually means an input is wrong -- weather, rest days, "
                       "or the injury feed -- rather than the model being stale.")

    print(f"   overall it was right {was_right.mean():.1%} of the time")
    if tripped:
        print("\n   *** ALARM")
        for t in tripped:
            print(f"   - {t}")
    else:
        print(f"\n   all clear -- both gaps are within what randomness explains "
              f"({abs(off_right):.1f} and {abs(off_home):.1f}, "
              f"limit {ALARM_THRESHOLD})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--backfill", action="store_true")
    ap.add_argument("--first-season", type=int, default=2015)
    ap.add_argument("--date", default=None, help="stamp to record under")
    args = ap.parse_args()

    feat = pd.read_parquet("features.parquet")
    if args.backfill:
        backfill(feat, args.first_season)
    if args.record:
        record(feat, args.date or date.today().isoformat())
    if args.settle:
        settle(feat)
    if args.report or not any((args.record, args.settle, args.backfill)):
        report(feat)


if __name__ == "__main__":
    main()
