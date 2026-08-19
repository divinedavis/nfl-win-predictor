"""A play-level world model of an NFL game: state, actions, and dynamics.

The game-level model (features.py -> train.py) maps a pregame feature vector
straight to a win probability. This module models the thing underneath it: the
state of a football game and how one play moves it.

    state   = (possession, yardline, down, distance, clock, score, timeouts)
    action  = what the offense calls (run / pass / punt / field goal / kneel)
    outcome = yards, turnover, score, and how much clock it burned

nflverse play-by-play is already a labelled transition dataset -- every row
carries the state before the snap, the action, and the row after it is the
next state -- so the heads below are fit directly on it:

    policy       P(play type | state, team strength)      -- coaching behaviour
    pass_event   P(sack / int / incomplete / complete)
    fumble       P(fumble lost | play)
    penalty      P(pre-snap or accepted penalty | state)
    pass_yards   E[yards | complete] + a residual bootstrap
    run_yards    E[yards | run]      + a residual bootstrap
    fg           P(make | distance)

Everything that ends a possession -- punt, missed field goal, interception,
kickoff -- resolves through one empirical mechanism instead of hand-written
rules: `next_start` holds, for each terminal event and field-position bucket,
the actual yard line the receiving team started its next drive from. That is
measured off real games, so touchbacks, fair catches, returns and inside-the-20
punts are priced without a line of rulebook code.

Team strength enters every head through the same pregame features features.py
already builds (rolling offensive/defensive EPA, quarterback rating, Elo), so
the simulator inherits the injury, weather and form work rather than
duplicating it -- and stays leak-free for the same reason those features are.

Build:  python world_model.py            (fits the heads, writes world_model.pkl)
Use:    from world_model import Simulator
        sim = Simulator.load(); sim.simulate_game(features_row, n=5000)
"""

from __future__ import annotations

import argparse
import pickle
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import nflreadpy as nfl
from xgboost import XGBClassifier, XGBRegressor

MODEL_PATH = "world_model.pkl"

# Scrimmage dynamics are fit from 2015 on: the same window train.py backtests,
# recent enough that the rules resemble today's game and small enough that the
# play table fits the droplet's 2 GB.
FIRST_SEASON = 2015
# Kickoffs were re-written for 2024 (the "dynamic" kickoff moved the average
# start back), so anything downstream of a kick is measured on 2024+ only.
KICKOFF_ERA_FIRST_SEASON = 2024

# Play types the policy head chooses between. Everything else in the feed
# (kickoffs, extra points, timeouts) is driven by the game loop, not chosen.
PLAY_TYPES = ["run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike", "no_play"]
PASS_EVENTS = ["complete", "incomplete", "sack", "interception"]

# The state the offense sees at the snap.
STATE_FEATURES = [
    "down", "ydstogo", "yardline_100", "goal_to_go",
    "half_seconds_remaining", "game_seconds_remaining",
    "score_differential", "posteam_timeouts_remaining",
    "defteam_timeouts_remaining", "qtr", "is_home",
]
# Who is playing, in the same currency features.py already uses. Injuries are
# carried at the position-group level for both sides, so an absence reprices
# every snap: a defensive tackle out lifts the *offense's* expected yards and
# drops its sack risk on all sixty plays, rather than nudging one game-level
# input once.
OFF_OUT = ["qb_out_epa", "rb_out_epa", "wr_out_epa", "te_out_epa",
           "ol_out_wt", "qb_changed"]
DEF_OUT = ["dl_out_val", "lb_out_val", "db_out_val",
           "dl_out_wt", "lb_out_wt", "db_out_wt"]
STRENGTH_FEATURES = (
    ["off_epa8", "def_epa8_opp", "qb_val", "elo_off", "elo_def"]
    + [f"off_{c}" for c in OFF_OUT] + [f"def_{c}" for c in DEF_OUT]
)
HEAD_FEATURES = STATE_FEATURES + STRENGTH_FEATURES

# Residual bootstrap: predicted means are bucketed into this many bins and each
# bin keeps a fixed-size pool of real residuals to draw from. A pool preserves
# the long right tail of NFL gains, which no parametric error term does.
N_RESID_BINS = 10
POOL_SIZE = 4000

# Field-position buckets for the empirical next-drive-start tables.
YARDLINE_BINS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 101])

# Inside the twenty, gains are truncated by the goal line: a touchdown from the
# three is recorded as exactly three yards, so the real distribution has a point
# mass sitting on the end zone. Pooling those residuals with midfield plays
# smears that mass to just short of the goal line, and the simulator settles for
# field goals it should be scoring on. Residuals in the red zone are therefore
# drawn from plays at the same yard line -- to the yard inside the ten.
RZ_ZONE_EDGES = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 16, 21])
N_RZ_ZONES = len(RZ_ZONE_EDGES)  # 11 zones inside the 20, plus everything beyond


def rz_zone(yardline_100: np.ndarray) -> np.ndarray:
    """Zone index for the residual pools; the last index is 'outside the 20'."""
    return np.digitize(yardline_100, RZ_ZONE_EDGES)

XGB_KW = dict(
    n_estimators=300, max_depth=6, learning_rate=0.06,
    subsample=0.8, colsample_bytree=0.8, n_jobs=4,
    tree_method="hist", random_state=42,
)


# ---------------------------------------------------------------------------
# Play table
# ---------------------------------------------------------------------------

def team_strength_table(feat: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team) with that team's pregame strength.

    features.parquet is home/away-wide; the world model is possession-oriented,
    so this melts it into the frame a play actually sees.
    """
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({
            "game_id": feat["game_id"],
            "team": feat[f"{side}_team"],
            "is_home": 1 if side == "home" else 0,
            "off_epa8": feat[f"{side}_off_epa8"],
            "def_epa8": feat[f"{side}_def_epa8"],
            "def_epa8_opp": feat[f"{opp}_def_epa8"],
            "qb_val": feat[f"{side}_qb_val"],
            "elo_off": feat[f"elo_{side}"],
            "elo_def": feat[f"elo_{opp}"],
            # The offense's own absences, and the defense it is facing.
            **{f"off_{c}": feat[f"{side}_{c}"] for c in OFF_OUT},
            **{f"def_{c}": feat[f"{opp}_{c}"] for c in DEF_OUT},
        }))
    return pd.concat(rows, ignore_index=True)


def load_plays(seasons: list[int]) -> pd.DataFrame:
    """Scrimmage plays with the state before the snap and what happened."""
    keep = [
        "game_id", "season", "week", "play_id", "posteam", "defteam", "qtr",
        "down", "ydstogo", "yardline_100", "goal_to_go", "half_seconds_remaining",
        "game_seconds_remaining", "score_differential", "posteam_timeouts_remaining",
        "defteam_timeouts_remaining", "play_type", "yards_gained", "sack",
        "interception", "complete_pass", "incomplete_pass", "fumble_lost",
        "touchdown", "pass_touchdown", "rush_touchdown", "return_touchdown",
        "field_goal_attempt", "field_goal_result", "kick_distance", "punt_attempt",
        "kickoff_attempt", "extra_point_attempt", "extra_point_result",
        "two_point_attempt",
        "two_point_conv_result", "penalty", "penalty_yards", "penalty_team",
        "first_down", "safety", "aborted_play", "td_team", "game_half",
    ]
    frames = []
    for season in seasons:
        # The upcoming season has no play-by-play file until kickoff.
        try:
            pbp = nfl.load_pbp([season])
        except (ValueError, FileNotFoundError) as exc:
            print(f"  {season}: skipped ({exc})", file=sys.stderr)
            continue
        cols = [c for c in keep if c in pbp.columns]
        frames.append(pbp.select(cols).to_pandas())
        print(f"  {season}: {len(frames[-1]):,} rows", file=sys.stderr)
    if not frames:
        raise SystemExit("no play-by-play available for the requested seasons")
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype == object and c not in {
            "game_id", "posteam", "defteam", "play_type", "field_goal_result",
            "two_point_conv_result", "penalty_team", "td_team", "game_half",
        }:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_play_table(plays: pd.DataFrame, feat: pd.DataFrame) -> pd.DataFrame:
    """Join every play to its offense's pregame strength and derive the labels."""
    strength = team_strength_table(feat)
    df = plays.merge(strength, left_on=["game_id", "posteam"],
                     right_on=["game_id", "team"], how="inner")

    # Time consumed by a play *and* the huddle after it, which is what the
    # simulator needs to advance the clock: the gap to the next snap.
    df = df.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    nxt = df.groupby("game_id")["game_seconds_remaining"].shift(-1)
    df["elapsed"] = (df["game_seconds_remaining"] - nxt).clip(0, 120)

    # Collapse the feed's play_type into the classes the policy head chooses.
    pt = df["play_type"].fillna("")
    df["ptype"] = np.select(
        [
            pt.eq("no_play") & df["penalty"].fillna(0).gt(0),
            pt.isin(["run"]),
            pt.isin(["pass"]),
            pt.eq("punt"),
            pt.eq("field_goal"),
            pt.eq("qb_kneel"),
            pt.eq("qb_spike"),
        ],
        ["no_play", "run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"],
        default="",
    )

    df["pass_event"] = np.select(
        [df["sack"].fillna(0).gt(0), df["interception"].fillna(0).gt(0),
         df["complete_pass"].fillna(0).gt(0)],
        ["sack", "interception", "complete"],
        default="incomplete",
    )
    return df


# ---------------------------------------------------------------------------
# Empirical pools
# ---------------------------------------------------------------------------

def _pool(values: np.ndarray, rng: np.random.Generator, size: int = POOL_SIZE,
          what: str = "pool") -> np.ndarray:
    """A fixed-size resample of `values`, so sim-time draws are one index op.

    An empty input is a bug, not a zero: a pool of zeros looks like a working
    simulator that quietly puts every drive on the goal line.
    """
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise ValueError(f"no observations to build the {what} pool from")
    return rng.choice(values, size=size, replace=True)


def residual_pools(y: np.ndarray, pred: np.ndarray, yardline: np.ndarray,
                   rng: np.random.Generator) -> dict:
    """Bucket residuals so a sampled gain lands where a real one would.

    Two regimes. Beyond the twenty, gains are unconstrained and the spread
    depends on how big a play was expected -- a checkdown and a shot play sit
    very differently around the same expectation -- so residuals are bucketed by
    predicted mean. Inside the twenty the goal line truncates everything, so
    residuals are bucketed by the yard line instead and that structure survives.
    """
    resid = y - pred
    edges = np.quantile(pred, np.linspace(0, 1, N_RESID_BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf

    open_field = yardline > 20
    idx = np.clip(np.digitize(pred, edges[1:-1]), 0, N_RESID_BINS - 1)
    open_pools = np.stack([
        _pool(resid[open_field & (idx == b)], rng, what=f"open-field residual {b}")
        for b in range(N_RESID_BINS)])

    zone = rz_zone(yardline)
    rz_pools = np.stack([
        _pool(resid[zone == z], rng, what=f"red-zone residual {z}")
        for z in range(N_RZ_ZONES)])
    return {"edges": edges, "open": open_pools, "rz": rz_pools}


SCRIMMAGE = {"run", "pass", "punt", "field_goal", "qb_kneel", "qb_spike"}


def add_next_snap(plays: pd.DataFrame) -> pd.DataFrame:
    """Attach, to every row, the next real snap in that game.

    "Next row" is not good enough: penalties, extra points and timeouts sit
    between a possession ending and the next drive actually starting, and their
    yard lines describe something else. Looking forward to the next scrimmage
    play is what gives the drive's true starting spot.
    """
    df = plays.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    is_snap = df["play_type"].isin(SCRIMMAGE)
    snap_pos = df["posteam"].where(is_snap)
    snap_yl = df["yardline_100"].where(is_snap)
    grp = df.groupby("game_id")
    # bfill after a backward shift = "the next snap strictly after this row".
    df["next_pos"] = grp.apply(
        lambda g: snap_pos.loc[g.index].shift(-1).bfill(), include_groups=False
    ).reset_index(level=0, drop=True)
    df["next_yl"] = grp.apply(
        lambda g: snap_yl.loc[g.index].shift(-1).bfill(), include_groups=False
    ).reset_index(level=0, drop=True)
    return df


def next_start_table(plays: pd.DataFrame, mask: pd.Series, rng: np.random.Generator,
                     require_change: bool = True) -> np.ndarray:
    """Where the receiving team actually starts after a possession-ending play.

    Measured, not derived. Returns a (len(YARDLINE_BINS)-1, POOL_SIZE) matrix
    keyed by the field position the play happened from.

    `require_change` is False for kickoffs: nflverse marks the *receiving* team
    as posteam on a kick, so possession does not change across it.
    """
    ok = mask.reindex(plays.index).fillna(False) & \
        plays["next_pos"].notna() & plays["next_yl"].notna()
    if require_change:
        ok &= plays["next_pos"].ne(plays["posteam"])
    src, dst = plays.loc[ok, "yardline_100"].to_numpy(), plays.loc[ok, "next_yl"].to_numpy()
    bins = np.clip(np.digitize(src, YARDLINE_BINS[1:-1]), 0, len(YARDLINE_BINS) - 2)
    out = []
    for b in range(len(YARDLINE_BINS) - 1):
        vals = dst[bins == b]
        # Thin buckets (a punt from the opponent's 5 barely happens) fall back
        # to the league-wide distribution rather than inventing a number.
        out.append(_pool(vals if len(vals) >= 30 else dst, rng))
    return np.stack(out)


def clock_pools(plays: pd.DataFrame, rng: np.random.Generator) -> dict:
    """Seconds burned per play cycle, keyed by what stops the clock."""
    pools = {}
    late = plays["half_seconds_remaining"] < 300
    classes = {
        "run": plays["ptype"].eq("run"),
        "pass_complete": plays["ptype"].eq("pass") & plays["pass_event"].eq("complete"),
        "pass_incomplete": plays["ptype"].eq("pass") & plays["pass_event"].eq("incomplete"),
        "sack": plays["ptype"].eq("pass") & plays["pass_event"].eq("sack"),
        "punt": plays["ptype"].eq("punt"),
        "field_goal": plays["ptype"].eq("field_goal"),
        "qb_kneel": plays["ptype"].eq("qb_kneel"),
        "qb_spike": plays["ptype"].eq("qb_spike"),
        "no_play": plays["ptype"].eq("no_play"),
    }
    for name, m in classes.items():
        for lately, lm in (("late", late), ("early", ~late)):
            v = plays.loc[m & lm, "elapsed"].to_numpy()
            pools[(name, lately)] = _pool(v if len(v) >= 100 else
                                          plays.loc[m, "elapsed"].to_numpy(), rng)
    return pools


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _fit_classifier(df: pd.DataFrame, label: pd.Series, classes: list[str]):
    y = pd.Categorical(label, categories=classes).codes
    ok = y >= 0
    model = XGBClassifier(objective="multi:softprob", num_class=len(classes), **XGB_KW)
    model.fit(df.loc[ok, HEAD_FEATURES].to_numpy(float), y[ok])
    return model


def train(seasons: list[int] | None = None, out_path: str | None = None) -> dict:
    seasons = seasons or list(range(FIRST_SEASON, 2027))
    feat = pd.read_parquet("features.parquet")
    rng = np.random.default_rng(42)

    print("loading play-by-play...", file=sys.stderr)
    all_plays = add_next_snap(load_plays(seasons))
    plays = build_play_table(all_plays, feat)
    plays = plays[plays["posteam"].notna() & plays["down"].notna()]
    print(f"{len(plays):,} plays with a pregame feature join", file=sys.stderr)

    for c in HEAD_FEATURES:
        plays[c] = pd.to_numeric(plays[c], errors="coerce").fillna(0.0)

    bundle: dict = {"seasons": seasons, "features": HEAD_FEATURES}

    # 1. Coaching policy -----------------------------------------------------
    chooseable = plays[plays["ptype"].isin(PLAY_TYPES)]
    print(f"policy head on {len(chooseable):,} plays", file=sys.stderr)
    bundle["policy"] = _fit_classifier(chooseable, chooseable["ptype"], PLAY_TYPES)

    # 2. What happens on a pass ---------------------------------------------
    passes = plays[plays["ptype"].eq("pass")]
    print(f"pass head on {len(passes):,} plays", file=sys.stderr)
    bundle["pass_event"] = _fit_classifier(passes, passes["pass_event"], PASS_EVENTS)

    # 3. Yards ---------------------------------------------------------------
    comp = passes[passes["pass_event"].eq("complete")]
    m = XGBRegressor(objective="reg:squarederror", **XGB_KW)
    m.fit(comp[HEAD_FEATURES].to_numpy(float), comp["yards_gained"])
    bundle["pass_yards"] = m
    bundle["pass_yards_resid"] = residual_pools(
        comp["yards_gained"].to_numpy(), m.predict(comp[HEAD_FEATURES].to_numpy(float)),
        comp["yardline_100"].to_numpy(), rng)

    runs = plays[plays["ptype"].eq("run")]
    m = XGBRegressor(objective="reg:squarederror", **XGB_KW)
    m.fit(runs[HEAD_FEATURES].to_numpy(float), runs["yards_gained"])
    bundle["run_yards"] = m
    bundle["run_yards_resid"] = residual_pools(
        runs["yards_gained"].to_numpy(), m.predict(runs[HEAD_FEATURES].to_numpy(float)),
        runs["yardline_100"].to_numpy(), rng)

    # 4. Fumbles: rare enough that a rate per play class beats a model --------
    bundle["fumble_rate"] = {
        "run": float(runs["fumble_lost"].fillna(0).mean()),
        "complete": float(comp["fumble_lost"].fillna(0).mean()),
        "sack": float(passes.loc[passes["pass_event"].eq("sack"),
                                 "fumble_lost"].fillna(0).mean()),
    }

    # 5. Sack yardage and penalties -----------------------------------------
    bundle["sack_yards"] = _pool(
        passes.loc[passes["pass_event"].eq("sack"), "yards_gained"].to_numpy(), rng)
    pens = plays[plays["ptype"].eq("no_play")].copy()
    # Sign the penalty from the offense's point of view.
    signed = np.where(pens["penalty_team"].eq(pens["posteam"]),
                      -pens["penalty_yards"].fillna(0), pens["penalty_yards"].fillna(0))
    bundle["penalty_yards"] = _pool(signed.astype(float), rng)
    bundle["penalty_first_down"] = float(pens["first_down"].fillna(0).mean())

    # 6. Field goals: a logistic on distance, fit on real attempts -----------
    fgs = plays[plays["ptype"].eq("field_goal")].copy()
    fgs["made"] = fgs["field_goal_result"].eq("made").astype(int)
    fgs["dist"] = fgs["kick_distance"].fillna(fgs["yardline_100"] + 17)
    fg_ok = fgs["dist"].between(15, 75)
    from sklearn.linear_model import LogisticRegression
    fg = LogisticRegression()
    fg.fit(fgs.loc[fg_ok, ["dist"]], fgs.loc[fg_ok, "made"])
    bundle["fg"] = (float(fg.coef_[0][0]), float(fg.intercept_[0]))

    # 7. Where the next drive starts, measured off real games ----------------
    pt = all_plays["play_type"].fillna("")
    bundle["next_start"] = {
        "punt": next_start_table(all_plays, pt.eq("punt"), rng),
        "fg_miss": next_start_table(
            all_plays, pt.eq("field_goal") & all_plays["field_goal_result"].ne("made"), rng),
        "interception": next_start_table(all_plays, all_plays["interception"].fillna(0).gt(0), rng),
        "fumble": next_start_table(all_plays, all_plays["fumble_lost"].fillna(0).gt(0), rng),
        "downs": next_start_table(
            all_plays, all_plays["down"].eq(4) & all_plays["first_down"].fillna(0).eq(0)
            & pt.isin(["run", "pass"]), rng),
    }

    # Kickoffs come from the 2024 rule era only -- earlier kicks describe a
    # game that no longer exists.
    ko = all_plays[all_plays["season"] >= KICKOFF_ERA_FIRST_SEASON]
    if ko.empty:
        # Fitting a window that predates the rule change (an out-of-sample
        # test, say): use the newest seasons on hand rather than nothing.
        newest = sorted(all_plays["season"].unique())[-2:]
        ko = all_plays[all_plays["season"].isin(newest)]
        print(f"  kickoffs: no {KICKOFF_ERA_FIRST_SEASON}+ data, "
              f"using {newest}", file=sys.stderr)
    ko_next = next_start_table(ko, ko["kickoff_attempt"].fillna(0).gt(0), rng,
                               require_change=False)
    # Field position after a kickoff barely depends on where it was kicked from,
    # so collapse the buckets into one pool.
    bundle["kickoff_start"] = ko_next.reshape(-1)[
        rng.choice(ko_next.size, size=POOL_SIZE, replace=True)]

    # 8. Scoring plays -------------------------------------------------------
    xps = all_plays[all_plays["extra_point_attempt"].fillna(0).gt(0)]
    bundle["xp_rate"] = float(xps["extra_point_result"].eq("good").mean())
    twos = all_plays[all_plays["two_point_attempt"].fillna(0).gt(0)]
    bundle["two_point_rate"] = float(twos["two_point_conv_result"].eq("success").mean())
    bundle["return_td_rate"] = {
        "punt": float(all_plays.loc[pt.eq("punt"), "return_touchdown"].fillna(0).mean()),
        "interception": float(all_plays.loc[all_plays["interception"].fillna(0).gt(0),
                                            "return_touchdown"].fillna(0).mean()),
    }

    # 9. Clock ---------------------------------------------------------------
    bundle["clock"] = clock_pools(plays, rng)

    with open(out_path or MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"wrote {out_path or MODEL_PATH}", file=sys.stderr)
    _report(bundle)
    return bundle


def _report(b: dict) -> None:
    slope, intercept = b["fg"]
    print("\nfitted dynamics")
    print(f"  fumble lost: run {b['fumble_rate']['run']:.3%}  "
          f"completion {b['fumble_rate']['complete']:.3%}  sack {b['fumble_rate']['sack']:.3%}")
    for d in (25, 35, 45, 55):
        p = 1 / (1 + np.exp(-(slope * d + intercept)))
        print(f"  field goal {d}yd: {p:.1%}")
    print(f"  extra point {b['xp_rate']:.1%}   two-point {b['two_point_rate']:.1%}")
    print(f"  mean kickoff start: own {100 - b['kickoff_start'].mean():.1f}")
    print(f"  mean drive start after a punt: own "
          f"{100 - b['next_start']['punt'].mean():.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--first-season", type=int, default=FIRST_SEASON)
    ap.add_argument("--last-season", type=int, default=2026,
                    help="fit only through this season, for out-of-sample tests")
    ap.add_argument("-o", "--out", default=MODEL_PATH)
    args = ap.parse_args()
    train(list(range(args.first_season, args.last_season + 1)), out_path=args.out)
