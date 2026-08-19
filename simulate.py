"""Roll a football game forward, play by play, thousands of times at once.

world_model.py fits the dynamics; this drives them. Given the same pregame
feature row train.py scores, it starts at the opening kickoff and simulates the
whole game -- possession, field position, down and distance, clock and score --
until time runs out, then reads the win probability off the outcomes.

The reason to do this rather than classify the game directly:

  * the answer is a distribution, not a number. Win probability falls out, but
    so do the final margin and total, which a win/loss classifier cannot give.
  * it starts from *any* state, not just the kickoff, so the same model that
    prices a game on Thursday prices it again on 3rd and 7 in the fourth
    quarter. That is what `simulate_from` is for.
  * a counterfactual moves every snap of the game rather than one input. Ruling
    a quarterback out lowers the offense's expected yards on all sixty plays,
    and the drive structure absorbs it the way a real game would.

Everything is vectorised across replicates: one step advances all N copies of
the game together, so each XGBoost call scores a batch of thousands of states
instead of one. A game at n=5000 takes a few seconds rather than an hour.

    from simulate import Simulator
    sim = Simulator.load()
    out = sim.simulate_game(features_row, n=5000)
    out["home_win_prob"], out["margin"].mean()
"""

from __future__ import annotations

import os
import pickle
import zlib

import numpy as np
import pandas as pd

from world_model import (
    MODEL_PATH, HEAD_FEATURES, PLAY_TYPES, PASS_EVENTS,
    N_RESID_BINS, YARDLINE_BINS, OFF_OUT, DEF_OUT, rz_zone,
)

HALF_SECONDS = 1800
OT_SECONDS = 600            # regular-season overtime
MAX_STEPS = 400             # a safety stop; a real game is ~170 plays
KICKOFF_SECONDS = 5         # clock burned by a kick and its return, on average
DEFAULT_SIMS = 5000

# Drawing every play's residual independently makes a team's efficiency average
# out over a game, so simulated margins bunch far tighter than real ones: no
# offence ever simply has a bad day beyond what its pregame numbers said. Each
# replicate therefore gives each team one latent efficiency draw, in yards per
# play, held for the whole game. Tuned so the simulated margin spread matches
# the real one -- see sim_backtest.py, question 1.
try:
    SHOCK_SD = float(os.environ.get("WORLD_MODEL_SHOCK_SD", 0.5))
except ValueError:
    SHOCK_SD = 0.5

# Column positions in the feature matrix handed to the heads.
IDX = {name: i for i, name in enumerate(HEAD_FEATURES)}

# Two-point tries: trailing by 2, 5 or 10, or leading by 1, are the spots where
# the chart says go, and coaches only really follow it late.
TWO_POINT_DIFFS = (-2, -5, -10, 1)
TWO_POINT_CLOCK = 900       # last quarter


def game_seed(game_id: str) -> int:
    """A stable per-game seed.

    Python salts str.__hash__ per process, so hashing the game id would give a
    different simulation every run and the daily rebuild would print different
    numbers for a game nothing had changed about.
    """
    return zlib.crc32(str(game_id).encode()) & 0x7FFFFFFF


def _sample_categorical(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """One draw per row of a probability matrix."""
    cum = probs.cumsum(axis=1)
    u = rng.random((len(probs), 1)) * cum[:, -1:]
    return (u > cum).sum(axis=1)


def _sample_pool(pool: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    return pool[rng.integers(0, len(pool), size=size)]


def _sample_resid(entry: dict, pred: np.ndarray, yardline: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
    """Draw a gain residual from the pool that matches where the ball is."""
    out = np.empty(len(pred))
    inside = yardline <= 20
    if inside.any():
        z = rz_zone(yardline[inside])
        pools = entry["rz"]
        out[inside] = pools[z, rng.integers(0, pools.shape[1], size=int(inside.sum()))]
    if (~inside).any():
        b = np.clip(np.digitize(pred[~inside], entry["edges"][1:-1]), 0, N_RESID_BINS - 1)
        pools = entry["open"]
        out[~inside] = pools[b, rng.integers(0, pools.shape[1], size=int((~inside).sum()))]
    return out


def _sample_start(table: np.ndarray, yl: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Draw the receiving team's starting yard line from an empirical table."""
    b = np.clip(np.digitize(yl, YARDLINE_BINS[1:-1]), 0, table.shape[0] - 1)
    j = rng.integers(0, table.shape[1], size=len(yl))
    return table[b, j]


class Simulator:
    def __init__(self, bundle: dict):
        self.b = bundle
        slope, intercept = bundle["fg"]
        self._fg_slope, self._fg_intercept = slope, intercept

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "Simulator":
        # world_model.py writes this file and only this process reads it.
        # Unpickling executes code, so never point this at a path that anything
        # outside the pipeline can write -- it is not a data format.
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    # -- helpers ----------------------------------------------------------

    def _fg_prob(self, distance: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-(self._fg_slope * distance + self._fg_intercept)))

    def _clock(self, key: str, late: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Seconds burned, drawn from the pool matching the clock situation."""
        out = np.empty(len(late))
        for lately in (True, False):
            m = late == lately
            if m.any():
                pool = self.b["clock"][(key, "late" if lately else "early")]
                out[m] = _sample_pool(pool, int(m.sum()), rng)
        return out

    @staticmethod
    def strength_rows(row: pd.Series) -> np.ndarray:
        """The strength half of the feature vector, one row per team in possession."""
        out = np.zeros((2, len(HEAD_FEATURES)))
        for i, (side, opp) in enumerate((("home", "away"), ("away", "home"))):
            out[i, IDX["is_home"]] = 1.0 if side == "home" else 0.0
            out[i, IDX["off_epa8"]] = row[f"{side}_off_epa8"]
            out[i, IDX["def_epa8_opp"]] = row[f"{opp}_def_epa8"]
            out[i, IDX["qb_val"]] = row[f"{side}_qb_val"]
            out[i, IDX["elo_off"]] = row[f"elo_{side}"]
            out[i, IDX["elo_def"]] = row[f"elo_{opp}"]
            for c in OFF_OUT:
                out[i, IDX[f"off_{c}"]] = row[f"{side}_{c}"]
            for c in DEF_OUT:
                out[i, IDX[f"def_{c}"]] = row[f"{opp}_{c}"]
        return np.nan_to_num(out)

    # -- the loop ---------------------------------------------------------

    def simulate_game(self, row: pd.Series, n: int = DEFAULT_SIMS,
                      seed: int | None = None) -> dict:
        """Simulate a full game from the opening kickoff."""
        rng = np.random.default_rng(seed)
        # The home team receives the opening kickoff in half the replicates;
        # the other team receives to start the second half.
        recv = rng.integers(0, 2, size=n)
        state = dict(
            pos=recv.copy(), yl=np.full(n, 70.0), down=np.ones(n, int),
            dist=np.full(n, 10.0), half_clock=np.full(n, float(HALF_SECONDS)),
            half=np.ones(n, int), score_h=np.zeros(n, int), score_a=np.zeros(n, int),
            to_h=np.full(n, 3), to_a=np.full(n, 3),
            live=np.ones(n, bool), kickoff=np.ones(n, bool),
            second_half_recv=1 - recv, ot_possessions=np.zeros(n, int),
        )
        return self._run(state, row, rng)

    def simulate_from(self, row: pd.Series, *, pos_is_home: bool, yardline_100: float,
                      down: int, ydstogo: float, half_seconds_remaining: float,
                      half: int, home_score: int, away_score: int,
                      home_timeouts: int = 3, away_timeouts: int = 3,
                      n: int = DEFAULT_SIMS, seed: int | None = None) -> dict:
        """Simulate the rest of a game from a live state.

        This is what makes the model useful in-game: hand it the situation off
        the scoreboard feed and it prices the game from there.
        """
        rng = np.random.default_rng(seed)
        state = dict(
            pos=np.full(n, 0 if pos_is_home else 1),
            yl=np.full(n, float(yardline_100)), down=np.full(n, int(down)),
            dist=np.full(n, float(ydstogo)),
            half_clock=np.full(n, float(half_seconds_remaining)),
            half=np.full(n, int(half)),
            score_h=np.full(n, int(home_score)), score_a=np.full(n, int(away_score)),
            to_h=np.full(n, int(home_timeouts)), to_a=np.full(n, int(away_timeouts)),
            live=np.ones(n, bool), kickoff=np.zeros(n, bool),
            second_half_recv=rng.integers(0, 2, size=n),
            ot_possessions=np.full(n, 2 if half >= 3 else 0),
        )
        return self._run(state, row, rng)

    def _run(self, s: dict, row: pd.Series, rng: np.random.Generator) -> dict:
        b, n = self.b, len(s["pos"])
        strength = self.strength_rows(row)
        # One efficiency draw per team per replicate, held all game.
        shock = rng.normal(0.0, SHOCK_SD, size=(n, 2))
        # Counts of what the simulator actually did, for validation.
        self.trace = {"steps": 0, "plays": 0, "calls": {}, "elapsed": [],
                      "yards": {}, "events": {}, "ends": {}}

        for _ in range(MAX_STEPS):
            if not s["live"].any():
                break

            # --- kickoffs ----------------------------------------------------
            ko = s["live"] & s["kickoff"]
            if ko.any():
                s["yl"][ko] = _sample_pool(b["kickoff_start"], int(ko.sum()), rng)
                s["down"][ko] = 1
                s["dist"][ko] = 10.0
                s["half_clock"][ko] -= KICKOFF_SECONDS
                s["kickoff"][ko] = False
                self._check_half_end(s, rng)
                if not s["live"].any():
                    break

            idx = np.flatnonzero(s["live"] & ~s["kickoff"])
            if len(idx) == 0:
                continue

            X = self._features(s, idx, strength)

            # --- what does the offense call? --------------------------------
            probs = b["policy"].predict_proba(X)
            call = _sample_categorical(probs, rng)
            ptype = np.array(PLAY_TYPES, dtype=object)[call]
            self.trace["steps"] += 1
            self.trace["plays"] += len(idx)
            for t, c in zip(*np.unique(ptype, return_counts=True)):
                self.trace["calls"][t] = self.trace["calls"].get(t, 0) + int(c)

            # Yards gained (signed, offense's view) and how the down ends.
            yards = np.zeros(len(idx))
            turnover = np.zeros(len(idx), bool)
            turnover_kind = np.empty(len(idx), dtype=object)
            return_td = np.zeros(len(idx), bool)
            free_down = np.zeros(len(idx), bool)     # penalties do not use a down
            clock_key = np.array(["run"] * len(idx), dtype=object)

            # --- runs --------------------------------------------------------
            m = ptype == "run"
            if m.any():
                rows = idx[m]
                mu = b["run_yards"].predict(X[m]) + shock[rows, s["pos"][rows]]
                yards[m] = np.round(mu + _sample_resid(
                    b["run_yards_resid"], mu, s["yl"][rows], rng))
                lost = rng.random(int(m.sum())) < b["fumble_rate"]["run"]
                mi = np.flatnonzero(m)
                turnover[mi[lost]] = True
                turnover_kind[mi[lost]] = "fumble"
                clock_key[m] = "run"

            # --- passes ------------------------------------------------------
            m = ptype == "pass"
            if m.any():
                mi = np.flatnonzero(m)
                ev = _sample_categorical(b["pass_event"].predict_proba(X[m]), rng)
                ev = np.array(PASS_EVENTS, dtype=object)[ev]
                for t, c in zip(*np.unique(ev, return_counts=True)):
                    self.trace["events"][t] = self.trace["events"].get(t, 0) + int(c)

                comp = ev == "complete"
                if comp.any():
                    ci = mi[comp]
                    crows = idx[ci]
                    mu = b["pass_yards"].predict(X[ci]) + shock[crows, s["pos"][crows]]
                    yards[ci] = np.round(mu + _sample_resid(
                        b["pass_yards_resid"], mu, s["yl"][crows], rng))
                    lost = rng.random(len(ci)) < b["fumble_rate"]["complete"]
                    turnover[ci[lost]] = True
                    turnover_kind[ci[lost]] = "fumble"
                    clock_key[ci] = "pass_complete"

                inc = ev == "incomplete"
                clock_key[mi[inc]] = "pass_incomplete"

                sack = ev == "sack"
                if sack.any():
                    si = mi[sack]
                    yards[si] = _sample_pool(b["sack_yards"], len(si), rng)
                    lost = rng.random(len(si)) < b["fumble_rate"]["sack"]
                    turnover[si[lost]] = True
                    turnover_kind[si[lost]] = "fumble"
                    clock_key[si] = "sack"

                pick = ev == "interception"
                if pick.any():
                    pi = mi[pick]
                    turnover[pi] = True
                    turnover_kind[pi] = "interception"
                    ret = rng.random(len(pi)) < b["return_td_rate"]["interception"]
                    return_td[pi[ret]] = True
                    clock_key[pi] = "pass_incomplete"

            # --- punts -------------------------------------------------------
            m = ptype == "punt"
            if m.any():
                mi = np.flatnonzero(m)
                turnover[mi] = True
                turnover_kind[mi] = "punt"
                ret = rng.random(len(mi)) < b["return_td_rate"]["punt"]
                return_td[mi[ret]] = True
                clock_key[mi] = "punt"

            # --- field goals --------------------------------------------------
            m = ptype == "field_goal"
            fg_good = np.zeros(len(idx), bool)
            if m.any():
                mi = np.flatnonzero(m)
                dist_yd = s["yl"][idx[mi]] + 17.0        # snap + hold behind the LOS
                good = rng.random(len(mi)) < self._fg_prob(dist_yd)
                fg_good[mi[good]] = True
                turnover[mi[~good]] = True
                turnover_kind[mi[~good]] = "fg_miss"
                clock_key[mi] = "field_goal"

            # --- kneels, spikes, penalties ------------------------------------
            m = ptype == "qb_kneel"
            if m.any():
                yards[m] = -1.0
                clock_key[m] = "qb_kneel"
            m = ptype == "qb_spike"
            if m.any():
                clock_key[m] = "qb_spike"
            m = ptype == "no_play"
            if m.any():
                mi = np.flatnonzero(m)
                yards[mi] = _sample_pool(b["penalty_yards"], len(mi), rng)
                # Half-the-distance: a flag never marches an offense into its
                # own end zone, so it can never be scored as a safety.
                yards[mi] = np.maximum(yards[mi], s["yl"][idx[mi]] - 99.0)
                free_down[mi] = True
                auto_first = rng.random(len(mi)) < b["penalty_first_down"]
                clock_key[mi] = "no_play"
                # An automatic first down overrides the usual down bookkeeping.
                free_down[mi[auto_first]] = False
                yards[mi[auto_first]] = np.maximum(
                    yards[mi[auto_first]], s["dist"][idx[mi[auto_first]]])

            yl_now = s["yl"][idx]
            self.trace["rz"] = self.trace.get("rz", 0) + int((yl_now <= 20).sum())
            self.trace["rz_first"] = self.trace.get("rz_first", 0) + int(
                ((yl_now <= 20) & (s["down"][idx] == 1)).sum())
            self.trace["gain20"] = self.trace.get("gain20", 0) + int((yards >= 20).sum())
            self.trace["scrim"] = self.trace.get("scrim", 0) + int(
                np.isin(ptype, ["run", "pass"]).sum())
            for t in set(ptype.tolist()):
                tm = ptype == t
                y = self.trace["yards"].setdefault(t, [0.0, 0])
                y[0] += float(yards[tm].sum()); y[1] += int(tm.sum())
            for k in ("interception", "fumble", "punt", "fg_miss"):
                self.trace["ends"][k] = self.trace["ends"].get(k, 0) + \
                    int((turnover_kind == k).sum())
            self.trace["ends"]["fg_good"] = self.trace["ends"].get("fg_good", 0) + int(fg_good.sum())
            self.trace["ends"]["td"] = self.trace["ends"].get("td", 0) + \
                int(((~turnover) & ((s["yl"][idx] - yards) <= 0)).sum())

            # --- clock --------------------------------------------------------
            late = s["half_clock"][idx] < 300
            elapsed = np.zeros(len(idx))
            for key in set(clock_key.tolist()):
                km = clock_key == key
                elapsed[km] = self._clock(key, late[km], rng)
            s["half_clock"][idx] -= elapsed
            self.trace["elapsed"].append(float(elapsed.mean()))

            # --- resolve the new state ----------------------------------------
            prev_pos = s["pos"].copy()
            self._apply(s, idx, yards, turnover, turnover_kind, fg_good,
                        return_td, free_down, rng)
            changed = (s["pos"] != prev_pos) & (s["half"] >= 3)
            s["ot_possessions"][changed] += 1
            self._check_half_end(s, rng)
            self._check_ot_end(s)

        margin = s["score_h"] - s["score_a"]
        return {
            "home_score": s["score_h"], "away_score": s["score_a"],
            "margin": margin, "total": s["score_h"] + s["score_a"],
            "home_win_prob": float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
            "tie_prob": float((margin == 0).mean()),
            "n": n,
        }

    def _features(self, s: dict, idx: np.ndarray, strength: np.ndarray) -> np.ndarray:
        """Assemble the head input for the live replicates."""
        pos = s["pos"][idx]
        X = strength[pos].copy()
        X[:, IDX["down"]] = s["down"][idx]
        X[:, IDX["ydstogo"]] = s["dist"][idx]
        X[:, IDX["yardline_100"]] = s["yl"][idx]
        X[:, IDX["goal_to_go"]] = (s["dist"][idx] >= s["yl"][idx]).astype(float)
        X[:, IDX["half_seconds_remaining"]] = s["half_clock"][idx]
        X[:, IDX["game_seconds_remaining"]] = np.where(
            s["half"][idx] == 1, s["half_clock"][idx] + HALF_SECONDS,
            np.where(s["half"][idx] == 2, s["half_clock"][idx], 0.0))
        off_score = np.where(pos == 0, s["score_h"][idx], s["score_a"][idx])
        def_score = np.where(pos == 0, s["score_a"][idx], s["score_h"][idx])
        X[:, IDX["score_differential"]] = off_score - def_score
        X[:, IDX["posteam_timeouts_remaining"]] = np.where(
            pos == 0, s["to_h"][idx], s["to_a"][idx])
        X[:, IDX["defteam_timeouts_remaining"]] = np.where(
            pos == 0, s["to_a"][idx], s["to_h"][idx])
        # Quarter, derived from the half clock so the heads see the same
        # variable the training rows carried.
        q_in_half = np.where(s["half_clock"][idx] > 900, 1, 2)
        X[:, IDX["qtr"]] = np.where(s["half"][idx] >= 3, 5,
                                    (s["half"][idx] - 1) * 2 + q_in_half)
        return X

    def _score(self, s: dict, who: np.ndarray, rows: np.ndarray, points: int) -> None:
        """Add points to the team identified by `who` (0 home, 1 away)."""
        h = rows[who == 0]
        a = rows[who == 1]
        s["score_h"][h] += points
        s["score_a"][a] += points

    def _apply(self, s, idx, yards, turnover, kind, fg_good, return_td,
               free_down, rng) -> None:
        b = self.b
        pos = s["pos"][idx]
        new_yl = s["yl"][idx] - yards

        # --- touchdowns ---------------------------------------------------
        td = (~turnover) & (new_yl <= 0)
        if td.any():
            rows, who = idx[td], pos[td]
            self._score(s, who, rows, 6)
            self._convert(s, rows, who, rng)
            s["kickoff"][rows] = True
            s["pos"][rows] = 1 - who          # the other team receives
            s["down"][rows] = 1
            s["dist"][rows] = 10.0

        # --- return touchdowns off a turnover ------------------------------
        rtd = turnover & return_td
        if rtd.any():
            rows, who = idx[rtd], 1 - pos[rtd]
            self._score(s, who, rows, 6)
            self._convert(s, rows, who, rng)
            s["kickoff"][rows] = True
            s["pos"][rows] = 1 - who
            s["down"][rows] = 1
            s["dist"][rows] = 10.0

        # --- made field goals ----------------------------------------------
        if fg_good.any():
            rows, who = idx[fg_good], pos[fg_good]
            self._score(s, who, rows, 3)
            s["kickoff"][rows] = True
            s["pos"][rows] = 1 - who
            s["down"][rows] = 1
            s["dist"][rows] = 10.0

        # --- safeties --------------------------------------------------------
        safety = (~turnover) & (~td) & (new_yl >= 100)
        if safety.any():
            rows, who = idx[safety], 1 - pos[safety]
            self._score(s, who, rows, 2)
            s["kickoff"][rows] = True
            s["pos"][rows] = who              # the scoring team receives the free kick
            s["down"][rows] = 1
            s["dist"][rows] = 10.0

        # --- turnovers that stay on the field ---------------------------------
        to_field = turnover & ~return_td
        if to_field.any():
            rows = idx[to_field]
            spot = np.clip(new_yl[to_field], 1, 99)
            for key in ("punt", "fg_miss", "interception", "fumble"):
                km = kind[to_field] == key
                if km.any():
                    s["yl"][rows[km]] = _sample_start(b["next_start"][key], spot[km], rng)
            s["pos"][rows] = 1 - pos[to_field]
            s["down"][rows] = 1
            s["dist"][rows] = 10.0

        # --- ordinary plays ----------------------------------------------------
        plain = (~turnover) & (~td) & (~safety)
        if plain.any():
            rows = idx[plain]
            yl = np.clip(new_yl[plain], 1, 99)
            gained = yards[plain] >= s["dist"][rows]
            s["yl"][rows] = yl
            # A penalty does not burn a down.
            nd = np.where(free_down[plain], s["down"][rows], s["down"][rows] + 1)
            nd = np.where(gained, 1, nd)
            nl = np.where(gained, np.minimum(10.0, yl),
                          np.maximum(1.0, s["dist"][rows] - yards[plain]))
            # Fourth down came up short: the defense takes over at the spot.
            lost_downs = nd > 4
            if lost_downs.any():
                lr = rows[lost_downs]
                s["yl"][lr] = _sample_start(b["next_start"]["downs"], yl[lost_downs], rng)
                s["pos"][lr] = 1 - s["pos"][lr]
                nd[lost_downs] = 1
                nl[lost_downs] = 10.0
            s["down"][rows] = nd
            s["dist"][rows] = nl

    def _convert(self, s: dict, rows: np.ndarray, who: np.ndarray,
                 rng: np.random.Generator) -> None:
        """Extra point or two-point try, on the chart when it is late."""
        b = self.b
        own = np.where(who == 0, s["score_h"][rows], s["score_a"][rows])
        opp = np.where(who == 0, s["score_a"][rows], s["score_h"][rows])
        diff = own - opp
        late = (s["half"][rows] >= 2) & (s["half_clock"][rows] < TWO_POINT_CLOCK)
        go_for_two = late & np.isin(diff, TWO_POINT_DIFFS)

        made = np.zeros(len(rows), bool)
        one = ~go_for_two
        if one.any():
            made[one] = rng.random(int(one.sum())) < b["xp_rate"]
        if go_for_two.any():
            made[go_for_two] = rng.random(int(go_for_two.sum())) < b["two_point_rate"]

        pts = np.where(go_for_two, 2, 1) * made
        for p in (1, 2):
            m = (pts == p)
            if m.any():
                self._score(s, who[m], rows[m], p)

    def _check_half_end(self, s: dict, rng: np.random.Generator) -> None:
        """Roll the clock over between halves, into overtime, and off the game."""
        over = s["live"] & (s["half_clock"] <= 0)
        if not over.any():
            return
        # Snapshot the masks before mutating `half` -- promoting a replicate to
        # the second half must not make it look like the second half just ended.
        first = over & (s["half"] == 1)
        second = over & (s["half"] == 2)
        overtime = over & (s["half"] >= 3)

        # First half ends: the other team receives to start the third quarter.
        if first.any():
            s["half"][first] = 2
            s["half_clock"][first] = HALF_SECONDS
            s["pos"][first] = s["second_half_recv"][first]
            s["kickoff"][first] = True
            s["down"][first] = 1
            s["dist"][first] = 10.0
            s["to_h"][first] = 3
            s["to_a"][first] = 3

        # Regulation ends: settled games are done, ties go to overtime.
        if second.any():
            tied = second & (s["score_h"] == s["score_a"])
            s["live"][second & ~tied] = False
            if tied.any():
                s["half"][tied] = 3
                s["half_clock"][tied] = OT_SECONDS
                s["pos"][tied] = rng.integers(0, 2, size=int(tied.sum()))
                s["kickoff"][tied] = True
                s["down"][tied] = 1
                s["dist"][tied] = 10.0
                s["to_h"][tied] = 2
                s["to_a"][tied] = 2
                s["ot_possessions"][tied] = 0

        # Overtime expires: a tie is a real result.
        s["live"][overtime] = False

    def _check_ot_end(self, s: dict) -> None:
        """Sudden death: once both teams have had the ball, the next lead ends it.

        This is the rule as it stands -- both sides are guaranteed a possession
        even if the first drive ends in a touchdown.
        """
        done = (s["live"] & (s["half"] >= 3) & (s["ot_possessions"] >= 2)
                & (s["score_h"] != s["score_a"]))
        s["live"][done] = False
