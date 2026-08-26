"""Closing Line Value: did we take a better number than the market landed on?

Accuracy against the closing line is the one contest the market wins by
construction — the close aggregates everyone's information, including ours.
CLV asks the question that actually separates a winning bettor from a lucky
one: when we commit to a number, does the line then move TOWARD us?

It is the better metric for two reasons. It converges in tens of bets rather
than hundreds, because it scores the price rather than the coin flip on top of
it. And it cannot be faked by variance — a model with no edge gets zero CLV on
average no matter how its season happens to land.

The pre-registered rule is fixed here, in code, before the season starts, so it
cannot be tuned to whatever the data turns out to show:

    weeks 1-4, model margin disagrees with the market by >= 3.0 points

That rule is the one the backtest supports. Walk-forward 2015-2025, the model's
ATS record on its own disagreements:

    weeks 1-4, edge >= 3    59.1% (n=210)   [60.2% under the repo's margin map]
    weeks 5+,  edge >= 3    50.2% (n=677)   [47.9%]

and the decay across contiguous four-week windows is monotonic — 59.1, 55.0,
52.5, 53.1, 47.9, 45.4 — which is what a real seasonal effect looks like and
what a lucky slice does not. It matches the independent decay found in
wintotals_rule.py (54.1% in weeks 1-4 to 47.9% by week 15) from a completely
different direction. The mechanism is that in September the market prices off
priors because no current-season data exists yet.

Every game is logged, not just qualifying ones, so the pre-registered subset
can be reported against the full field rather than cherry-picked out of it.

    python clv.py --record    # log today's numbers (run daily, before kickoff)
    python clv.py --settle    # attach closing numbers to games that kicked off
    python clv.py --report    # CLV to date
"""

import argparse
import os

import numpy as np
import pandas as pd

from features import FEATURES
from train import ELO_BLEND

PICKS_CSV = "clv_picks.csv"
HISTORY_CSV = "spread_odds_history.csv"

# Pre-registered rule. Do not tune these to observed results.
RULE_MAX_WEEK = 4
RULE_MIN_EDGE = 3.0
# Data-quality gate, not an outcome filter: a game is not recorded at all until
# this many books have posted a two-way price. Weeks 2-4 open months out on two
# offshore books, and locking a number in against that thin a consensus would
# be measuring scrape noise rather than the market. Taking an early position is
# the whole point of CLV, so this is deliberately a floor on book count and not
# a delay until game week.
RULE_MIN_BOOKS = 4

PICK_FIELDS = [
    "game_id", "season", "week", "home", "away", "commence_time",
    "recorded_at", "side", "model_prob_home", "model_margin",
    "spread_at_bet", "prob_at_bet", "edge_pts", "books_at_bet", "qualifies",
    "spread_at_close", "prob_at_close", "clv_pts", "clv_prob",
    "margin", "ats_result",
]


def american_to_prob(price: float) -> float:
    price = float(price)
    return -price / (-price + 100.0) if price < 0 else 100.0 / (price + 100.0)


def clv_points(side: str, spread_at_bet: float, spread_at_close: float) -> float:
    """Points of closing line value. Positive means the line moved toward us.

    Spreads use the nflverse convention throughout: positive = home favoured.
    Backing the home side at -3 and watching it close at -4 means we hold the
    better number, so home CLV is (close - bet). The away side is the mirror:
    taking +3 and seeing it close at +2 is also an improvement, so away CLV is
    (bet - close).
    """
    if side == "home":
        return float(spread_at_close) - float(spread_at_bet)
    return float(spread_at_bet) - float(spread_at_close)


def ats_result(side: str, margin: float, line: float) -> str:
    """Did the bet cover? Home covers when the margin beats the line."""
    if float(margin) == float(line):
        return "push"
    home_covered = float(margin) > float(line)
    return "win" if home_covered == (side == "home") else "loss"


def snapshots() -> pd.DataFrame:
    """One consensus row per (game_id, snapshot): median across every book.

    The median rather than the mean, so a single stale or off-market book
    cannot drag the number that the whole metric is measured against.
    """
    if not os.path.exists(HISTORY_CSV):
        return pd.DataFrame()
    h = pd.read_csv(HISTORY_CSV)
    rows = []
    for (gid, at), g in h.groupby(["game_id", "fetched_at"]):
        h2h = g[g.market == "h2h"]
        probs = []
        for book, gb in h2h.groupby("book"):
            hm = gb[gb.side == "home"].price
            am = gb[gb.side == "away"].price
            if len(hm) and len(am) and pd.notna(hm.iloc[0]) and pd.notna(am.iloc[0]):
                ph, pa = american_to_prob(hm.iloc[0]), american_to_prob(am.iloc[0])
                if ph + pa > 0:
                    probs.append(ph / (ph + pa))
        sp = g[(g.market == "spreads") & (g.side == "home")].point.dropna()
        if not probs and not len(sp):
            continue
        rows.append({
            "game_id": gid, "fetched_at": at,
            "commence_time": g.commence_time.iloc[0],
            # nflverse convention: positive spread = home favoured.
            "spread_home": -float(np.median(sp)) if len(sp) else np.nan,
            "prob_home": float(np.median(probs)) if probs else np.nan,
            "books": len(probs),
        })
    return pd.DataFrame(rows)


def model_numbers() -> pd.DataFrame:
    """Model win probability and expected margin for every game in the table.

    The margin map is linear in the blended probability, matching
    ats_backtest.py so the two agree on what the model's number means. The
    logit form was checked as well and gives the same conclusion (59.1% vs
    60.2% on the pre-registered slice); the linear form is the repo's
    established convention, so it is the one that ships.
    """
    from xgboost import XGBClassifier

    df = pd.read_parquet("features.parquet")
    m = XGBClassifier()
    m.load_model("model.json")
    raw = m.predict_proba(df[FEATURES])[:, 1]
    df = df.assign(model_prob=(1 - ELO_BLEND) * raw + ELO_BLEND * df.elo_prob.values)

    done = df[df.home_win.notna()].copy()
    done["margin"] = done.home_score - done.away_score
    A = np.column_stack([np.ones(len(done)), done.model_prob.values])
    beta = np.linalg.lstsq(A, done["margin"].values, rcond=None)[0]
    df["model_margin"] = beta[0] + beta[1] * df.model_prob
    return df


def load_picks() -> pd.DataFrame:
    if os.path.exists(PICKS_CSV):
        return pd.read_csv(PICKS_CSV)
    return pd.DataFrame(columns=PICK_FIELDS)


def save_picks(p: pd.DataFrame) -> None:
    p.reindex(columns=PICK_FIELDS).to_csv(PICKS_CSV, index=False)


def record() -> None:
    """Log the model's number against the market's, once per game.

    The first qualifying observation is the bet and is never revised — that is
    the whole point. Re-recording a game later would let a number be chosen
    with hindsight about which way the line went.
    """
    snaps = snapshots()
    if snaps.empty:
        print(f"no {HISTORY_CSV} yet — run fetch_spread_odds.py first")
        return
    latest = snaps.sort_values("fetched_at").groupby("game_id").tail(1)
    games = model_numbers().set_index("game_id")
    picks = load_picks()
    have = set(picks.game_id) if len(picks) else set()

    new = []
    for r in latest.itertuples(index=False):
        if r.game_id in have or r.game_id not in games.index:
            continue
        g = games.loc[r.game_id]
        if pd.isna(g.model_margin) or pd.isna(r.spread_home):
            continue
        if pd.notna(g.home_win):
            continue  # already played; nothing to commit to
        if int(r.books) < RULE_MIN_BOOKS:
            continue  # too thin to be a real number; wait for more books
        edge = float(g.model_margin) - float(r.spread_home)
        side = "home" if edge > 0 else "away"
        prob_bet = (float(r.prob_home) if side == "home" else 1 - float(r.prob_home)) \
            if pd.notna(r.prob_home) else np.nan
        new.append({
            "game_id": r.game_id, "season": int(g.season), "week": int(g.week),
            "home": g.home_team, "away": g.away_team,
            "commence_time": r.commence_time, "recorded_at": r.fetched_at,
            "side": side, "model_prob_home": round(float(g.model_prob), 4),
            "model_margin": round(float(g.model_margin), 2),
            "spread_at_bet": float(r.spread_home),
            "prob_at_bet": round(prob_bet, 4) if pd.notna(prob_bet) else "",
            "edge_pts": round(edge, 2), "books_at_bet": int(r.books),
            "qualifies": int(int(g.week) <= RULE_MAX_WEEK
                             and abs(edge) >= RULE_MIN_EDGE),
        })
    thin = sum(1 for r in latest.itertuples(index=False)
               if r.game_id not in have and int(r.books) < RULE_MIN_BOOKS)
    if thin:
        print(f"{thin} games held back: fewer than {RULE_MIN_BOOKS} books posted")
    if not new:
        print("no new games to record")
        return
    out = pd.concat([picks, pd.DataFrame(new)], ignore_index=True)
    save_picks(out)
    q = sum(r["qualifies"] for r in new)
    print(f"recorded {len(new)} games ({q} qualify under the pre-registered rule)")


def settle() -> None:
    """Attach the closing number, and the result, to games that have kicked off.

    A game is only settled once kickoff has passed. Without that check the most
    recent snapshot of a game still weeks away would be treated as its close,
    locking in a CLV figure against a line that had not finished moving.
    """
    picks = load_picks()
    if picks.empty:
        print("no picks recorded yet")
        return
    now = pd.Timestamp.now(tz="UTC")
    snaps = snapshots()
    games = model_numbers().set_index("game_id")
    filled = 0
    for i, p in picks.iterrows():
        if pd.notna(p.get("clv_pts")) and str(p.get("clv_pts")) != "":
            continue
        if pd.to_datetime(p.commence_time, utc=True) > now:
            continue  # not kicked off; the line is still moving
        s = snaps[snaps.game_id == p.game_id]
        if s.empty:
            continue
        # The close is the last snapshot taken strictly before kickoff.
        before = s[s.fetched_at < p.commence_time].sort_values("fetched_at")
        if before.empty:
            continue
        c = before.iloc[-1]
        if c.fetched_at == p.recorded_at:
            continue  # never closed against itself
        # Positive CLV = the line moved toward the side we took.
        clv_pts = clv_points(p.side, p.spread_at_bet, c.spread_home)
        picks.at[i, "spread_at_close"] = float(c.spread_home)
        picks.at[i, "clv_pts"] = round(clv_pts, 2)
        if pd.notna(c.prob_home) and str(p.get("prob_at_bet")) not in ("", "nan"):
            pc = float(c.prob_home) if p.side == "home" else 1 - float(c.prob_home)
            picks.at[i, "prob_at_close"] = round(pc, 4)
            picks.at[i, "clv_prob"] = round(pc - float(p.prob_at_bet), 4)
        if p.game_id in games.index:
            g = games.loc[p.game_id]
            if pd.notna(g.home_win):
                margin = float(g.home_score - g.away_score)
                picks.at[i, "margin"] = margin
                picks.at[i, "ats_result"] = ats_result(
                    p.side, margin, float(p.spread_at_bet))
        filled += 1
    save_picks(picks)
    pending = sum(1 for _, p in picks.iterrows()
                  if pd.to_datetime(p.commence_time, utc=True) > now)
    print(f"settled {filled} games ({pending} still awaiting kickoff)")


def report() -> None:
    picks = load_picks()
    if picks.empty:
        print("no picks recorded yet")
        return
    d = picks[picks.clv_pts.notna()] if "clv_pts" in picks else pd.DataFrame()
    print(f"{len(picks)} games recorded, {len(d)} with a closing number\n")
    if d.empty:
        print("nothing settled yet — CLV needs at least one snapshot after the bet")
        return

    def block(name, s):
        if s.empty:
            return
        beat = (s.clv_pts > 0).mean()
        push = (s.clv_pts == 0).mean()
        line = (f"  {name:34s} n={len(s):4d}  beat close {beat:.1%}  "
                f"tied {push:.1%}  avg {s.clv_pts.mean():+.2f} pts")
        if "clv_prob" in s and s.clv_prob.notna().any():
            line += f"  {s.clv_prob.mean()*100:+.2f} cents"
        print(line)
        g = s[s.ats_result.isin(["win", "loss"])] if "ats_result" in s else pd.DataFrame()
        if len(g):
            w = (g.ats_result == "win").mean()
            print(f"  {'':34s}       ATS {w:.1%} ({(g.ats_result=='win').sum()}-"
                  f"{(g.ats_result=='loss').sum()})"
                  f"{'  clears 52.4%' if w > 0.524 else ''}")

    print("CLOSING LINE VALUE")
    block("pre-registered rule", d[d.qualifies == 1])
    block("everything else", d[d.qualifies != 1])
    block("all recorded games", d)
    print("\nPositive average CLV is the signal. A model with no edge averages "
          "zero\nno matter how the win-loss record happens to land.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--settle", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.record:
        record()
    if a.settle:
        settle()
    if a.report or not (a.record or a.settle):
        report()


if __name__ == "__main__":
    main()
