"""Every outside opinion on a game, keyed by nflverse game_id.

Four perspectives are collected, and they are not equally independent:

  model    this project (Elo + XGBoost + the market feature)
  books    median no-vig consensus across the US sportsbooks in the Odds API
           feed -- 10 books, but far fewer independent opinions, since the
           offshore ones largely copy and two of them (betonlineag, lowvig)
           are the same operator with different juice
  kalshi   a CFTC-regulated exchange: different participants, orders matched
           against each other rather than a house, no vig in the price
  fpi      ESPN's Football Power Index -- a different model by different
           people, the only outside source here with real history

Sources are optional by design. A missing file, a game nobody has priced, or a
market too wide to mean anything all resolve to None rather than raising, so a
page or a backtest degrades to whatever is actually available.
"""

import os
import re
from datetime import datetime, timezone

import pandas as pd

BOOKS_CSV = "market_consensus.csv"
KALSHI_CSV = "kalshi_consensus.csv"
FPI_CSV = "espn_fpi.csv"

# A Kalshi game event ticker looks like KXNFLGAME-26SEP21NYGLAR: the series,
# a two-digit year, a month, a day, then the two team codes run together with
# no separator. Team codes are 2-3 characters, so the concatenation cannot be
# split reliably -- the pair is recovered from the per-outcome tickers instead.
EVENT_RE = re.compile(r"^KXNFLGAME-(\d{2})([A-Z]{3})(\d{2})([A-Z]{4,6})$")
MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}
KALSHI_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS", "JAC": "JAX"}


def _canon(t: str) -> str:
    return KALSHI_TO_NFLVERSE.get(t.upper(), t.upper())


def books_probs() -> dict:
    """game_id -> no-vig consensus home win probability."""
    if not os.path.exists(BOOKS_CSV):
        return {}
    try:
        d = pd.read_csv(BOOKS_CSV)
    except Exception:
        return {}
    d = d[d.mkt_prob.notna()]
    return dict(zip(d.game_id, d.mkt_prob.astype(float)))


def fpi_probs() -> dict:
    """game_id -> ESPN FPI home win probability (renormalised two-way)."""
    if not os.path.exists(FPI_CSV):
        return {}
    try:
        d = pd.read_csv(FPI_CSV)
    except Exception:
        return {}
    d = d[d.fpi_home_prob.notna()]
    return dict(zip(d.game_id, d.fpi_home_prob.astype(float)))


def kalshi_probs(schedule: pd.DataFrame) -> dict:
    """game_id -> Kalshi mid-price probability for the home team.

    Kalshi tickers carry a date and a team, never an nflverse game_id, so the
    join is (kickoff date, team pair). Each game event holds one market per
    side; the home side's usable midpoint is the probability, and a market too
    wide to price is dropped rather than guessed at.
    """
    if not os.path.exists(KALSHI_CSV):
        return {}
    try:
        k = pd.read_csv(KALSHI_CSV)
    except Exception:
        return {}
    k = k[k.series == "KXNFLGAME"]
    if k.empty:
        return {}

    # Two indexes, because a market can be priced on one side only. The pair
    # index is preferred; the single-team index catches the case where the
    # other side has no price at all and the team pair therefore cannot be
    # reconstructed -- the event ticker concatenates both codes with no
    # separator, and 2-3 character codes cannot be split back apart reliably.
    sched: dict = {}
    by_team: dict = {}
    for r in schedule.itertuples(index=False):
        day = pd.to_datetime(r.gameday).date()
        for offset in (-1, 0, 1):   # kickoff crosses UTC midnight for night games
            d = day + pd.Timedelta(days=offset)
            sched[(d, frozenset({r.home_team, r.away_team}))] = (r.game_id, r.home_team)
            for t in (r.home_team, r.away_team):
                by_team.setdefault((d, t), []).append((r.game_id, r.home_team))

    out: dict = {}
    for event, grp in k.groupby("event_ticker"):
        m = EVENT_RE.match(str(event))
        if not m:
            continue
        yy, mon, dd, _ = m.groups()
        try:
            day = datetime(2000 + int(yy), MONTHS[mon], int(dd), tzinfo=timezone.utc).date()
        except (KeyError, ValueError):
            continue
        sides = {}
        for r in grp.itertuples(index=False):
            team = _canon(str(r.ticker).rsplit("-", 1)[-1])
            # Prefer a quoted midpoint tight enough to mean something; fall
            # back to the last trade, which is a price somebody actually paid
            # and beats discarding the game entirely. Both sides are kept so
            # they can be averaged below.
            if pd.notna(r.usable_mid):
                sides[team] = float(r.usable_mid)
            elif pd.notna(r.last_price):
                sides[team] = float(r.last_price)
        if not sides:
            continue
        for offset in (0, -1, 1):
            d = day + pd.Timedelta(days=offset)
            hit = sched.get((d, frozenset(sides)))
            if not hit and len(sides) == 1:
                # One priced side: identify the game by that team and date.
                # Ambiguous only if a team played twice in a day, which does
                # not happen, so a single candidate is the match.
                cands = by_team.get((d, next(iter(sides))), [])
                hit = cands[0] if len(cands) == 1 else None
            if not hit:
                continue
            gid, home = hit
            away = next((t for t in sides if t != home), None)
            if home in sides and away in sides:
                # Two independent quotes for the same event. On an exchange
                # they need not sum to 1, so averaging the home price against
                # the away price's complement uses both and cancels a one-
                # sided skew.
                out[gid] = (sides[home] + (1.0 - sides[away])) / 2
            elif home in sides:
                out[gid] = sides[home]
            elif away in sides:
                out[gid] = 1.0 - sides[away]
            break
    return out


# Kalshi prop series -> the stat names props.py projects. Series it lists that
# this model has no projection for (passing TDs, sacks, anytime TD) are simply
# not mapped, so they are collected but never compared against nothing.
KALSHI_PROP_STAT = {
    "KXNFLPASSYDS": "passing_yards",
    "KXNFLRECYDS": "receiving_yards",
    "KXNFLRUSHYDS": "rushing_yards",
    "KXNFLRECEPTIONS": "receptions",
}
# yes_sub_title reads "Will Levis: 75+". The ticker mangles the name into
# TENWLEVIS8, so the subtitle is the only reliable source of both the player
# and the threshold.
SUB_RE = re.compile(r"^(?P<player>.+?):\s*(?P<line>[\d.]+)\+")


def kalshi_props() -> dict:
    """(normalised player, stat) -> [(threshold, P(stat >= threshold)), ...].

    Kalshi prices props as threshold contracts -- "75+ passing yards" is a
    market that settles yes at 75 -- which lines up directly with the quantile
    projections in props.py, where prob_over answers the same question. Each
    player/stat can carry several thresholds; they are returned sorted so a
    caller can pick the one nearest whatever line it cares about.

    Prop series only open in game week. Outside it this returns {} and every
    caller degrades to showing no Kalshi number, which is correct.
    """
    if not os.path.exists(KALSHI_CSV):
        return {}
    try:
        k = pd.read_csv(KALSHI_CSV)
    except Exception:
        return {}
    if "sub_title" not in k.columns:
        return {}
    k = k[k.series.isin(KALSHI_PROP_STAT)]
    out: dict = {}
    for r in k.itertuples(index=False):
        m = SUB_RE.match(str(r.sub_title or ""))
        if not m:
            continue
        price = r.usable_mid if pd.notna(r.usable_mid) else r.last_price
        if pd.isna(price):
            continue
        stat = KALSHI_PROP_STAT[r.series]
        key = (_norm_player(m.group("player")), stat)
        out.setdefault(key, []).append((float(m.group("line")), float(price)))
    return {k2: sorted(v) for k2, v in out.items()}


def _norm_player(name: str) -> str:
    """Match props.py's norm_name so Kalshi and the projections agree on who
    a player is."""
    from features import norm_name
    return norm_name(str(name))


def all_sources(schedule: pd.DataFrame) -> dict:
    """game_id -> {source name: home win probability}. Missing sources absent."""
    books, kalshi, fpi = books_probs(), kalshi_probs(schedule), fpi_probs()
    ids = set(books) | set(kalshi) | set(fpi)
    return {gid: {k: v for k, v in
                  (("books", books.get(gid)), ("kalshi", kalshi.get(gid)),
                   ("fpi", fpi.get(gid))) if v is not None}
            for gid in ids}


if __name__ == "__main__":
    import nflreadpy as nfl

    s = nfl.load_schedules().to_pandas()
    s = s[s.season >= 2026]
    s["gameday"] = pd.to_datetime(s["gameday"])
    a = all_sources(s)
    print(f"{len(a)} games with at least one outside source")
    for gid in sorted(a)[:12]:
        print(f"  {gid:22s} " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(a[gid].items())))
