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
KALSHI_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}


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

    # (date, frozenset(teams)) -> nflverse game_id
    sched: dict = {}
    for r in schedule.itertuples(index=False):
        day = pd.to_datetime(r.gameday).date()
        for offset in (-1, 0, 1):   # kickoff crosses UTC midnight for night games
            d = day + pd.Timedelta(days=offset)
            sched[(d, frozenset({r.home_team, r.away_team}))] = (r.game_id, r.home_team)

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
            if pd.notna(r.usable_mid):
                sides[team] = float(r.usable_mid)
        if len(sides) < 2:
            continue
        for offset in (0, -1, 1):
            hit = sched.get((day + pd.Timedelta(days=offset), frozenset(sides)))
            if hit:
                gid, home = hit
                if home in sides:
                    out[gid] = sides[home]
                break
    return out


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
