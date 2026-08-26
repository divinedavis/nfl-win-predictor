"""Kalshi NFL prices -> kalshi_consensus.csv + kalshi_history.csv.

Kalshi is a CFTC-regulated exchange, not a sportsbook. That makes it a
genuinely independent read on the same games: different participants, orders
matched against each other rather than against a house, and no vig baked into
the price -- the exchange charges a fee instead of a spread, so the midpoint is
already a probability and needs no de-juicing.

The API is public and needs no key or authentication, which also makes it a
free fallback for the odds history when Odds API credits run out.

FIELD-NAME TRAP: Kalshi moved its numbers to decimal-string fields. The old
integer-cent keys (`yes_bid`, `last_price`, `volume`) still appear in every
response and are always None -- reading those yields an empty dataset with no
error of any kind. The live values are `yes_bid_dollars`, `yes_ask_dollars`,
`last_price_dollars`, and `orderbook_fp`/`count_fp` on the depth endpoints.

Markets covered: KXNFLGAME (moneyline), KXNFLSPREAD, KXNFLTOTAL, and the player
prop series, which open in game week and are simply absent until then.

    python fetch_kalshi.py            # snapshot every open NFL market
    python fetch_kalshi.py --props    # include prop series (slower)
"""

import argparse
import csv
import os
import sys
import time
from datetime import datetime, timezone

import requests

BASE = "https://api.elections.kalshi.com/trade-api/v2"
CONSENSUS_CSV = "kalshi_consensus.csv"
HISTORY_CSV = "kalshi_history.csv"

GAME_SERIES = ["KXNFLGAME", "KXNFLSPREAD", "KXNFLTOTAL"]
PROP_SERIES = ["KXNFLPASSYDS", "KXNFLRECYDS", "KXNFLRUSHYDS", "KXNFLANYTD",
               "KXNFLPASSTDS", "KXNFLSACK", "KXNFLRECEPTIONS"]

DELAY = 0.15
TIMEOUT = 25
# Kalshi's ticker abbreviations against nflverse's.
KALSHI_TO_NFLVERSE = {"LAR": "LA", "WSH": "WAS"}
MAX_SPREAD_CENTS = 15   # wider than this is not a price, it is an empty book


def canon(t: str) -> str:
    return KALSHI_TO_NFLVERSE.get(t.upper(), t.upper())


def num(rec: dict, key: str):
    """Read a Kalshi decimal-string field. See the field-name trap above."""
    v = rec.get(key)
    if v in (None, ""):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def markets(series: str, status: str = "open") -> list:
    out, cursor = [], ""
    while True:
        p = {"series_ticker": series, "limit": 1000, "status": status}
        if cursor:
            p["cursor"] = cursor
        try:
            r = requests.get(f"{BASE}/markets", params=p, timeout=TIMEOUT)
            if r.status_code != 200:
                print(f"  {series}: HTTP {r.status_code}", file=sys.stderr)
                return out
            d = r.json()
        except requests.RequestException as e:
            print(f"  {series}: {e}", file=sys.stderr)
            return out
        out += d.get("markets", [])
        cursor = d.get("cursor", "")
        if not cursor or not d.get("markets"):
            return out
        time.sleep(DELAY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-props", action="store_true",
                    help="skip the player prop series")
    args = ap.parse_args()

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # Props are pulled by default. They cost almost nothing when closed --
    # outside game week every prop series returns zero markets -- and being
    # absent is exactly why they were easy to forget about.
    series = GAME_SERIES + ([] if args.no_props else PROP_SERIES)
    rows, priced = [], 0

    for s in series:
        ms = markets(s)
        if not ms:
            print(f"  {s}: no open markets")
            continue
        for m in ms:
            bid, ask = num(m, "yes_bid_dollars"), num(m, "yes_ask_dollars")
            last = num(m, "last_price_dollars")
            mid = (bid + ask) / 2 if bid is not None and ask is not None else None
            width = (ask - bid) * 100 if mid is not None else None
            # A yawning book is a market nobody is making, not a cheap edge.
            usable = mid if (width is not None and width <= MAX_SPREAD_CENTS) else None
            if usable is not None or last is not None:
                priced += 1
            rows.append({
                "fetched_at": stamp, "series": s, "ticker": m.get("ticker"),
                "event_ticker": m.get("event_ticker"), "title": m.get("title", ""),
                # Prop markets are thresholds, and yes_sub_title carries the
                # player and the number in a parseable form ("Will Levis: 75+")
                # where the ticker mangles the name (TENWLEVIS8).
                "sub_title": m.get("yes_sub_title", ""),
                "close_time": m.get("close_time", ""),
                "yes_bid": bid, "yes_ask": ask, "yes_mid": mid,
                "spread_cents": round(width, 1) if width is not None else "",
                "usable_mid": usable, "last_price": last,
                "open_interest": num(m, "open_interest_dollars") or m.get("open_interest") or "",
            })
        print(f"  {s}: {len(ms)} markets")
        time.sleep(DELAY)

    if not rows:
        print("nothing fetched")
        return

    fields = ["fetched_at", "series", "ticker", "event_ticker", "title",
              "sub_title", "close_time", "yes_bid", "yes_ask", "yes_mid", "spread_cents",
              "usable_mid", "last_price", "open_interest"]
    with open(CONSENSUS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    fresh = not os.path.exists(HISTORY_CSV)
    with open(HISTORY_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if fresh:
            w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} markets ({priced} priced) -> {CONSENSUS_CSV}, "
          f"appended to {HISTORY_CSV}")


if __name__ == "__main__":
    main()
