"""Joins for the outside feeds.

All of this is dormant most of the time -- Kalshi's prop series do not exist
outside game week, and a game only appears once Kalshi opens it -- so these
tests stand in for the live data that will not be there to catch a regression
until the season starts.
"""

import pandas as pd
import pytest

import sources


def test_subtitle_parses_player_and_threshold():
    m = sources.SUB_RE.match("Will Levis: 75+")
    assert m.group("player") == "Will Levis"
    assert m.group("line") == "75"


def test_subtitle_handles_a_decimal_threshold():
    m = sources.SUB_RE.match("Puka Nacua: 5.5+")
    assert m.group("player") == "Puka Nacua"
    assert m.group("line") == "5.5"


def test_subtitle_tolerates_a_hyphenated_or_suffixed_name():
    m = sources.SUB_RE.match("Amon-Ra St. Brown: 90+")
    assert m.group("player") == "Amon-Ra St. Brown"
    assert m.group("line") == "90"


def test_subtitle_rejects_a_game_market_title():
    assert sources.SUB_RE.match("New York G wins") is None


@pytest.mark.parametrize("kalshi,expected", [
    ("LAR", "LA"),    # nflverse calls the Rams LA
    ("WSH", "WAS"),
    ("JAC", "JAX"),   # this one silently dropped every Jacksonville game
    ("KC", "KC"),
])
def test_team_aliases_match_nflverse(kalshi, expected):
    assert sources._canon(kalshi) == expected


def test_event_ticker_regex_reads_the_date():
    m = sources.EVENT_RE.match("KXNFLGAME-26SEP21NYGLAR")
    assert m.groups()[:3] == ("26", "SEP", "21")


def _schedule(game_id, home, away, day):
    return pd.DataFrame([{"game_id": game_id, "home_team": home,
                          "away_team": away, "gameday": pd.Timestamp(day)}])


def _market(ticker, event, mid, last=None, wide=False):
    return {"series": "KXNFLGAME", "ticker": ticker, "event_ticker": event,
            "usable_mid": None if wide else mid, "last_price": last,
            "sub_title": "", "title": ""}


def test_two_sided_quote_averages_both_books(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame([
        _market("KXNFLGAME-26SEP21NYGLAR-LAR", "KXNFLGAME-26SEP21NYGLAR", 0.80),
        _market("KXNFLGAME-26SEP21NYGLAR-NYG", "KXNFLGAME-26SEP21NYGLAR", 0.24),
    ]).to_csv(sources.KALSHI_CSV, index=False)
    got = sources.kalshi_probs(_schedule("2026_03_NYG_LA", "LA", "NYG", "2026-09-21"))
    # home 0.80 against away's complement 0.76 -> 0.78
    assert got["2026_03_NYG_LA"] == pytest.approx(0.78)


def test_a_wide_book_falls_back_to_the_last_trade(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame([
        _market("KXNFLGAME-26SEP21NYGLAR-LAR", "KXNFLGAME-26SEP21NYGLAR",
                None, last=0.70, wide=True),
        _market("KXNFLGAME-26SEP21NYGLAR-NYG", "KXNFLGAME-26SEP21NYGLAR",
                None, last=0.30, wide=True),
    ]).to_csv(sources.KALSHI_CSV, index=False)
    got = sources.kalshi_probs(_schedule("2026_03_NYG_LA", "LA", "NYG", "2026-09-21"))
    assert got["2026_03_NYG_LA"] == pytest.approx(0.70)


def test_one_priced_side_still_yields_a_probability(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pd.DataFrame([
        _market("KXNFLGAME-26SEP21NYGLAR-NYG", "KXNFLGAME-26SEP21NYGLAR", 0.25),
    ]).to_csv(sources.KALSHI_CSV, index=False)
    got = sources.kalshi_probs(_schedule("2026_03_NYG_LA", "LA", "NYG", "2026-09-21"))
    assert got["2026_03_NYG_LA"] == pytest.approx(0.75)   # home = 1 - away


def test_a_night_game_still_matches_across_the_utc_date(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Kalshi stamps the UTC date, which for a Sunday night game is the Monday.
    pd.DataFrame([
        _market("KXNFLGAME-26SEP22NYGLAR-LAR", "KXNFLGAME-26SEP22NYGLAR", 0.80),
        _market("KXNFLGAME-26SEP22NYGLAR-NYG", "KXNFLGAME-26SEP22NYGLAR", 0.20),
    ]).to_csv(sources.KALSHI_CSV, index=False)
    got = sources.kalshi_probs(_schedule("2026_03_NYG_LA", "LA", "NYG", "2026-09-21"))
    assert "2026_03_NYG_LA" in got


def test_missing_files_return_empty_rather_than_raising(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert sources.kalshi_props() == {}
    assert sources.books_probs() == {}
    assert sources.fpi_probs() == {}
    assert sources.kalshi_probs(_schedule("x", "LA", "NYG", "2026-09-21")) == {}
