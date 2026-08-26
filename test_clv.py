"""Sign conventions for CLV and ATS grading.

These two functions are the whole metric, and an inverted sign in either would
not raise anything — it would quietly report a losing model as a winning one.
Spreads follow the nflverse convention everywhere: positive = home favoured.
"""

from clv import ats_result, clv_points


def test_home_clv_positive_when_line_moves_toward_home():
    # Took home -3, closed home -4: our number is the better one.
    assert clv_points("home", 3.0, 4.0) == 1.0


def test_home_clv_negative_when_line_moves_away():
    assert clv_points("home", 3.0, 2.0) == -1.0


def test_away_clv_positive_when_line_moves_toward_away():
    # Took away +3, closed away +2: our number is the better one.
    assert clv_points("away", 3.0, 2.0) == 1.0


def test_away_clv_negative_when_line_moves_away():
    assert clv_points("away", 3.0, 4.0) == -1.0


def test_clv_is_zero_on_an_unmoved_line():
    assert clv_points("home", 3.0, 3.0) == 0.0
    assert clv_points("away", 3.0, 3.0) == 0.0


def test_home_favourite_covers_only_by_more_than_the_line():
    assert ats_result("home", margin=7, line=3.0) == "win"
    assert ats_result("home", margin=1, line=3.0) == "loss"


def test_away_dog_covers_when_the_favourite_falls_short():
    assert ats_result("away", margin=1, line=3.0) == "win"
    assert ats_result("away", margin=7, line=3.0) == "loss"


def test_exact_landing_is_a_push_for_either_side():
    assert ats_result("home", margin=3, line=3.0) == "push"
    assert ats_result("away", margin=3, line=3.0) == "push"


def test_home_underdog_covers_by_losing_narrowly():
    # spread_line -2.5 means the home team is a 2.5-point underdog.
    assert ats_result("home", margin=-1, line=-2.5) == "win"
    assert ats_result("home", margin=-6, line=-2.5) == "loss"
    assert ats_result("away", margin=-6, line=-2.5) == "win"
