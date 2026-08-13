"""Build the static web app: inject current-season predictions into the HTML
template. Rerun weekly after features.py + train.py, then republish.

    python export_web.py   ->  web/index.html
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES, LAST_SEASON, POS_GROUPS
from train import ELO_BLEND

TEAMS = {
    "BUF": ("Buffalo Bills", "AFC East"), "MIA": ("Miami Dolphins", "AFC East"),
    "NE": ("New England Patriots", "AFC East"), "NYJ": ("New York Jets", "AFC East"),
    "BAL": ("Baltimore Ravens", "AFC North"), "CIN": ("Cincinnati Bengals", "AFC North"),
    "CLE": ("Cleveland Browns", "AFC North"), "PIT": ("Pittsburgh Steelers", "AFC North"),
    "HOU": ("Houston Texans", "AFC South"), "IND": ("Indianapolis Colts", "AFC South"),
    "JAX": ("Jacksonville Jaguars", "AFC South"), "TEN": ("Tennessee Titans", "AFC South"),
    "DEN": ("Denver Broncos", "AFC West"), "KC": ("Kansas City Chiefs", "AFC West"),
    "LV": ("Las Vegas Raiders", "AFC West"), "LAC": ("Los Angeles Chargers", "AFC West"),
    "DAL": ("Dallas Cowboys", "NFC East"), "NYG": ("New York Giants", "NFC East"),
    "PHI": ("Philadelphia Eagles", "NFC East"), "WAS": ("Washington Commanders", "NFC East"),
    "CHI": ("Chicago Bears", "NFC North"), "DET": ("Detroit Lions", "NFC North"),
    "GB": ("Green Bay Packers", "NFC North"), "MIN": ("Minnesota Vikings", "NFC North"),
    "ATL": ("Atlanta Falcons", "NFC South"), "CAR": ("Carolina Panthers", "NFC South"),
    "NO": ("New Orleans Saints", "NFC South"), "TB": ("Tampa Bay Buccaneers", "NFC South"),
    "ARI": ("Arizona Cardinals", "NFC West"), "LA": ("Los Angeles Rams", "NFC West"),
    "SF": ("San Francisco 49ers", "NFC West"), "SEA": ("Seattle Seahawks", "NFC West"),
}


def _f(x, nd=2):
    """Round for the payload; NaN → None so the page can say "no data yet"
    instead of rendering the string "nan"."""
    if x is None or pd.isna(x):
        return None
    v = round(float(x), nd)
    return int(v) if nd == 0 else v


def main() -> None:
    df = pd.read_parquet("features.parquet")
    season = df[(df.season == LAST_SEASON) & (df.game_type == "REG")].copy()

    model = XGBClassifier()
    model.load_model("model.json")
    season["xgb_raw"] = model.predict_proba(season[FEATURES])[:, 1]
    season["home_prob"] = ((1 - ELO_BLEND) * season["xgb_raw"]
                           + ELO_BLEND * season["elo_prob"])

    games = []
    for r in season.sort_values(["week", "gameday"]).itertuples(index=False):
        # Everything the "why this pick" panel cites, keyed H/A. Kept raw —
        # the page does the phrasing, so wording can follow the pick.
        factors = {
            "eloH": _f(r.elo_home, 0), "eloA": _f(r.elo_away, 0),
            "eloProb": _f(r.elo_prob, 3), "xgb": _f(r.xgb_raw, 3),
            "restH": _f(r.home_rest, 0), "restA": _f(r.away_rest, 0),
            "div": bool(r.div_game), "dome": bool(r.is_dome),
            "temp": _f(r.temp, 0), "wind": _f(r.wind, 0),
            "outH": _f(r.home_n_out, 0), "outA": _f(r.away_n_out, 0),
            "questH": _f(r.home_n_quest, 0), "questA": _f(r.away_n_quest, 0),
            "qbH": _f(r.home_qb_changed, 0), "qbA": _f(r.away_qb_changed, 0),
            "posH": {grp: w for grp in POS_GROUPS
                     if (w := _f(getattr(r, f"home_{grp}_out_wt"))) },
            "posA": {grp: w for grp in POS_GROUPS
                     if (w := _f(getattr(r, f"away_{grp}_out_wt"))) },
            "pdiffH": _f(r.home_pdiff8, 1), "pdiffA": _f(r.away_pdiff8, 1),
            "wrH": _f(r.home_winrate8), "wrA": _f(r.away_winrate8),
            "offH": _f(r.home_off_epa8, 1), "offA": _f(r.away_off_epa8, 1),
            "defH": _f(r.home_def_epa8, 1), "defA": _f(r.away_def_epa8, 1),
        }
        games.append({
            "week": int(r.week),
            "date": pd.Timestamp(r.gameday).strftime("%Y-%m-%d"),
            "away": r.away_team,
            "home": r.home_team,
            "homeProb": round(float(r.home_prob), 3),
            "awayScore": None if pd.isna(r.away_score) else int(r.away_score),
            "homeScore": None if pd.isna(r.home_score) else int(r.home_score),
            "f": factors,
        })

    data = {
        "season": int(LAST_SEASON),
        "generated": date.today().isoformat(),
        "teams": {abbr: {"name": name, "division": div}
                  for abbr, (name, div) in TEAMS.items()},
        "games": games,
    }

    template = Path("web/template.html").read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
    Path("web/index.html").write_text(html)
    print(f"Wrote web/index.html ({len(games)} games, generated {data['generated']})")


if __name__ == "__main__":
    main()
