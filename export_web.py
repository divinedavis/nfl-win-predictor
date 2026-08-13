"""Build the static web app: inject current-season predictions into the HTML
template. Rerun weekly after features.py + train.py, then republish.

    python export_web.py   ->  web/index.html
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
from xgboost import XGBClassifier

from features import FEATURES, LAST_SEASON
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


def main() -> None:
    df = pd.read_parquet("features.parquet")
    season = df[(df.season == LAST_SEASON) & (df.game_type == "REG")].copy()

    model = XGBClassifier()
    model.load_model("model.json")
    raw = model.predict_proba(season[FEATURES])[:, 1]
    season["home_prob"] = (1 - ELO_BLEND) * raw + ELO_BLEND * season["elo_prob"]

    games = []
    for r in season.sort_values(["week", "gameday"]).itertuples(index=False):
        games.append({
            "week": int(r.week),
            "date": pd.Timestamp(r.gameday).strftime("%Y-%m-%d"),
            "away": r.away_team,
            "home": r.home_team,
            "homeProb": round(float(r.home_prob), 3),
            "awayScore": None if pd.isna(r.away_score) else int(r.away_score),
            "homeScore": None if pd.isna(r.home_score) else int(r.home_score),
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
