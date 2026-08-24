"""Preseason Vegas win totals per team-season.

Source: SportsOddsHistory (now hosted under covers.com) publishes the preseason
over/under win total, the actual win count and the O/U/P result for every team
back to 1989. We scrape the 2010s and 2020s tables; the current season has no
result yet, so its numbers live in win_totals_current.csv (hand-entered once a
year from a named sportsbook snapshot).

Writes win_totals.csv: season, team, win_total, actual_wins, result.
"""
import csv
import html
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent
PAGES = {
    "2010s": "https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team-2010s/",
    "2020s": "https://www.covers.com/sportsoddshistory/nfl-regular-season-win-total-results-by-team/",
}
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/126 Safari/537.36"

# franchise -> current nflverse abbreviation (features.parquet uses current codes
# for every season, so the moved franchises map to where they play today)
TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Oakland Raiders": "LV",
    "Los Angeles Chargers": "LAC", "San Diego Chargers": "LAC",
    "Los Angeles Rams": "LA", "St. Louis Rams": "LA",
    "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN", "New England Patriots": "NE",
    "New Orleans Saints": "NO", "New York Giants": "NYG", "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS", "Washington Football Team": "WAS", "Washington Redskins": "WAS",
}


def _text(cell):
    return html.unescape(re.sub(r"<[^>]+>", " ", cell)).replace("\xa0", " ").strip()


def scrape(url):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        page = r.read().decode("utf-8", "ignore")
    table = re.search(r'<table class="soh1".*?</table>', page, re.S)
    if not table:
        raise SystemExit(f"no win-total table found at {url}")
    rows = re.findall(r"<tr.*?</tr>", table.group(0), re.S)

    seasons = []
    out = []
    for row in rows:
        cells = [_text(c) for c in re.findall(r"<t[dh].*?</t[dh]>", row, re.S)]
        if not cells:
            continue
        if cells[0] == "Team":                      # header carries the season columns
            for c in cells[1:]:
                m = re.fullmatch(r"[’']?(\d{2})", c)
                seasons.append(2000 + int(m.group(1)) if m else None)
            continue
        team = TEAMS.get(cells[0])
        if team is None:
            continue
        for season, cell in zip(seasons, cells[1:]):
            if season is None:
                continue
            m = re.match(r"([\d.]+)\s+(\d+)\s+([OUP])", cell)
            if not m:                               # "?" for a season not yet played
                continue
            total = float(m.group(1))
            if total > 20:                          # source typo: "115" for 11.5
                total /= 10
                print(f"  fixed {season} {team}: {m.group(1)} -> {total}", file=sys.stderr)
            if not 1 <= total <= 16:
                print(f"  WARN {season} {team}: implausible total {total}", file=sys.stderr)
                continue
            out.append((season, team, total, int(m.group(2)), m.group(3)))
    return out


def main():
    rows = []
    for era, url in PAGES.items():
        got = scrape(url)
        print(f"{era}: {len(got)} team-seasons")
        rows += got

    manual = ROOT / "win_totals_current.csv"
    if manual.exists():
        with manual.open() as f:
            for r in csv.DictReader(f):
                rows.append((int(r["season"]), r["team"], float(r["win_total"]), "", ""))
        print(f"current: {sum(1 for r in rows if r[3] == '')} team-seasons from {manual.name}")

    rows.sort(key=lambda r: (r[0], r[1]))
    seen = set()
    with (ROOT / "win_totals.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["season", "team", "win_total", "actual_wins", "result"])
        for r in rows:
            if (r[0], r[1]) in seen:
                continue
            seen.add((r[0], r[1]))
            w.writerow(r)
    seasons = sorted({r[0] for r in rows})
    print(f"wrote win_totals.csv: {len(seen)} rows, {seasons[0]}-{seasons[-1]}")
    for s in seasons:
        n = sum(1 for r in seen if r[0] == s)
        if n != 32:
            print(f"  WARN {s}: {n} teams", file=sys.stderr)


if __name__ == "__main__":
    main()
