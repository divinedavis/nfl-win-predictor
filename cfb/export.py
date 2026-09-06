"""Build the college page: web/college/index.html from web/template.html.

Same template, same payload shape as export_web.py, so every view the NFL
page has -- week list, team schedules, the why panel, what everyone thinks,
key players, the props tab -- renders for FBS without a second front end.
League-specific copy is driven by DATA.league and DATA.meta.

    python -m cfb.export
"""

import json
import math
import re
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from cfb.data import LAST_SEASON, Teams, load_injuries
from cfb.features import FEATURES
from cfb.train import META, MODEL, load_blend
from export_web import _f, _last_game
from props import prob_over

POWER = ["SEC", "Big Ten", "Big 12", "ACC"]
STAR_AT = 0.85       # confidence tier the page stars; 94% correct in backtest
UNIT = {"passing_yards": "pass yds", "rushing_yards": "rush yds",
        "receiving_yards": "rec yds"}
STAT_LABELS = {"passing_yards": "Passing yards", "rushing_yards": "Rushing yards",
               "receiving_yards": "Receiving yards", "receptions": "Receptions"}


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def props_payload(teams: Teams) -> dict | None:
    path = Path("cfb_props_projections.csv")
    if not path.exists():
        return None
    proj = pd.read_csv(path)
    if proj.empty:
        return None
    stats = []
    for stat, label in STAT_LABELS.items():
        d = proj[proj.stat == stat].nlargest(40, "p50")
        players = [{
            "pid": int(r.player_id), "player": r.player, "pos": r.position,
            "team": r.team, "opp": r.opp, "home": bool(r.is_home),
            "p10": r.p10, "p25": r.p25, "p50": r.p50, "p75": r.p75, "p90": r.p90,
            "status": r.status if isinstance(r.status, str) else "",
            "vsN": int(r.vs_opp_n or 0), "vsAvg": _f(r.vs_opp_avg, 1),
            "vsMed": _f(r.vs_opp_med, 1), "vsLog": r.vs_opp_log if isinstance(r.vs_opp_log, str) else "",
            "carAvg": _f(r.career_avg, 1), **_last_game(r), "week": int(r.week),
        } for r in d.itertuples(index=False)]
        if players:
            stats.append({"key": stat, "label": label, "players": players})
    if not stats:
        return None
    return {"week": int(proj.week.min()), "season": int(proj.season.iloc[0]),
            "results": {}, "stats": stats, "paper": None}


def key_players(teams: Teams) -> dict:
    """(team, opp) -> QB + top two receivers + top rusher for the game."""
    path = Path("cfb_props_projections.csv")
    out: dict = {}
    if not path.exists():
        return out
    proj = pd.read_csv(path)
    keep = []
    for team, grp in proj.groupby("team"):
        qb = grp[grp.stat == "passing_yards"].nlargest(1, "p50")
        rec = grp[grp.stat == "receiving_yards"].nlargest(2, "p50")
        ru = grp[(grp.stat == "rushing_yards") & (grp.position == "RB")].nlargest(1, "p50")
        keep.append(pd.concat([qb, rec, ru]))
    if not keep:
        return out
    for r in pd.concat(keep).itertuples(index=False):
        entry = {"n": r.player, "pos": r.position, "u": UNIT.get(r.stat, ""),
                 "v": _f(r.p50, 0), "st": r.status if isinstance(r.status, str) else "",
                 "vsN": int(r.vs_opp_n or 0), "vsA": _f(r.vs_opp_avg, 1),
                 "vsM": _f(r.vs_opp_med, 1), "car": _f(r.career_avg, 1),
                 **_last_game(r)}
        out.setdefault((r.team, r.opp), []).append(entry)
    return out


def main() -> None:
    teams = Teams()
    df = pd.read_parquet("cfb_features.parquet")
    meta = json.load(open(META))
    blend = load_blend()
    model = XGBClassifier()
    model.load_model(MODEL)

    season = df[df.season == LAST_SEASON].copy()
    season["xgb_raw"] = model.predict_proba(season[FEATURES])[:, 1]
    season["home_prob"] = (1 - blend) * season["xgb_raw"] + blend * season["elo_prob"]

    # Win prob -> expected margin, fit on history, for the ATS comparison.
    hist = df[df["home_win"].notna() & (df.season >= 2015)].copy()
    hp = (1 - blend) * model.predict_proba(hist[FEATURES])[:, 1] + blend * hist["elo_prob"].values
    # Margins scale with confidence non-linearly in college (a 95% favorite
    # wins by 35, not 20), so fit on the logit of the probability.
    logit = lambda p: np.log(np.clip(p, 1e-3, 1 - 1e-3) / (1 - np.clip(p, 1e-3, 1 - 1e-3)))
    A = np.column_stack([np.ones(len(hp)), logit(hp)])
    margin = (hist["home_score"] - hist["away_score"]).values
    beta = np.linalg.lstsq(A, margin, rcond=None)[0]
    sigma = float(np.std(margin - A @ beta))
    season["model_margin"] = beta[0] + beta[1] * logit(season["home_prob"].values)
    season["cover_prob"] = [
        norm_cdf((m - s) / sigma) if pd.notna(s) else np.nan
        for m, s in zip(season["model_margin"], season["spread_line"])]

    # Outside opinions: books from the ESPN spread, plus the fetched FPI and
    # Kalshi numbers. All as home win probability.
    src_rows = {}
    sp = Path("cfb_sources.csv")
    if sp.exists():
        for r in pd.read_csv(sp).itertuples(index=False):
            d = {}
            if pd.notna(r.fpi_home_prob):
                d["fpi"] = round(float(r.fpi_home_prob), 3)
            if pd.notna(r.kalshi_home_prob):
                d["kalshi"] = round(float(r.kalshi_home_prob), 3)
            src_rows[int(r.game_id)] = d

    # Home / road records, last 16 by team, over every completed game.
    done_all = df[df["home_win"].notna()].sort_values("game_date")
    home_hist: dict = {}
    road_hist: dict = {}
    for r in done_all.itertuples(index=False):
        home_hist.setdefault(r.home_team, []).append(int(r.home_win))
        road_hist.setdefault(r.away_team, []).append(1 - int(r.home_win))

    def rec16(hist, team):
        g = hist.get(team, [])[-16:]
        return f"{sum(g)}-{len(g) - sum(g)}" if g else ""

    # League ranks among FBS teams from each team's current trailing window.
    fbs_abbr = {teams.abbr(t) for t in teams.fbs}
    rank_of: dict = {}
    ordered = season.sort_values("game_date")
    for metric, col, desc in [("pa", "pa8", False), ("rushAll", "rush_all8", False),
                              ("passAll", "pass_all8", False), ("pf", "pf8", True),
                              ("rushOff", "rush_off8", True), ("passOff", "pass_off8", True)]:
        vals: dict = {}
        for r in ordered.itertuples(index=False):
            for team, v in ((r.home_team, getattr(r, f"home_{col}")),
                            (r.away_team, getattr(r, f"away_{col}"))):
                if team in fbs_abbr and team not in vals and pd.notna(v):
                    vals[team] = v
        ranked = sorted(vals, key=vals.get, reverse=desc)
        rank_of[metric] = {t: i + 1 for i, t in enumerate(ranked)}

    # ESPN's injury feed, where it has anything: names by team.
    inj = load_injuries()
    inj_by: dict = {}
    if len(inj):
        for r in inj.itertuples(index=False):
            if str(r.status).lower() in ("out", "doubtful", "injured reserve"):
                inj_by.setdefault(teams.abbr(r.team_id), []).append(
                    f"{r.athlete_display_name} ({r.athlete_position}, {r.status})")

    players = key_players(teams)
    games = []
    used_teams = set()
    for r in season.sort_values(["week", "game_date"]).itertuples(index=False):
        used_teams.update([r.home_team, r.away_team])
        f = {
            "eloH": _f(r.elo_home, 0), "eloA": _f(r.elo_away, 0),
            "eloProb": _f(r.elo_prob, 3), "xgb": _f(r.xgb_raw, 3),
            "restH": _f(r.home_rest, 0), "restA": _f(r.away_rest, 0),
            "confGame": bool(r.conf_game), "neutral": bool(r.neutral),
            "dome": bool(r.is_dome),
            "pdiffH": _f(r.home_pdiff8, 1), "pdiffA": _f(r.away_pdiff8, 1),
            "wrH": _f(r.home_winrate8), "wrA": _f(r.away_winrate8),
            "offH": _f(r.home_off_epa8, 2), "offA": _f(r.away_off_epa8, 2),
            "defH": _f(r.home_def_epa8, 2), "defA": _f(r.away_def_epa8, 2),
            "paH": _f(r.home_pa8, 1), "paA": _f(r.away_pa8, 1),
            "pfH": _f(r.home_pf8, 1), "pfA": _f(r.away_pf8, 1),
            "passAllH": _f(r.home_pass_all8, 0), "passAllA": _f(r.away_pass_all8, 0),
            "rushAllH": _f(r.home_rush_all8, 0), "rushAllA": _f(r.away_rush_all8, 0),
            "rushOffH": _f(r.home_rush_off8, 0), "rushOffA": _f(r.away_rush_off8, 0),
            "passOffH": _f(r.home_pass_off8, 0), "passOffA": _f(r.away_pass_off8, 0),
            "talentH": _f(r.home_talent, 0), "talentA": _f(r.away_talent, 0),
            "hRec": rec16(home_hist, r.home_team), "aRec": rec16(road_hist, r.away_team),
            "injH": inj_by.get(r.home_team, []), "injA": inj_by.get(r.away_team, []),
            "playersH": players.get((r.home_team, r.away_team), []),
            "playersA": players.get((r.away_team, r.home_team), []),
        }
        for m in ("pa", "pf", "rushAll", "passAll", "rushOff", "passOff"):
            f[m + "RkH"] = rank_of[m].get(r.home_team)
            f[m + "RkA"] = rank_of[m].get(r.away_team)
        src = dict(src_rows.get(int(r.game_id), {}))
        if pd.notna(r.mkt_prob):
            src["books"] = round(float(r.mkt_prob), 3)
        games.append({
            "week": int(r.week) if r.game_type == "REG" else 15 + int(r.week),
            "post": r.game_type == "POST",
            "date": str(r.gameday),
            "kick": pd.Timestamp(r.game_date).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "away": r.away_team, "home": r.home_team,
            "homeProb": round(float(r.home_prob), 3),
            "vegasLine": _f(r.spread_line, 1),
            "modelMargin": _f(r.model_margin, 1),
            "coverProb": _f(r.cover_prob, 3),
            "src": src,
            "spH": None, "spA": None,
            "awayScore": None if pd.isna(r.away_score) else int(r.away_score),
            "homeScore": None if pd.isna(r.home_score) else int(r.home_score),
            "f": f,
        })

    tdict = {}
    for tid in list(teams.fbs) + [t for t in teams.rows if teams.abbr(t) in used_teams]:
        ab = teams.abbr(tid)
        if ab in tdict:
            continue
        tdict[ab] = {"name": teams.name(tid), "short": teams.short(tid),
                     "division": teams.conf(tid), "fbs": teams.is_fbs(tid)}

    # Team pickers: Power Four on one side, everyone else on the other, each
    # grouped by conference like the NFL page groups by division.
    confs: dict = {}
    for ab, t in tdict.items():
        if t["fbs"]:
            confs.setdefault(t["division"], []).append(ab)
    p4 = {c: sorted(v) for c, v in confs.items() if c in POWER}
    g5 = {c: sorted(v) for c, v in confs.items() if c not in POWER}
    pickers = [{"id": "p4", "label": "Power Four", "groups": p4},
               {"id": "g5", "label": "Group of Five + independents", "groups": g5}]
    filters = [{"key": "p4", "label": "Power Four games",
                "teams": sorted(sum(p4.values(), []))}]
    for c in sorted(confs):
        filters.append({"key": c, "label": c, "teams": sorted(confs[c])})

    data = {
        "league": "cfb", "leagueName": "College Football (FBS)",
        "season": int(LAST_SEASON), "generated": date.today().isoformat(),
        "teams": tdict, "games": games, "pickers": pickers, "filters": filters,
        "props": props_payload(teams), "qb": None,
        "starAt": STAR_AT,
        "meta": {"acc": round(meta["backtest_acc"] * 100),
                 "brier": round(meta["backtest_brier"], 3),
                 "starAcc": round(meta["tier_acc"][str(STAR_AT)] * 100),
                 "games": meta["games"], "seasons": meta["seasons"],
                 "eloBlend": blend, "sigma": round(sigma, 1)},
    }
    template = Path("web/template.html").read_text()
    blob = (json.dumps(data, separators=(",", ":"))
            .replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026"))
    html = template.replace("__DATA_JSON__", blob)
    # Crawlers read the meta tags, not the script, so the college page gets
    # its own title and description at build time.
    html = re.sub(r"(<title>)[^<]*(</title>)",
                  r"\1Sputter Bets — College Football Win Probabilities\2", html)
    html = html.replace('content="Sputter Bets — NFL Win Probabilities"',
                        'content="Sputter Bets — College Football Win Probabilities"')
    html = re.sub(r'(<meta property="og:description" content=")[^"]*(")',
                  r"\1Weekly FBS picks with calibrated win probabilities, why-this-pick "
                  r"reasoning, player prop projections and against-the-spread leans — "
                  r"from an Elo + XGBoost model trained on 21 seasons.\2", html)
    Path("web/college").mkdir(exist_ok=True)
    Path("web/college/index.html").write_text(html)
    print(f"Wrote web/college/index.html ({len(games)} games, {len(tdict)} teams, "
          f"{len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
