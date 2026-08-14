"""Build the static web app: inject current-season predictions into the HTML
template. Rerun weekly after features.py + train.py, then republish.

    python export_web.py   ->  web/index.html
"""

import json
from datetime import date
from pathlib import Path

import pandas as pd
import nflreadpy as nfl
from xgboost import XGBClassifier

from features import (DEF_GROUPS, FEATURES, LAST_SEASON, OFF_GROUPS,
                      POS_GROUPS, canon, load_team_epa)
from train import ELO_BLEND

# Favorites missing ~3 starters vs a healthy opponent won 62% (2013-2025)
# instead of the 67% all favorites win — the "upset watch" evidence base.
# 2.0 expected starters (snap-weighted, Questionable discounted, IR included)
# corresponds to ~3 actual bodies missing.
UPSET_OUT_THRESHOLD = 2.0
UPSET_OPP_HEALTHY = 0.8     # opponent must be under this


def injury_cols(side: str) -> list:
    return ([f"{side}_n_out", f"{side}_n_quest", f"{side}_qb_changed"]
            + [f"{side}_{g}_out_wt" for g in POS_GROUPS]
            + [f"{side}_{g}_out_epa" for g in OFF_GROUPS]
            + [f"{side}_{g}_out_val" for g in DEF_GROUPS])

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


STAT_LABELS = {
    "passing_yards": "Passing yards", "rushing_yards": "Rushing yards",
    "receiving_yards": "Receiving yards", "receptions": "Receptions",
}


def props_payload():
    """Player-prop projections + paper-trade record for the dashboard's
    props section; None (section hidden) until props.py has run."""
    path = Path("props_projections.csv")
    if not path.exists():
        return None
    proj = pd.read_csv(path)
    if proj.empty:
        return None
    stats = []
    for stat, label in STAT_LABELS.items():
        d = proj[proj.stat == stat].nlargest(12, "p50")
        players = [{
            "player": r.player, "pos": r.position, "team": r.team,
            "opp": r.opp, "home": bool(r.is_home),
            "p10": r.p10, "p25": r.p25, "p50": r.p50,
            "p75": r.p75, "p90": r.p90,
            "status": getattr(r, "status", "") or "",
            "vsN": int(getattr(r, "vs_opp_n", 0) or 0),
            "vsAvg": _f(getattr(r, "vs_opp_avg", None), 1),
            "vsLog": getattr(r, "vs_opp_log", "") or "",
            "carAvg": _f(getattr(r, "career_avg", None), 1),
        } for r in d.itertuples(index=False)]
        if players:
            stats.append({"key": stat, "label": label, "players": players})
    if not stats:
        return None

    paper = None
    ledger_path = Path("paper_trades.csv")
    if ledger_path.exists():
        led = pd.read_csv(ledger_path)
        done = led[led["won"].isin(["True", "False"])]
        wins = int((done["won"] == "True").sum())
        losses = len(done) - wins
        roi = None
        if len(done):
            payouts = [100 / -o if o < 0 else o / 100 for o in done["odds"]]
            pnl = [p if w == "True" else -1.0
                   for p, w in zip(payouts, done["won"])]
            roi = round(100 * sum(pnl) / len(pnl), 1)
        paper = {"wins": wins, "losses": losses, "open": int(len(led) - len(done)),
                 "roi": roi}

    return {"week": int(proj["week"].iloc[0]),
            "season": int(proj["season"].iloc[0]),
            "stats": stats, "paper": paper}


def main() -> None:
    df = pd.read_parquet("features.parquet")
    season = df[(df.season == LAST_SEASON) & (df.game_type == "REG")].copy()

    model = XGBClassifier()
    model.load_model("model.json")

    def blend(frame):
        raw = model.predict_proba(frame[FEATURES])[:, 1]
        return (1 - ELO_BLEND) * raw + ELO_BLEND * frame["elo_prob"].values

    season["xgb_raw"] = model.predict_proba(season[FEATURES])[:, 1]
    season["home_prob"] = ((1 - ELO_BLEND) * season["xgb_raw"]
                           + ELO_BLEND * season["elo_prob"])

    # ATS: map win prob -> expected home margin (linear fit on history), then
    # P(home covers) from the margin residual distribution. Backtested 55.2%
    # vs closing spreads, above breakeven in 10/11 seasons; tracked live.
    hist = df[df["home_win"].notna()].copy()
    hist_prob = ((1 - ELO_BLEND) * model.predict_proba(hist[FEATURES])[:, 1]
                 + ELO_BLEND * hist["elo_prob"])
    hist_margin = hist["home_score"] - hist["away_score"]
    import numpy as np
    A = np.column_stack([np.ones(len(hist_prob)), hist_prob])
    beta = np.linalg.lstsq(A, hist_margin.values, rcond=None)[0]
    sigma = float(np.std(hist_margin.values - A @ beta))
    season["model_margin"] = beta[0] + beta[1] * season["home_prob"]
    from math import erf, sqrt
    season["cover_prob"] = [
        np.nan if pd.isna(sp) else
        1 - 0.5 * (1 + erf((sp - mm) / (sigma * sqrt(2))))
        for sp, mm in zip(season["spread_line"], season["model_margin"])]

    # Counterfactual: each side at full strength (its injury features zeroed),
    # to show what the absences cost. Elo is injury-blind so it stays fixed.
    healthy_home = season.copy(); healthy_home[injury_cols("home")] = 0.0
    healthy_away = season.copy(); healthy_away[injury_cols("away")] = 0.0
    season["prob_home_healthy"] = blend(healthy_home)
    season["prob_away_healthy"] = blend(healthy_away)
    for side in ("home", "away"):
        season[f"{side}_starters_out"] = sum(
            season[f"{side}_{g}_out_wt"].fillna(0) for g in POS_GROUPS
        ) + season[f"{side}_ir_wt"].fillna(0)

    # Live spread juice per side (droplet fetch_spread_odds.py); games not in
    # the snapshot fall back to standard -110 in the page.
    spread_prices: dict = {}
    odds_path = Path("spread_odds.csv")
    if odds_path.exists():
        for r in pd.read_csv(odds_path).itertuples(index=False):
            spread_prices.setdefault((r.home, r.away), {})[r.side] = int(r.price)

    # Key players per team for the expanded game panel: QB + top receivers +
    # top rusher, with active status and history vs this week's opponent.
    UNIT = {"passing_yards": "pass yds", "rushing_yards": "rush yds",
            "receiving_yards": "rec yds"}

    # Latest DK player-prop lines (fetch_props.py snapshots); empty until
    # books post them in game week.
    from features import norm_name
    from props import prob_over
    prop_lines: dict = {}
    snaps = sorted(Path("props_lines").glob("*.csv"))
    if snaps:
        for lr in pd.read_csv(snaps[-1]).itertuples(index=False):
            if lr.side == "Over" and pd.notna(lr.line):
                prop_lines[(norm_name(lr.player), lr.stat)] = float(lr.line)

    team_players: dict = {}
    proj_path = Path("props_projections.csv")
    if proj_path.exists():
        proj = pd.read_csv(proj_path)
        keep = []
        for team, grp in proj.groupby("team"):
            qb = grp[grp.stat == "passing_yards"].nlargest(1, "p50")
            rec = grp[grp.stat == "receiving_yards"].nlargest(2, "p50")
            ru = grp[(grp.stat == "rushing_yards") & (grp.position == "RB")
                     ].nlargest(1, "p50")
            keep.append(pd.concat([qb, rec, ru]))
        for r in pd.concat(keep).itertuples(index=False):
            entry = {
                "n": r.player, "pos": r.position, "u": UNIT.get(r.stat, ""),
                "v": _f(r.p50, 0), "st": (r.status if isinstance(r.status, str)
                                          else "") or "",
                "vsN": int(r.vs_opp_n or 0), "vsA": _f(r.vs_opp_avg, 1),
                "car": _f(r.career_avg, 1),
            }
            vl = prop_lines.get((norm_name(r.player), r.stat))
            if vl is not None:
                p_over = prob_over([r.p10, r.p25, r.p50, r.p75, r.p90], vl)
                entry["vl"] = vl
                entry["lean"] = "Over" if p_over >= 0.5 else "Under"
                entry["lp"] = _f(max(p_over, 1 - p_over), 2)
            team_players.setdefault((r.team, r.opp), []).append(entry)

    # ---- situational context (home/road records, coaches, offense style) ----
    done_all = df[df["home_win"].notna()].sort_values(["gameday", "game_id"])
    home_hist: dict = {}
    road_hist: dict = {}
    won_map: dict = {}
    for r in done_all.itertuples(index=False):
        home_hist.setdefault(r.home_team, []).append(int(r.home_win))
        road_hist.setdefault(r.away_team, []).append(1 - int(r.home_win))
        won_map[(r.season, r.week, r.home_team)] = int(r.home_win)
        won_map[(r.season, r.week, r.away_team)] = 1 - int(r.home_win)

    def rec16(hist: dict, team: str) -> str:
        g = hist.get(team, [])[-16:]
        return f"{sum(g)}-{len(g) - sum(g)}" if g else ""

    sched_all = nfl.load_schedules(list(range(1999, LAST_SEASON + 1))).to_pandas()
    coach_wins: dict = {}
    game_coaches: dict = {}
    for r in sched_all.itertuples(index=False):
        if pd.notna(r.home_coach) and pd.notna(r.away_coach):
            game_coaches[r.game_id] = (r.home_coach, r.away_coach)
            if pd.notna(r.home_score) and r.home_score != r.away_score:
                w = r.home_coach if r.home_score > r.away_score else r.away_coach
                l = r.away_coach if r.home_score > r.away_score else r.home_coach
                coach_wins[(w, l)] = coach_wins.get((w, l), 0) + 1

    # Offense style: rush share of plays, last 8 team-games; 30+ carry games.
    ts = load_team_epa()
    ts = ts[ts["attempts"].notna() & ts["carries"].notna()]
    ts = ts.sort_values(["season", "week"])
    style: dict = {}
    grind: dict = {}
    for team, grp in ts.groupby("team"):
        t8 = grp.tail(8)
        share = float(t8["carries"].sum()
                      / max(t8["carries"].sum() + t8["attempts"].sum(), 1))
        style[team] = share
        t16 = grp.tail(16)
        hits = [(int(g.carries >= 30),
                 won_map.get((g.season, g.week, team), 0))
                for g in t16.itertuples(index=False)]
        n30 = sum(h for h, _ in hits)
        w30 = sum(w for h, w in hits if h)
        grind[team] = (n30, w30)
    style_rank = {t: i + 1 for i, t in enumerate(
        sorted(style, key=style.get, reverse=True))}  # 1 = most run-heavy
    recent = ts[ts["season"] >= 2015]
    league30 = float(np.mean([won_map.get((g.season, g.week, g.team), 0)
                              for g in recent[recent["carries"] >= 30]
                              .itertuples(index=False)]))

    # Current league ranks (1 = fewest allowed) from each team's most recent
    # rolling value — their next game row carries the as-of-now number.
    rank_of: dict = {}
    ordered = season.sort_values(["week", "gameday"])
    for metric, hcol, acol, desc in [
            ("pa", "home_pa8", "away_pa8", False),          # fewer = better D
            ("rushAll", "home_rush_all8", "away_rush_all8", False),
            ("passAll", "home_pass_all8", "away_pass_all8", False),
            ("pf", "home_pf8", "away_pf8", True),           # more = better O
            ("rushOff", "home_rush_off8", "away_rush_off8", True),
            ("passOff", "home_pass_off8", "away_pass_off8", True)]:
        vals: dict = {}
        for r in ordered.itertuples(index=False):
            for team, col in ((r.home_team, hcol), (r.away_team, acol)):
                v = getattr(r, col)
                if team not in vals and pd.notna(v):
                    vals[team] = v
        ranked = sorted(vals, key=vals.get, reverse=desc)
        rank_of[metric] = {t: i + 1 for i, t in enumerate(ranked)}

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
            "qbValH": _f(r.home_qb_val, 1), "qbValA": _f(r.away_qb_val, 1),
            "qbNameH": r.home_qb_pred_name or "", "qbNameA": r.away_qb_pred_name or "",
            "keyOutsH": r.home_key_outs or "", "keyOutsA": r.away_key_outs or "",
            "posH": {grp: w for grp in POS_GROUPS
                     if (w := _f(getattr(r, f"home_{grp}_out_wt"))) },
            "posA": {grp: w for grp in POS_GROUPS
                     if (w := _f(getattr(r, f"away_{grp}_out_wt"))) },
            # Injury cost in win-probability points, per team, vs full strength
            "impH": _f((r.prob_home_healthy - r.home_prob) * 100, 0),
            "impA": _f((r.home_prob - r.prob_away_healthy) * 100, 0),
            "outStH": _f(r.home_starters_out, 1), "outStA": _f(r.away_starters_out, 1),
            "pdiffH": _f(r.home_pdiff8, 1), "pdiffA": _f(r.away_pdiff8, 1),
            "wrH": _f(r.home_winrate8), "wrA": _f(r.away_winrate8),
            "offH": _f(r.home_off_epa8, 1), "offA": _f(r.away_off_epa8, 1),
            "defH": _f(r.home_def_epa8, 1), "defA": _f(r.away_def_epa8, 1),
            "paH": _f(r.home_pa8, 1), "paA": _f(r.away_pa8, 1),
            "passAllH": _f(r.home_pass_all8, 0), "passAllA": _f(r.away_pass_all8, 0),
            "rushAllH": _f(r.home_rush_all8, 0), "rushAllA": _f(r.away_rush_all8, 0),
            "hRec": rec16(home_hist, r.home_team),
            "aRec": rec16(road_hist, r.away_team),
            "coachH": game_coaches.get(r.game_id, ("", ""))[0],
            "coachA": game_coaches.get(r.game_id, ("", ""))[1],
            "h2hHW": coach_wins.get((game_coaches.get(r.game_id, ("", ""))[0],
                                     game_coaches.get(r.game_id, ("", ""))[1]), 0),
            "h2hAW": coach_wins.get((game_coaches.get(r.game_id, ("", ""))[1],
                                     game_coaches.get(r.game_id, ("", ""))[0]), 0),
            "styShH": _f(style.get(r.home_team), 2),
            "styShA": _f(style.get(r.away_team), 2),
            "styRkH": style_rank.get(r.home_team),
            "styRkA": style_rank.get(r.away_team),
            "g30nH": grind.get(r.home_team, (0, 0))[0],
            "g30wH": grind.get(r.home_team, (0, 0))[1],
            "g30nA": grind.get(r.away_team, (0, 0))[0],
            "g30wA": grind.get(r.away_team, (0, 0))[1],
            "paRkH": rank_of["pa"].get(r.home_team),
            "paRkA": rank_of["pa"].get(r.away_team),
            "rushRkH": rank_of["rushAll"].get(r.home_team),
            "rushRkA": rank_of["rushAll"].get(r.away_team),
            "passRkH": rank_of["passAll"].get(r.home_team),
            "passRkA": rank_of["passAll"].get(r.away_team),
            "pfH": _f(r.home_pf8, 1), "pfA": _f(r.away_pf8, 1),
            "pfRkH": rank_of["pf"].get(r.home_team),
            "pfRkA": rank_of["pf"].get(r.away_team),
            "rushOffH": _f(r.home_rush_off8, 0), "rushOffA": _f(r.away_rush_off8, 0),
            "rushOffRkH": rank_of["rushOff"].get(r.home_team),
            "rushOffRkA": rank_of["rushOff"].get(r.away_team),
            "passOffH": _f(r.home_pass_off8, 0), "passOffA": _f(r.away_pass_off8, 0),
            "passOffRkH": rank_of["passOff"].get(r.home_team),
            "passOffRkA": rank_of["passOff"].get(r.away_team),
        }
        factors["playersH"] = team_players.get((r.home_team, r.away_team), [])
        factors["playersA"] = team_players.get((r.away_team, r.home_team), [])
        fav_is_home = r.home_prob >= 0.5
        fav_out = r.home_starters_out if fav_is_home else r.away_starters_out
        opp_out = r.away_starters_out if fav_is_home else r.home_starters_out
        factors["upset"] = bool(fav_out >= UPSET_OUT_THRESHOLD
                                and opp_out <= UPSET_OPP_HEALTHY)
        games.append({
            "week": int(r.week),
            "date": pd.Timestamp(r.gameday).strftime("%Y-%m-%d"),
            "away": r.away_team,
            "home": r.home_team,
            "homeProb": round(float(r.home_prob), 3),
            "vegasLine": _f(r.spread_line, 1),
            "modelMargin": _f(r.model_margin, 1),
            "coverProb": _f(r.cover_prob, 3),
            "spH": spread_prices.get((r.home_team, r.away_team), {}).get("home"),
            "spA": spread_prices.get((r.home_team, r.away_team), {}).get("away"),
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
        "props": props_payload(),
        "run30": round(league30, 2),
    }

    template = Path("web/template.html").read_text()
    html = template.replace("__DATA_JSON__", json.dumps(data, separators=(",", ":")))
    Path("web/index.html").write_text(html)
    print(f"Wrote web/index.html ({len(games)} games, generated {data['generated']})")


if __name__ == "__main__":
    main()
