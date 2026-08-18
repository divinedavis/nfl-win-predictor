"""Grade quarterbacks by quarter and by late-game situation, and backtest how
starting quarterbacks perform in the regular season versus the playoffs.

Everything reads the cached aggregates from build_qb_splits.py, so the grade
formula can be re-cut without re-downloading play-by-play.

The grade: EPA per dropback inside the bucket, pulled toward the league mean by
that bucket's own noise level (empirical Bayes — a fourth-quarter, under-five-
minute sample is a fraction of a full season's, so it moves less), then placed
on a 50-plus-or-minus-20 scale against the passers it is being shown next to.
Each bucket is scaled on its own, so 70 in the fourth quarter and 70 in the
first mean the same thing: about a standard deviation above the field.

League-wide questions — does play actually fall off late — are answered with
raw rates and paired tests, never with the graded scale.

    python qb_grades.py            # print the report
    python qb_grades.py --json out.json
"""

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from build_qb_splits import CSV, DEF_CSV, FIRST_SEASON
from build_qbr import SEASON_CSV, WEEK_CSV

DISPLAY_BUCKETS = [
    ("q1", "1st Q"), ("q2", "2nd Q"), ("q3", "3rd Q"), ("q4", "4th Q"),
    ("late", "Under 5:00"), ("clutch", "Under 5:00, one score"),
]
# order of the per-entity grade array in the web payload; "early" rides along
# only to compute the late-game swing and is dropped before shipping
PAYLOAD_BUCKETS = [k for k, _ in DISPLAY_BUCKETS] + ["all"]
MIN_SEASON_DB = 200     # a starter's regular season (roughly half a year)
MIN_WINDOW_DB = 400     # a starter across the five-season window
MIN_POST_DB = 20        # enough playoff snaps to compare against
MIN_BUCKET_DB = 10      # below this a bucket is left blank rather than graded
MIN_QBR_PLAYS = 250     # ESPN action plays for the 1-32 board (a starter's year)
MIN_QBR_WEEKS = 6       # weeks needed to draw a passer's QBR line
GRADE_SD = 20.0         # grade points per standard deviation
SUM_COLS = ["games", "n", "epa", "epa2", "succ", "cpoe", "cpoe_n", "sacks",
            "ints", "tds", "att", "comp"]


# ---------------------------------------------------------------- loading

def load(path: str = CSV) -> pd.DataFrame:
    return add_derived(pd.read_csv(path))


def add_derived(d: pd.DataFrame) -> pd.DataFrame:
    """Two buckets that are arithmetic on the stored ones: everything before
    the fourth quarter, and one-score plays that are NOT in the last five
    minutes — the control that separates late-game situation from late-game
    performance."""
    keys = [c for c in ("season", "season_type", "qb_id", "qb", "team")
            if c in d.columns]
    sums = [c for c in SUM_COLS if c in d.columns]
    early = (d[d.bucket.isin(["q1", "q2", "q3"])]
             .groupby(keys, dropna=False)[sums].sum().reset_index())
    early["bucket"] = "early"
    close = d[d.bucket == "close"].set_index(keys)[sums]
    clutch = d[d.bucket == "clutch"].set_index(keys)[sums]
    ec = close.subtract(clutch.reindex(close.index).fillna(0)).reset_index()
    ec["bucket"] = "earlyclose"
    ec = ec[ec.n > 0]
    return pd.concat([d, early, ec], ignore_index=True)


def roll(d: pd.DataFrame, keys: list) -> pd.DataFrame:
    """Collapse to one row per key+bucket (a traded passer has a row per team)."""
    sums = [c for c in SUM_COLS if c in d.columns]
    return d.groupby(keys + ["bucket"], dropna=False)[sums].sum().reset_index()


def qualified_index(d: pd.DataFrame, keys: list, min_db: int) -> pd.Index:
    tot = roll(d[d.bucket == "all"], keys)
    return tot[tot.n >= min_db].set_index(keys).index


# ---------------------------------------------------------------- grading

def shrink_k(rows: pd.DataFrame) -> float:
    """Empirical-Bayes shrinkage constant for one bucket, in dropbacks: the
    play-to-play variance divided by the spread of real differences between
    passers. Small buckets and buckets where everyone performs alike shrink
    hardest."""
    n_tot = rows.n.sum()
    mu = rows.epa.sum() / n_tot
    var_play = max((rows.epa2.sum() - n_tot * mu ** 2) / max(n_tot - 1, 1), 1e-6)
    obs = rows.epa / rows.n
    var_true = max(float(((obs - mu) ** 2).mean() - (var_play / rows.n).mean()),
                   var_play / n_tot)
    return float(var_play / var_true)


def bucket_ks(d: pd.DataFrame, keys: list, min_db: int) -> dict:
    """One shrinkage constant per bucket, fit once on the full window so that a
    single season's grades sit on the same footing as the five-year ones."""
    qual = qualified_index(d, keys, min_db)
    rolled = roll(d, keys)
    rolled = rolled[rolled.set_index(keys).index.isin(qual)]
    return {b: shrink_k(g) for b, g in rolled.groupby("bucket")
            if len(g) > 2 and g.n.min() > 0}


def shrunk_rate(rows: pd.DataFrame, k: float, weight_n=None,
                mu: float = None) -> pd.Series:
    """EPA per dropback pulled toward the league mean. `weight_n` overrides the
    sample size doing the pulling — paired comparisons pass the smaller of the
    two samples for both sides, so a 17-game season and a 3-game playoff run are
    held to the same confidence instead of one shrinking far harder. `mu` pins
    the target of the pull, so playoff snaps are measured against the same
    regular-season baseline they are being compared to."""
    mu = rows.epa.sum() / rows.n.sum() if mu is None else mu
    obs = rows.epa / rows.n
    w = rows.n if weight_n is None else weight_n
    return mu + (w / (w + k)) * (obs - mu)


def grade(shrunk: pd.Series, center: float = None, scale: float = None):
    c = shrunk.mean() if center is None else center
    s = shrunk.std(ddof=1) if scale is None else scale
    s = s if s and s > 1e-9 else 1.0
    return np.clip(50 + GRADE_SD * (shrunk - c) / s, 0, 100), float(c), float(s)


def graded_rows(d: pd.DataFrame, keys: list, min_db: int, ks: dict):
    """Grade every qualified entity in every bucket, one scale per bucket."""
    qual = qualified_index(d, keys, min_db)
    rolled = roll(d, keys)
    rolled = rolled[rolled.set_index(keys).index.isin(qual)]
    out = []
    for bucket, g in rolled.groupby("bucket"):
        if bucket not in ks:
            continue
        g = g[g.n >= MIN_BUCKET_DB].copy()
        if len(g) < 3:
            continue
        g["epa_db"] = g.epa / g.n
        g["grade"], _, _ = grade(shrunk_rate(g, ks[bucket]))
        out.append(g)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ------------------------------------------------------- league-wide splits

def league_splits(d: pd.DataFrame) -> list:
    """Pooled rates per bucket over qualified starters — the league-level answer
    to 'does play fall off in the fourth quarter'."""
    qual = qualified_index(d, ["season", "qb_id"], MIN_SEASON_DB)
    rolled = roll(d, ["season", "qb_id"])
    rolled = rolled[rolled.set_index(["season", "qb_id"]).index.isin(qual)]
    labels = DISPLAY_BUCKETS + [("early", "1st-3rd Q"),
                                ("earlyclose", "One score, before 5:00"),
                                ("all", "All dropbacks")]
    out = []
    for bucket, label in labels:
        g = rolled[rolled.bucket == bucket]
        if g.empty:
            continue
        n = float(g.n.sum())
        out.append({
            "key": bucket, "label": label, "n": int(n),
            "epaDb": round(float(g.epa.sum() / n), 3),
            "succ": round(100 * float(g.succ.sum() / n), 1),
            "sack": round(100 * float(g.sacks.sum() / n), 1),
            "int": round(100 * float(g.ints.sum() / max(g.att.sum(), 1)), 1),
            "td": round(100 * float(g.tds.sum() / max(g.att.sum(), 1)), 1),
            "comp": round(100 * float(g.comp.sum() / max(g.att.sum(), 1)), 1),
        })
    return out


def late_swing(d: pd.DataFrame) -> dict:
    """Paired within-passer tests: each qualified starter's own early-game rate
    against his own late rate, so the answer isn't driven by which passers take
    the most late snaps."""
    rolled = roll(d, ["season", "qb_id"])
    qual = qualified_index(d, ["season", "qb_id"], MIN_SEASON_DB)
    rolled = rolled[rolled.set_index(["season", "qb_id"]).index.isin(qual)]
    piv = rolled.pivot_table(index=["season", "qb_id"], columns="bucket",
                             values=["epa", "n"])
    pairs = {"lateVsEarly": ("late", "early"),
             "clutchVsEarlyClose": ("clutch", "earlyclose"),
             "q4VsQ13": ("q4", "early")}
    out = {}
    for name, (a, b) in pairs.items():
        sub = pd.DataFrame({
            "a": piv[("epa", a)] / piv[("n", a)],
            "b": piv[("epa", b)] / piv[("n", b)],
            "na": piv[("n", a)], "nb": piv[("n", b)]}).dropna()
        sub = sub[(sub.na >= 20) & (sub.nb >= 100)]
        diff = sub.a - sub.b
        t, p = stats.ttest_rel(sub.a, sub.b)
        out[name] = {"n": int(len(sub)), "mean": round(float(diff.mean()), 3),
                     "worse": int((diff < 0).sum()),
                     "t": round(float(t), 2), "p": round(float(p), 4)}
    return out


# ------------------------------------------------------- playoff backtest

def playoff_backtest(d: pd.DataFrame, dd: pd.DataFrame, ks: dict) -> dict:
    """Every starter with real playoff snaps, his regular season against his own
    postseason that year. Paired, so each passer is his own control."""
    reg, post = d[d.season_type == "REG"], d[d.season_type == "POST"]
    rows, per_season = [], {}
    for season in sorted(d.season.unique()):
        r = roll(reg[reg.season == season], ["qb_id", "qb"])
        p = roll(post[post.season == season], ["qb_id", "qb"])
        if p.empty:
            continue
        starters = set(r[(r.bucket == "all") & (r.n >= MIN_SEASON_DB)].qb_id)
        played = set(p[(p.bucket == "all") & (p.n >= MIN_POST_DB)].qb_id)
        ids = starters & played
        if not ids:
            continue
        per_season[int(season)] = len(ids)
        team_of = (reg[(reg.season == season) & (reg.bucket == "all")]
                   .sort_values("n", ascending=False)
                   .drop_duplicates("qb_id").set_index("qb_id")["team"])
        for bucket in ("all", "early", "q4", "late", "clutch"):
            rb = r[(r.bucket == bucket) & r.qb_id.isin(starters)]
            pb = p[(p.bucket == bucket) & p.qb_id.isin(ids)]
            if bucket not in ks or rb.empty or pb.empty:
                continue
            # the season's regular-season scale, but with both sides shrunk at
            # the playoff sample's confidence so the grades are comparable
            rb = rb.set_index("qb_id")
            pb = pb.set_index("qb_id")
            common = rb.index.intersection(pb.index)
            if not len(common):
                continue
            w = pd.Series(np.minimum(rb.loc[common, "n"].values,
                                     pb.loc[common, "n"].values), index=common)
            mu = float(rb.epa.sum() / rb.n.sum())
            base = shrunk_rate(rb.reset_index(), ks[bucket], mu=mu)
            _, c, s = grade(base)
            wr = w.reset_index(drop=True)
            rg, _, _ = grade(shrunk_rate(rb.loc[common].reset_index(), ks[bucket],
                                         weight_n=wr, mu=mu), c, s)
            pg, _, _ = grade(shrunk_rate(pb.loc[common].reset_index(), ks[bucket],
                                         weight_n=wr, mu=mu), c, s)
            for i, qid in enumerate(common):
                rr, pp = rb.loc[qid], pb.loc[qid]
                rows.append({
                    "season": int(season), "bucket": bucket, "qb_id": qid,
                    "qb": rr["qb"], "team": team_of.get(qid, ""),
                    "regN": int(rr.n), "postN": int(pp.n),
                    "regEpa": float(rr.epa / rr.n), "postEpa": float(pp.epa / pp.n),
                    "regG": float(rg.iloc[i]), "postG": float(pg.iloc[i]),
                    "regSucc": float(rr.succ / rr.n), "postSucc": float(pp.succ / pp.n),
                    "regSack": float(rr.sacks / rr.n), "postSack": float(pp.sacks / pp.n),
                    "regInt": float(rr.ints / max(rr.att, 1)),
                    "postInt": float(pp.ints / max(pp.att, 1)),
                })

    R = pd.DataFrame(rows)
    tests = {}
    for bucket in ("all", "early", "q4", "late", "clutch"):
        b = R[(R.bucket == bucket)
              & (R.postN >= (MIN_POST_DB if bucket == "all" else MIN_BUCKET_DB))]
        if len(b) < 5:
            continue
        diff = b.postEpa - b.regEpa
        t, p = stats.ttest_rel(b.postEpa, b.regEpa)
        tests[bucket] = {
            "n": int(len(b)),
            "regEpa": round(float(b.regEpa.mean()), 3),
            "postEpa": round(float(b.postEpa.mean()), 3),
            "diff": round(float(diff.mean()), 3),
            "worse": int((diff < 0).sum()),
            "t": round(float(t), 2), "p": round(float(p), 4),
            "regG": round(float(b.regG.mean()), 1),
            "postG": round(float(b.postG.mean()), 1),
            "regSucc": round(100 * float(b.regSucc.mean()), 1),
            "postSucc": round(100 * float(b.postSucc.mean()), 1),
            "regSack": round(100 * float(b.regSack.mean()), 1),
            "postSack": round(100 * float(b.postSack.mean()), 1),
            "regInt": round(100 * float(b.regInt.mean()), 1),
            "postInt": round(100 * float(b.postInt.mean()), 1),
        }

    # How much of the drop is simply the caliber of January defenses: the
    # regular-season EPA allowed per dropback of the teams that reached the
    # playoffs, against the league as a whole.
    dreg = dd[(dd.season_type == "REG") & (dd.bucket == "all")]
    post_teams = dd[(dd.season_type == "POST") & (dd.bucket == "all")][
        ["season", "team"]].drop_duplicates()
    league, playoff_d = [], []
    for season, grp in dreg.groupby("season"):
        league.append(grp.epa.sum() / grp.n.sum())
        teams = set(post_teams[post_teams.season == season].team)
        pt = grp[grp.team.isin(teams)]
        if len(pt):
            playoff_d.append(pt.epa.sum() / pt.n.sum())
    defense = {"league": round(float(np.mean(league)), 3),
               "playoff": round(float(np.mean(playoff_d)), 3),
               "gap": round(float(np.mean(playoff_d) - np.mean(league)), 3)}

    table = R[R.bucket == "all"].sort_values(["season", "postEpa"],
                                             ascending=[False, False])
    return {
        "perSeason": per_season,
        "tests": tests,
        "defense": defense,
        "rows": [{
            "season": int(r.season), "qb": r.qb, "team": r.team,
            "regN": r.regN, "postN": r.postN,
            "regEpa": round(r.regEpa, 3), "postEpa": round(r.postEpa, 3),
            "regG": round(r.regG, 1), "postG": round(r.postG, 1),
            "regSucc": round(100 * r.regSucc, 1), "postSucc": round(100 * r.postSucc, 1),
            "regSack": round(100 * r.regSack, 1), "postSack": round(100 * r.postSack, 1),
            "regInt": round(100 * r.regInt, 1), "postInt": round(100 * r.postInt, 1),
            "diff": round(r.postEpa - r.regEpa, 3),
            "gDiff": round(r.postG - r.regG, 1),
        } for r in table.itertuples(index=False)],
    }



# ------------------------------------------------------------- ESPN QBR

def name_key(name: str) -> str:
    """'J.Allen' and 'Josh Allen' are the same passer — first initial plus
    surname joins the play-by-play grades to ESPN's board."""
    suffix = {"jr", "sr", "ii", "iii", "iv", "v"}
    parts = [x for x in str(name).replace(".", " ").split() if x]
    parts = [x for x in parts if x.lower().strip(".") not in suffix] or parts
    return (parts[0][0] + "." + parts[-1]).lower() if len(parts) > 1 else str(name).lower()


def qbr_payload(seasons: list, grade_lookup: dict) -> dict:
    """ESPN Total QBR: the 1-32 board per scope and every starter's week-by-week
    line. Missing files (nobody has run build_qbr.py) leave the sections out
    rather than breaking the page."""
    try:
        wk = pd.read_csv(WEEK_CSV)
        sn = pd.read_csv(SEASON_CSV)
    except FileNotFoundError:
        return None
    wk = wk[wk.season_type == "Regular"]
    sn = sn[(sn.season_type == "Regular") & (sn.plays >= MIN_QBR_PLAYS)]

    def board_rows(frame, scope):
        rows = []
        for i, r in enumerate(frame.itertuples(index=False)):
            rows.append({"r": i + 1, "n": r.qb, "t": r.team,
                         "qbr": round(float(r.qbr), 1), "plays": int(r.plays),
                         "pa": round(float(r.pts_added), 1),
                         "g": grade_lookup.get((scope, name_key(r.qb)))})
        return rows

    board = {}
    for season in seasons:
        b = sn[sn.season == season].sort_values("qbr", ascending=False).head(32)
        board[str(season)] = board_rows(b, str(season))
    # career-in-window board: QBR averaged over the window, weighted by plays
    win = sn.copy()
    win["wq"] = win.qbr * win.plays
    agg = (win.groupby(["qb"], as_index=False)
           .agg(wq=("wq", "sum"), plays=("plays", "sum"),
                pts_added=("pts_added", "sum"),
                team=("team", "last"), seasons=("season", "nunique")))
    agg["qbr"] = agg.wq / agg.plays
    board["all"] = board_rows(agg.sort_values("qbr", ascending=False).head(32), "all")

    weekly, league = {}, {}
    for season in seasons:
        w = wk[wk.season == season]
        if w.empty:
            continue
        lines = []
        for _, g in w.groupby("player_id"):
            g = g.sort_values("week_num")
            if len(g) < MIN_QBR_WEEKS:
                continue
            lines.append({
                "n": g.qb.iloc[-1], "t": g.team.iloc[-1],
                "avg": round(float((g.qbr * g.plays).sum() / g.plays.sum()), 1),
                "w": [[int(r.week_num), round(float(r.qbr), 1)]
                      for r in g.itertuples(index=False) if pd.notna(r.qbr)],
            })
        lines.sort(key=lambda x: -x["avg"])
        weekly[str(season)] = lines
        league[str(season)] = [[int(week), round(float(gg.qbr.mean()), 1)]
                               for week, gg in w.groupby("week_num")]
    return {"board": board, "weekly": weekly, "league": league,
            "minPlays": MIN_QBR_PLAYS, "weeks": sorted(weekly.keys())}


# ---------------------------------------------------------------- payload

def entity_rows(d: pd.DataFrame, keys: list, min_db: int, ks: dict,
                label_fn) -> list:
    g = graded_rows(d, keys, min_db, ks)
    if g.empty:
        return []
    out = {}
    for r in g.itertuples(index=False):
        key = tuple(getattr(r, k) for k in keys)
        e = out.setdefault(key, {"b": {}})
        e["b"][r.bucket] = [round(float(r.grade), 1), round(float(r.epa_db), 3),
                            int(r.n)]
        if r.bucket == "all":
            e["db"], e["games"] = int(r.n), int(r.games)
    rows = []
    for key, e in out.items():
        b = e.pop("b")
        if "all" not in b:
            continue
        e.update(label_fn(key, d))
        e["ov"] = b["all"][0]
        # the number the tab exists for: late-game standing against this
        # passer's own standing before the fourth quarter
        if "late" in b and "early" in b:
            e["swing"] = round(b["late"][0] - b["early"][0], 1)
        e["g"] = [b.get(k) for k in PAYLOAD_BUCKETS]
        rows.append(e)
    rows.sort(key=lambda x: -x["ov"])
    return rows


def qb_label(key, d):
    qb_id, qb = key
    teams = (d[(d.qb_id == qb_id) & (d.bucket == "all")]
             .groupby("team")["n"].sum().sort_values(ascending=False))
    return {"n": qb, "t": "/".join(list(teams.index[:2]))}


def payload() -> dict:
    d = load()
    dd = pd.read_csv(DEF_CSV)
    reg = d[d.season_type == "REG"]
    seasons = sorted(int(s) for s in reg.season.unique())
    win = reg[reg.season >= FIRST_SEASON]

    qb_k = bucket_ks(win, ["qb_id", "qb"], MIN_WINDOW_DB)
    team_k = bucket_ks(win, ["team"], MIN_WINDOW_DB)
    season_qb_k = bucket_ks(reg, ["season", "qb_id", "qb"], MIN_SEASON_DB)

    qbs = {"all": entity_rows(win, ["qb_id", "qb"], MIN_WINDOW_DB, qb_k, qb_label)}
    team_label = lambda k, _d: {"n": k[0], "t": k[0]}
    teams = {"all": entity_rows(win, ["team"], MIN_WINDOW_DB, team_k, team_label)}
    for s in seasons:
        sd = reg[reg.season == s]
        qbs[str(s)] = entity_rows(sd, ["qb_id", "qb"], MIN_SEASON_DB, qb_k, qb_label)
        teams[str(s)] = entity_rows(sd, ["team"], MIN_SEASON_DB, team_k, team_label)

    grade_lookup = {}
    for scope, rows in qbs.items():
        for r in rows:
            grade_lookup[(scope, name_key(r["n"]))] = r["ov"]

    return {
        "window": [seasons[0], seasons[-1]],
        "scopes": ["all"] + [str(s) for s in seasons],
        "buckets": [{"key": k, "label": l} for k, l in DISPLAY_BUCKETS],
        # index into each entity's "g" array, same order for QBs and teams
        "order": PAYLOAD_BUCKETS,
        "qbs": qbs,
        "teams": teams,
        "league": league_splits(reg),
        "swing": late_swing(reg),
        "playoffs": playoff_backtest(d, dd, season_qb_k),
        "qbr": qbr_payload(seasons, grade_lookup),
        "minSeason": MIN_SEASON_DB, "minWindow": MIN_WINDOW_DB,
        "minPost": MIN_POST_DB,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="also write the payload to this path")
    args = ap.parse_args()
    p = payload()

    print(f"\nQB grades {p['window'][0]}-{p['window'][1]} "
          f"(qualified starters: {MIN_SEASON_DB}+ dropbacks in a season)\n")
    print("League rates by bucket")
    print(f"{'bucket':<26}{'plays':>8}{'EPA/db':>9}{'succ%':>8}{'sack%':>8}"
          f"{'int%':>7}{'comp%':>8}")
    for r in p["league"]:
        print(f"{r['label']:<26}{r['n']:>8}{r['epaDb']:>9.3f}{r['succ']:>8.1f}"
              f"{r['sack']:>8.1f}{r['int']:>7.1f}{r['comp']:>8.1f}")

    print("\nPaired within-passer tests (each starter against himself)")
    for k, v in p["swing"].items():
        print(f"  {k:<20} n={v['n']:<4} mean diff {v['mean']:+.3f} EPA/db, "
              f"{v['worse']}/{v['n']} worse, t={v['t']}, p={v['p']}")

    pl = p["playoffs"]
    print(f"\nRegular season vs playoffs, {p['window'][0]}-{p['window'][1]} "
          f"({sum(pl['perSeason'].values())} qualified starter-seasons)")
    for bucket, t in pl["tests"].items():
        print(f"  {bucket:<7} n={t['n']:<4} reg {t['regEpa']:+.3f} -> post "
              f"{t['postEpa']:+.3f} ({t['diff']:+.3f}), {t['worse']}/{t['n']} "
              f"declined, t={t['t']}, p={t['p']}; sack% {t['regSack']}->"
              f"{t['postSack']}, int% {t['regInt']}->{t['postInt']}")
    dfe = pl["defense"]
    print(f"  playoff defenses allowed {dfe['playoff']:+.3f} EPA/db in the "
          f"regular season vs league {dfe['league']:+.3f} (gap {dfe['gap']:+.3f})")

    for title, key in (("Biggest playoff drops", 1), ("Biggest playoff risers", -1)):
        print(f"\n{title} (EPA/dropback)")
        for r in sorted(pl["rows"], key=lambda x: key * x["diff"])[:8]:
            print(f"  {r['season']} {r['qb']:<14} {r['regEpa']:>+6.3f} -> "
                  f"{r['postEpa']:>+6.3f} ({r['diff']:+.3f})  grade "
                  f"{r['regG']:>4.0f} -> {r['postG']:<4.0f} {r['postN']} dropbacks")

    q = p.get("qbr")
    if q:
        print("\nESPN Total QBR, top 10 (window average, plays-weighted)")
        for r in q["board"]["all"][:10]:
            print(f"  {r['r']:>2}. {r['n']:<20} {r['t']:<4} QBR {r['qbr']:>5.1f}"
                  f"  {r['plays']:>5} plays   grade "
                  f"{('%.0f' % r['g']) if r['g'] is not None else '  -'}")

    if args.json:
        from pathlib import Path
        Path(args.json).write_text(json.dumps(p, separators=(",", ":")))
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
