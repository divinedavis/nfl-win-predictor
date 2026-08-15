# NFL Predictor

Weekly NFL win probabilities from an XGBoost model blended with an Elo rating
system, trained on free nflverse data (1999–present). Fully local, no API costs.

## Setup

```bash
python3.12 -m venv .venv          # needs Homebrew python3.12+ (system 3.9 is too old)
.venv/bin/pip install -r requirements.txt
# macOS: xgboost needs `brew install libomp`
```

## Usage

```bash
.venv/bin/python features.py      # download fresh data, rebuild features.parquet
.venv/bin/python train.py         # walk-forward backtest + save model.json
.venv/bin/python predict.py       # picks for the next unplayed week
.venv/bin/python predict.py --season 2026 --week 5
.venv/bin/python position_impact.py  # which position's injuries move win prob
.venv/bin/python export_web.py    # rebuild web/index.html for the web app
```

## Individual player ratings

Every QB/RB/WR/TE carries a personal rating: his EPA per game over his last
10 games, shrunk toward zero (`RATING_WINDOW`/`RATING_SHRINK` in features.py),
so rookies and backups sit near 0 and stars ride well above. The model uses:

- **QB ratings** for each game's expected starter (the announced starter
  historically; the incumbent for future games, or 0 if he's ruled out) —
  `qb_val_diff` ranks top-5 in feature importance.
- **Value-weighted outs**: each Out/Doubtful player contributes his rating,
  so Ja'Marr Chase out ≠ a WR4 out. Defenders are valued too — no public
  per-player defensive EPA exists, so DL/LB/DB ratings come from weighted
  box-score playmaking per game (sacks, INTs, passes defended, TFLs, forced
  fumbles; `DEF_VALUE_WEIGHTS`) — Myles Garrett rates ~6.4 vs ~0 for a
  rotational lineman. OL stays snap-weighted (no public stats at all).

`features.py` also snapshots current ratings to `player_ratings.csv`.

## Deployment

Lives at `/opt/nfl-predictor` on the 104.248.12.129 droplet, served by nginx
at `http://104.248.12.129/nfl/` (and `nfl.divinedavis.com` once DNS exists).
`refresh.sh` runs daily at 10:30 UTC via root's crontab: fresh injuries,
weather forecasts, retrain, redeploy to `/var/www/nfl`.

## Web app

`export_web.py` bakes the season's predictions into `web/index.html` (from
`web/template.html`) — a self-contained page with a team filter: pick a team,
see the win probability for every game on their schedule, next game featured.
Published as a Claude artifact; republish the same file to update in place.

Re-run all three each week during the season so ratings and rolling form
include the latest results. `predict.py` on a completed week shows HIT/MISS
and the week record.

## How it works

- **Elo** (538-style): K=20, home-field +52, margin-of-victory multiplier,
  1/3 reversion to the mean each offseason, franchise moves handled.
- **Rolling form** (last 8 games, leak-free): point differential, win rate,
  points for/against, offensive EPA, defensive EPA allowed.
- **Injuries** (2009+): players Out/Doubtful and Questionable per team, plus a
  QB-change flag — did the starter change from the team's previous game
  (announced pregame, so leak-free; for future games it fires when the
  incumbent QB is Out/Doubtful on the injury report).
- **Position-weighted outs** (2013+, snap counts): every Out/Doubtful player is
  weighted by his rolling snap share and summed per position group (QB, RB,
  WR, TE, OL, DL, LB, DB) — a full-time starter out ≈ 1.0, a rotational
  backup ≈ his share. `position_impact.py` runs the counterfactual: WR outs
  cost the most (~4% win probability for a full-time WR), then RB; QB shows
  small here only because the QB-change flag already captures it; individual
  defensive absences barely register on single-game outcomes.
- **Weather**: recorded temp/wind for training; free Open-Meteo forecasts at
  predict time for outdoor games within 16 days of kickoff (no API key).
- **Context**: rest days, divisional game, dome, week number.
- **Model**: XGBoost classifier on the above; final probability is
  `0.4 * xgboost + 0.6 * elo` (blend weight swept in backtest — it beats
  either component on calibration).

## Hypothesis scoreboard

Every idea gets the same treatment: build the feature, run the walk-forward
backtest, promote only on a Brier improvement across a clear majority of
seasons. What has been tested so far —

| Hypothesis | Script | Verdict |
|------------|--------|---------|
| Finishing / clutch (late one-score EPA) | `clutch_ablation.py` | **Promoted** — Brier better in 11/11 seasons |
| Kicker quality (distance-adjusted FGOE) | `kicker_ablation.py` | Real skill (split-half r=+0.55), no model gain — Elo absorbs it |
| Coaching + head-to-head matchup history | `coach_ablation.py` | Nothing beyond Elo; residual r=−0.17 |
| Upset indicators (luck, sacks, blitz, travel, letdown) | `backtest_groups.py` | None promoted; two kept as display flags |
| **2+ starting offensive linemen out** | `ol_ablation.py` | Real in raw numbers, already priced — see below |

The pattern: a feature wins only when it carries information Elo and rolling
EPA could not already have absorbed through game results.

### Offensive line injuries

Teams missing two or more starting linemen score 21.6 points a game and win
44%, versus 23.4 and 52% for teams with a whole line. Nearly all of that gap
is *which* teams get hurt, not the injury — measured against a team's own
last-8-game form, the same teams score +0.16 points (z +0.51) and allow +0.06
percentage points of sack rate (z +0.38). QB, RB and WR production versus each
player's own trailing-4 form moves less than a yard in every direction, and
props pinball loss is flat to worse. `features.py` builds
`{side}_ol_starters_out` and `{side}_ol_multi_out` (the ≥2 threshold) as a
candidate group; neither is in `FEATURES`. The dashboard names a patched-up
line as context, because linemen have no stat line and so never appear in the
key-absence list.

## Backtest (walk-forward, 2015–2025, 3,018 games)

Each season predicted by a model trained only on prior seasons.

| Predictor            | Accuracy | Brier |
|----------------------|----------|-------|
| Always pick home     | 55.0%    | —     |
| Elo alone            | 64.5%    | .2222 |
| **Blend (shipped)**  | **64.9%**| **.2206** |
| Vegas closing spread | 66.1%    | —     |

Selective picking (`predict.py --min-conf`): confidence is well calibrated,
so acting only on the model's strongest picks buys accuracy —

| Confidence cutoff | Picks/week | Accuracy |
|-------------------|-----------|----------|
| ≥ 0.60            | ~9.5      | 69.7%    |
| ≥ 0.65 (starred)  | ~6.8      | 73.1%    |
| ≥ 0.70            | ~4.6      | 76.5%    |

Honest read: single-game NFL outcomes are mostly noise, and ~67% on all games
is the practical ceiling (the Vegas closing line, which prices in everything,
sits at 66.1%). Injury + QB-change + weather features lifted the blend past
plain Elo; the remaining gap to Vegas is real. 70%+ on every game is not a
thing any honest model achieves — 70%+ on the starred subset is.

`spread_line` is stored in the feature table as a benchmark only; it is never
fed to the model.

## Player props (paper trading)

Quantile projections (P10/P25/P50/P75/P90) for the four liquid prop markets —
passing yards, rushing yards, receiving yards, receptions — from rolling
usage/efficiency, opponent positional defense, and game context. Walk-forward
validated 2021–2025: 6–10% better pinball loss than a player-history baseline,
with honest coverage (the 50% band catches ~52%, the 80% band ~81–86%).

```bash
.venv/bin/python props.py --validate   # walk-forward backtest
.venv/bin/python props.py              # project next week -> props_projections.csv
ODDS_API_KEY=... .venv/bin/python fetch_props.py   # snapshot DraftKings lines
.venv/bin/python paper_trade.py --pick     # log edges >=5% as $0 paper picks
.venv/bin/python paper_trade.py --settle   # score finished weeks
```

**No real money.** A backtest of the game model against 11 seasons of closing
moneylines lost at every edge threshold (and parlays compound the leak, -10%
ROI), so props follow a pre-committed rule: nothing is staked unless the paper
record clears the 53.3% breakeven (-114 juice) over 200+ settled picks.
Historical prop lines are paywalled — paper trading the live season is the only
honest test. Lines come from The Odds API free tier (hard-capped: one snapshot
per day, one book, four markets).
