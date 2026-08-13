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
- **Value-weighted outs**: each Out/Doubtful skill player contributes his
  rating, so Ja'Marr Chase out ≠ a WR4 out. Defense/OL stay snap-weighted
  (no public per-player EPA exists for them).

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
