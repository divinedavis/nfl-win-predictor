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
```

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
