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
- **Context**: rest days, divisional game, dome, week number.
- **Model**: XGBoost classifier on the above; final probability is
  `0.4 * xgboost + 0.6 * elo` (blend weight swept in backtest — it beats
  either component on calibration).

## Backtest (walk-forward, 2015–2025, 3,018 games)

Each season predicted by a model trained only on prior seasons.

| Predictor            | Accuracy | Brier |
|----------------------|----------|-------|
| Always pick home     | 55.0%    | —     |
| XGBoost alone        | 63.0%    | .2232 |
| Elo alone            | 64.5%    | .2222 |
| **Blend (shipped)**  | **64.2%**| **.2214** |
| Vegas closing spread | 66.1%    | —     |

Honest read: single-game NFL outcomes are mostly noise. Elo carries most of
the signal; XGBoost's extra features buy better-calibrated probabilities
(lowest Brier), not more raw accuracy. Vegas remains ~2 points better —
treat the spread as the ceiling, not the competition.

`spread_line` is stored in the feature table as a benchmark only; it is never
fed to the model.
