#!/usr/bin/env bash
# Daily refresh: pull fresh nflverse data (schedules, injuries, snap counts,
# player stats) and Open-Meteo forecasts, retrain, rebuild the web app, and
# publish it to the nginx docroot. Run from the project directory by cron.
#
#   refresh.sh              full run: retrain, props, rebuild, deploy
#   refresh.sh --gameday    fast run: late injury news + market, rebuild, deploy
#
# Gameday mode exists because the official injury report is filed Wednesday to
# Friday but inactives land 90 minutes before kickoff. It skips retraining and
# the weekly data builds — nothing they read changes on a Sunday morning — so
# it costs about a minute and can run in each pre-kickoff window.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG=refresh.log
MODE=${1:-}

alert() { echo "!!!!!!!!!! ALERT: $* !!!!!!!!!!"; }

{
  echo "=== refresh${MODE:+ $MODE} $(date -Is) ==="
  if [ -f .env ]; then set -a; . ./.env; set +a; fi

  # Late-breaking availability. Free, no API key, safe to run repeatedly.
  $PY fetch_inactives.py || echo "inactives fetch failed (non-fatal)"
  # 2 credits: 2 markets x 1 region. Every US book costs the same as one.
  $PY fetch_spread_odds.py || alert "spread odds fetch FAILED — market feature will fall back to the schedule's line"

  # Closing line value. --record locks in the model's number against the
  # market's the first time a game has enough books; it never revises a
  # recorded game, so this is safe to run as often as the odds refresh.
  $PY clv.py --record || echo "clv record failed (non-fatal)"
  $PY clv.py --settle || echo "clv settle failed (non-fatal)"

  # These two files exist nowhere else: /opt/nfl-predictor is a plain copy, not
  # a checkout, and both are gitignored because they are append-only runtime
  # data. Losing the droplet would lose the entire CLV experiment with it.
  mkdir -p backups
  for f in clv_picks.csv spread_odds_history.csv; do
    [ -f "$f" ] && cp "$f" "backups/$f.$(date +%Y%m%d)"
  done
  # Two weeks of dailies is plenty to notice a problem and recover.
  find backups -name '*.csv.*' -mtime +14 -delete 2>/dev/null || true

  if [ "$MODE" = "--gameday" ]; then
    # Rebuild features so the resolved inactives and the fresh consensus reach
    # the page, then republish. The model itself is normally unchanged.
    $PY features.py
    # ...unless FEATURES has moved since model.json was written, in which case
    # skipping training would abort the export on a feature_names mismatch
    # after the fetches had already run. Retrain instead of dying.
    if ! $PY scripts/check_model_features.py; then
      alert "model.json is stale for the current FEATURES — retraining on the gameday path"
      $PY train.py | tail -4
    fi
    $PY export_web.py
    $PY publish_results.py || echo "results publish failed (non-fatal)"
  else
    # Player-prop lines cost ~640 credits per pull — Wednesdays (catches
    # Wed/Thu kickoffs) and Sundays only.
    DOW=$(date +%u)
    if [ "$DOW" = 3 ] || [ "$DOW" = 7 ]; then
      # Props are the one job big enough to exhaust the plan on its own, so a
      # failure here is reported loudly rather than swallowed: a silent skip
      # looks identical to a healthy week with no edges.
      if $PY fetch_props.py; then
        $PY paper_trade.py --pick || alert "paper trade --pick FAILED"
        $PY paper_trade.py --settle || alert "paper trade --settle FAILED"
      else
        alert "prop lines fetch FAILED — paper trading did not run this cycle"
        alert "check the Odds API balance: quota exhaustion looks exactly like this"
      fi
    fi
    $PY build_clutch.py --update || echo "clutch refresh failed (non-fatal)"
    # quarterback splits: only the current season is re-pulled, history is static
    $PY build_qb_splits.py --update || echo "qb splits refresh failed (non-fatal)"
    $PY build_qbr.py || echo "espn qbr refresh failed (non-fatal)"
    $PY features.py
    $PY train.py | tail -4
    $PY props.py || echo "props projection failed (non-fatal)"
    # Write down what the model thinks today and score anything that has been
    # played since, so there is a permanent record to learn from later.
    $PY diary.py --settle --record || echo "diary update failed (non-fatal)"
    $PY diary.py --report | tail -12 || true
    $PY clv.py --report || true
    $PY export_web.py
    # Results the picks page grades against. Runs after export_web.py because
    # that is what appends this week's players to the projection log.
    $PY publish_results.py || echo "results publish failed (non-fatal)"
  fi

  if [ -d /var/www/nfl ]; then
    cp web/index.html /var/www/nfl/index.html
    # Static, but copied every run so an edit in the repo cannot sit
    # undeployed the way docroot pages have drifted on other projects.
    cp web/how-it-works.html /var/www/nfl/how-it-works.html
    echo "deployed to /var/www/nfl"
  fi
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
