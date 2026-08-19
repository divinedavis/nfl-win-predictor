#!/usr/bin/env bash
# Daily refresh: pull fresh nflverse data (schedules, injuries, snap counts,
# player stats) and Open-Meteo forecasts, retrain, rebuild the web app, and
# publish it to the nginx docroot. Run from the project directory by cron.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG=refresh.log

{
  echo "=== refresh $(date -Is) ==="
  if [ -f .env ]; then set -a; . ./.env; set +a; fi
  $PY fetch_spread_odds.py || echo "spread odds fetch failed (non-fatal)"
  # Player-prop lines cost ~640 credits per pull — Wednesdays (catches
  # Wed/Thu kickoffs) and Sundays only.
  DOW=$(date +%u)
  if [ "$DOW" = 3 ] || [ "$DOW" = 7 ]; then
    $PY fetch_props.py || echo "prop lines fetch failed (non-fatal)"
    $PY paper_trade.py --pick || true
    $PY paper_trade.py --settle || true
  fi
  $PY build_clutch.py --update || echo "clutch refresh failed (non-fatal)"
  # quarterback splits: only the current season is re-pulled, history is static
  $PY build_qb_splits.py --update || echo "qb splits refresh failed (non-fatal)"
  $PY build_qbr.py || echo "espn qbr refresh failed (non-fatal)"
  $PY features.py
  $PY train.py | tail -4
  $PY props.py || echo "props projection failed (non-fatal)"
  $PY export_web.py
  if [ -d /var/www/nfl ]; then
    cp web/index.html /var/www/nfl/index.html
    # Static, but copied every run so an edit in the repo cannot sit
    # undeployed the way docroot pages have drifted on other projects.
    cp web/how-it-works.html /var/www/nfl/how-it-works.html
    echo "deployed to /var/www/nfl"
  fi
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
