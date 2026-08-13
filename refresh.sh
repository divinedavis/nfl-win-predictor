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
  $PY features.py
  $PY train.py | tail -4
  $PY props.py || echo "props projection failed (non-fatal)"
  $PY export_web.py
  if [ -d /var/www/nfl ]; then
    cp web/index.html /var/www/nfl/index.html
    echo "deployed to /var/www/nfl"
  fi
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
