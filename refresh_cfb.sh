#!/usr/bin/env bash
# Daily college refresh: pull the sportsdataverse releases + ESPN scoreboard,
# rebuild features, retrain, project props, fetch outside opinions, rebuild
# the college page, publish it. Everything here is keyless and unmetered.
#
#   refresh_cfb.sh
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG=refresh_cfb.log

{
  echo "=== cfb refresh $(date -Is) ==="
  $PY -m cfb.features
  $PY -m cfb.train | tail -12
  $PY -m cfb.props || echo "cfb props failed (non-fatal)"
  $PY -m cfb.fetch_sources || echo "cfb sources failed (non-fatal)"
  $PY -m cfb.export
  if [ -d /var/www/nfl ]; then
    mkdir -p /var/www/nfl/college
    cp web/college/index.html /var/www/nfl/college/index.html
    echo "deployed to /var/www/nfl/college"
  fi
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
