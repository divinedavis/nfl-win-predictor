#!/usr/bin/env bash
# Live win probability, refreshed while games are being played.
#
# One request to ESPN's public scoreboard per run. When nothing is mid-drive
# the script writes an empty payload and exits in about a second, so the cost
# of running this on a schedule is a single HTTP call.
set -euo pipefail
cd "$(dirname "$0")"

DOCROOT=/var/www/nfl
LOG=live.log

{
  .venv/bin/python live_wp.py --n 1500 --out "$DOCROOT/live.json"
} >> "$LOG" 2>&1

# Keep the log from growing without bound; this runs every few minutes.
if [ "$(wc -c < "$LOG")" -gt 1000000 ]; then
  tail -c 200000 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
fi
