#!/usr/bin/env bash
# Refit the world model's dynamics and publish them to the droplet.
#
# Cadence is monthly on purpose. The dynamics this fits -- yards per carry on
# 2nd and 7, field-goal accuracy by distance, how much clock a play burns --
# move over seasons, not weeks; one more week of games against 365,000 plays
# changes nothing measurable. What DOES change weekly is team strength, and
# that lives in features.parquet, which refresh.sh already rebuilds daily. So
# a quarterback ruled out on Wednesday reaches Sunday's live win probability
# without any of this running.
#
# It runs here rather than on the droplet because fitting eleven seasons peaks
# around 3 GB, and the droplet has ~1.4 GB free alongside the other sites.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
LOG=retrain.log
DROPLET=root@104.248.12.129

{
  echo "=== retrain $(date -Is) ==="
  # Fit to a temporary file so a failed run never replaces a working model.
  $PY world_model.py --first-season 2015 --last-season "$(date +%Y)" \
      -o world_model.pkl.new
  mv world_model.pkl.new world_model.pkl
  if rsync -az --timeout=60 world_model.pkl "$DROPLET":/opt/nfl-predictor/; then
    echo "published to the droplet"
  else
    echo "rsync failed; the droplet keeps the model it already had"
  fi
  echo "=== done $(date -Is) ==="
} >> "$LOG" 2>&1
