#!/usr/bin/env bash
# Pull the append-only research data off the droplet into the local checkout.
#
# clv_picks.csv and spread_odds_history.csv accumulate only on the droplet --
# /opt/nfl-predictor is a plain copy, not a git checkout, and both files are
# gitignored because they grow every run. They are also the entire record of
# the CLV experiment, so a droplet rebuild without a local copy loses a season
# of data that cannot be reconstructed: no free source of historical opening
# lines exists.
#
# Run it whenever you want a local snapshot. Safe to repeat; it overwrites.
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${NFL_HOST:-root@104.248.12.129}
for f in clv_picks.csv spread_odds_history.csv; do
  if scp -q "$HOST:/opt/nfl-predictor/$f" "$f"; then
    echo "pulled $f ($(wc -l < "$f") lines)"
  else
    echo "could not pull $f (may not exist yet)" >&2
  fi
done
