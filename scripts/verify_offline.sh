#!/usr/bin/env bash
# System-level offline check (Linux): runs verify_offline.py inside a new network
# namespace that has no interfaces except a down loopback, so no process — including
# subprocess workers — can reach the network.
#   scripts/verify_offline.sh [--models-dir DIR] [FILE ...]
set -euo pipefail
PY="${PYTHON:-python3}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if ! unshare -rn true 2>/dev/null; then
  echo "unshare -rn is not permitted here; disconnect the network manually and run:"
  echo "  $PY $HERE/verify_offline.py $*"
  exit 2
fi
exec unshare -rn "$PY" "$HERE/verify_offline.py" "$@"
