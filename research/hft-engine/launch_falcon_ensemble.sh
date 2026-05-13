#!/usr/bin/env bash
# Launch the 8-instance HFT shadow ensemble on FALCON account.
#
# Each instance:
#   - same Rust binary, --shadow-only flag → no orders ever
#   - different RLS hyperparameter config
#   - own SQLite DB + tmux session + kill file
#   - FALCON Alpaca paper creds (aliased to ALPACA_FALCON_* the engine expects)
#
# Usage:
#   ./launch_falcon_ensemble.sh           # start all 8
#   ./launch_falcon_ensemble.sh stop      # tmux kill them all
#   ./launch_falcon_ensemble.sh status    # show running instances

set -euo pipefail
ROOT="/Users/weeoto/turbo-carnival"
ENG_DIR="$ROOT/research/hft-engine"
LIVE_DIR="$ROOT/research/live"
ENS_CONFIG_DIR="$ENG_DIR/config/ensemble"
LOG_DIR="$LIVE_DIR/logs"
mkdir -p "$LOG_DIR"

case "${1:-start}" in
  stop)
    for N in 1 2 3 4 5 6 7 8; do
      tmux kill-session -t "hft_ens_$N" 2>/dev/null || true
      echo "killed hft_ens_$N"
    done
    exit 0
    ;;
  status)
    tmux ls 2>/dev/null | grep -E "^hft_ens_" || echo "no ensemble sessions running"
    exit 0
    ;;
esac

# Sanity: ensure env file exists
if [ ! -f "$ROOT/council/.env.local" ]; then
  echo "FATAL: $ROOT/council/.env.local missing"
  exit 1
fi

# Sanity: ensure binary built
if [ ! -x "$ENG_DIR/target/release/hft-engine" ]; then
  echo "FATAL: hft-engine binary missing — run 'cargo build --release' in $ENG_DIR"
  exit 1
fi

cd "$ENG_DIR"
for N in 1 2 3 4 5 6 7 8; do
  SESSION="hft_ens_$N"
  CFG="$ENS_CONFIG_DIR/ens_${N}.yaml"
  DB="$LIVE_DIR/falcon_ens_${N}.sqlite3"
  KILLF="$LIVE_DIR/STOP_FALCON_ENS_${N}"
  LOG="$LOG_DIR/falcon_ens_${N}.log"

  if [ ! -f "$CFG" ]; then
    echo "skip $SESSION (no config $CFG)"
    continue
  fi

  # Kill any prior session so launch is idempotent
  tmux kill-session -t "$SESSION" 2>/dev/null || true

  # Build the command. We export FALCON creds (the engine reads ALPACA_FALCON_*).
  CMD="set -a && source $ROOT/council/.env.local && set +a"
  CMD="$CMD && export HFT_SQLITE_PATH=\"$DB\""
  CMD="$CMD && export HFT_KILL_FILE=\"$KILLF\""
  CMD="$CMD && echo \"[FALCON ens-$N] starting at \$(date) cfg=$CFG db=$DB\""
  CMD="$CMD && /usr/bin/caffeinate -dimsu ./target/release/hft-engine --shadow-only --config \"$CFG\" 2>&1 | tee -a \"$LOG\""

  tmux new -d -s "$SESSION" "$CMD"
  echo "launched $SESSION  cfg=ens_${N}  db=$(basename "$DB")"
done

echo ""
echo "--- post-launch tmux ---"
tmux ls 2>/dev/null | grep -E "^hft_ens_"
