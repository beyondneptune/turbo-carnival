#!/usr/bin/env bash
# Launch the Twitter scraper in its own tmux session (Rec #11).
#
# Why a separate session: Twitter's login flow needs a one-time interactive
# step (the user enters their cookie / 2FA code in the headed browser). After
# that the scraper runs headless inside this tmux session and survives even
# if Claude Code exits.
#
# Usage:
#   ./scripts/launch_twitter_scraper.sh           # start (creates session if missing)
#   ./scripts/launch_twitter_scraper.sh attach    # attach to running session
#   ./scripts/launch_twitter_scraper.sh stop      # kill session
#
# Operator step before first run:
#   1. Save your X session cookie to data/secrets/x_cookie.txt
#      (export from a logged-in browser; the file should contain the
#       auth_token cookie value, nothing else)
#   2. Make sure .venv/bin/python has playwright installed:
#         .venv/bin/python -m playwright install chromium
set -euo pipefail

REPO="/Users/weeoto/turbo-carnival"
SESSION="twitter_scraper"
COOKIE="$REPO/data/secrets/x_cookie.txt"
SCRIPT="$REPO/scripts/scrape_twitter.py"
VENV_PY="$REPO/.venv/bin/python"

cmd=${1:-start}

case "$cmd" in
    start)
        if tmux has-session -t "$SESSION" 2>/dev/null; then
            echo "session '$SESSION' already running. use 'attach' to see it."
            exit 0
        fi
        if [[ ! -f "$COOKIE" ]]; then
            echo "ERROR: missing $COOKIE"
            echo "Save your X auth_token cookie there first."
            exit 1
        fi
        if [[ ! -f "$SCRIPT" ]]; then
            echo "ERROR: $SCRIPT not present. Create it before launching."
            exit 1
        fi
        tmux new-session -d -s "$SESSION" \
            "cd $REPO && $VENV_PY $SCRIPT --cookie-file $COOKIE 2>&1 | tee -a data/logs/twitter_scraper.log"
        echo "started tmux session '$SESSION'. Attach with: tmux attach -t $SESSION"
        ;;
    attach)
        tmux attach -t "$SESSION"
        ;;
    stop)
        tmux kill-session -t "$SESSION" 2>/dev/null && echo "stopped" || echo "no session"
        ;;
    status)
        tmux has-session -t "$SESSION" 2>/dev/null && echo "running" || echo "not running"
        ;;
    *)
        echo "usage: $0 {start|attach|stop|status}"
        exit 1
        ;;
esac
