#!/bin/bash
# AlgoTrade startup script.
# Starts the dashboard and trading engine together.
# Logs everything to logs/algotrade-YYYY-MM-DD.log.
# Auto-restarts the engine on crash or disconnect (Ctrl+C stops cleanly).

cd "$(dirname "$0")"

mkdir -p logs
LOG="logs/algotrade-$(date '+%Y-%m-%d').log"

# Prevent display/system sleep
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true

echo "============================================"
echo "  AlgoTrade — $(date)"
echo "  Dashboard: http://localhost:3000"
echo "  Log file:  $LOG"
echo "  Press Ctrl+C to stop everything."
echo "============================================"

# Start Next.js dashboard in background
npm --prefix dashboard-ui run dev >> "logs/dashboard-$(date '+%Y-%m-%d').log" 2>&1 &
DASH_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Dashboard started (PID $DASH_PID)" | tee -a "$LOG"

cleanup() {
    echo ""
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Shutting down..." | tee -a "$LOG"
    kill "$DASH_PID" 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

while true; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting engine..." | tee -a "$LOG"
    .venv/bin/python main.py 2>&1 | tee -a "$LOG"
    ENGINE_EXIT=${PIPESTATUS[0]}

    if [ $ENGINE_EXIT -eq 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Engine stopped cleanly. Exiting." | tee -a "$LOG"
        kill "$DASH_PID" 2>/dev/null
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Engine exited (code $ENGINE_EXIT). Reconnecting in 60s..." | tee -a "$LOG"
    sleep 60
done
