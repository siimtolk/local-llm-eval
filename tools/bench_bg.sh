#!/usr/bin/env bash
# Run bench/bench.py detached from the calling agent, so no command time limit of the agent's harness
# cuts it off (a 2-hour background limit stopped the Inkling 2 Mini run on 2026-11-26). The bench has no
# time limit of its own: it runs until it is done or until `wait --stop-at` / `stop` ends it, and it saves
# after every section and every request (bench.py's checkpoint), so a stopped run is continued later by the
# same bench command (the runner queues the missing sections for the next night: pipeline/resume.py).
# One bench at a time; its state is in <repo>/state/bench-bg (pid, log path, exit code).
#
#   tools/bench_bg.sh start <log file> <bench.py args...>   start it in the background, print the PID
#   tools/bench_bg.sh wait [--stop-at 'YYYY-MM-DD HH:MM'] [--max SECONDS]
#         wait at most SECONDS (default 540, under the agent's 10-minute command limit); exit 0 when the
#         bench has finished (prints its exit code and the log's last lines), 3 while it is still running
#         (call wait again), 4 when --stop-at was reached: the bench was stopped there.
#   tools/bench_bg.sh stop      stop a running bench (the runner calls it after every test)
#   tools/bench_bg.sh status    running / finished / none
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DIR=${LLE_BENCH_BG_DIR:-$ROOT/state/bench-bg}
PIDF=$DIR/pid EXITF=$DIR/exit LOGF=$DIR/log

running() { [ -s "$PIDF" ] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

stop() {
    if running; then
        local pid; pid=$(cat "$PIDF")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 30); do running || break; sleep 1; done
        running && { kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null; }
        echo "stopped" > "$EXITF"
        echo "bench stopped (pid $pid); the finished sections and requests are saved"
    else
        echo "no bench running"
    fi
    rm -f "$PIDF"
}

finished() {
    echo "bench finished, exit $(cat "$EXITF" 2>/dev/null || echo unknown); log $(cat "$LOGF" 2>/dev/null)"
    [ -s "$LOGF" ] && tail -n 15 "$(cat "$LOGF")" 2>/dev/null
}

case "${1:-}" in
start)
    [ $# -ge 3 ] || { sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 2; }
    running && { echo "a bench is already running (pid $(cat "$PIDF")); wait for it or stop it first"; exit 1; }
    log=$2; shift 2
    mkdir -p "$DIR" "$(dirname "$log")"
    rm -f "$EXITF"
    echo "$log" > "$LOGF"
    cd "$ROOT" || exit 1
    # setsid: its own process group (stop ends the whole group), outside the agent's
    LOG="$log" EXITF="$EXITF" setsid nohup bash -c 'python3 bench/bench.py "$@" > "$LOG" 2>&1; echo $? > "$EXITF"' bench "$@" \
        < /dev/null > /dev/null 2>&1 &
    echo $! > "$PIDF"
    sleep 1
    echo "bench started, pid $(cat "$PIDF"), log $log"
    ;;
wait)
    shift
    stop_at="" max=540
    while [ $# -gt 0 ]; do
        case "$1" in
        --stop-at) stop_at=$(date -d "$2" +%s) || exit 2; shift 2 ;;
        --max) max=$2; shift 2 ;;
        *) echo "unknown option $1"; exit 2 ;;
        esac
    done
    end=$(( $(date +%s) + max ))
    while running; do
        now=$(date +%s)
        if [ -n "$stop_at" ] && [ "$now" -ge "$stop_at" ]; then
            stop
            echo "stop time $(date -d "@$stop_at" '+%F %H:%M') reached: the missing sections run the next night"
            exit 4
        fi
        [ "$now" -ge "$end" ] && { echo "still running (pid $(cat "$PIDF")); call wait again"; exit 3; }
        sleep 15
    done
    rm -f "$PIDF"
    finished
    ;;
stop) stop ;;
status)
    if running; then echo "running (pid $(cat "$PIDF"))"
    elif [ -e "$EXITF" ]; then finished
    else echo "none"; fi
    ;;
*) sed -n '2,/^set -u/p' "$0" | sed '$d'; exit 2 ;;
esac
