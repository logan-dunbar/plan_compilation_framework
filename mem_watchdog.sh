#!/usr/bin/env bash
# Shed the largest downward job when MemAvailable drops below the floor.
# Runs until no downward processes remain.
FLOOR_KB=$((12 * 1024 * 1024))
while pgrep -f 'bin/downwar[d]' >/dev/null; do
  avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$avail" -lt "$FLOOR_KB" ]; then
    victim=$(ps -eo pid,rss,args --sort=-rss | grep 'bin/downwar[d]' | head -1)
    pid=$(echo "$victim" | awk '{print $1}')
    job=$(echo "$victim" | grep -oE 'work/[a-z0-9_]+/[a-z_]+/seed_[0-9]' | head -1)
    echo "$(date +%H:%M:%S) MemAvailable=$((avail/1024)) MB < floor: killing $pid ($job, $(echo "$victim" | awk '{printf "%.1f", $2/1048576}') GB)"
    kill "$pid"
    sleep 20
  fi
  sleep 30
done
echo "$(date +%H:%M:%S) no downward processes left; watchdog exiting"
