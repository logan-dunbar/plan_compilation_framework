#!/usr/bin/env bash
# Run any plan_compilation_framework module inside a memory-capped systemd
# scope, so a runaway experiment can only kill itself — never the desktop.
#
#   ./run.sh -m plan_compilation_framework.run_gridworld --preset paper
#   PCF_MEM=80G ./run.sh -m plan_compilation_framework.run_diagnostics
#
# Defaults: MemoryMax=64G (hard, OOM-kills the scope), no swap for the scope
# (so it fails fast instead of thrashing the machine into oomd territory).
# The 17 Aug 2026 workstation crash is why this exists.
set -euo pipefail
cd "$(dirname "$0")"

MEM="${PCF_MEM:-64G}"
UNIT="pcf-$(date +%H%M%S)-$$"

exec systemd-run --user --scope -q \
  -p "MemoryMax=${MEM}" -p "MemorySwapMax=0" \
  --unit="${UNIT}" \
  uv run python "$@"
