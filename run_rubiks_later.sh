#!/usr/bin/env bash
# Wait until no other Fast Downward jobs are running, then run the Rubik's cube
# showcase on its own: FF only (CEA leaks memory across engine constructions in
# this FD version), PCF 10k episodes, planner baseline 2k (stationary), 3 seeds.
cd /home/logan/dev/plan_work/plan_compilation_framework
while [ "$(pgrep -fc 'bin/downwar[d]')" != "0" ]; do sleep 120; done
echo "$(date) machine quiet; starting rubiks"
PCF_MEM=100G ./run.sh -m plan_compilation_framework.run_pddl --preset paper --episodes 10000 --planner-episodes 2000 --seeds 3 --agents fd_gff pcf_fd_gff --domains rubiks1 --workers 6
uv run python -m plan_compilation_framework.plots pddl_rubiks1
echo "$(date) rubiks done"
