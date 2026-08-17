"""Rerun of the two configurations affected by the plan-memo cycle bug:
the 20x20 A4_no_latch ablation and the 50x50 no_latch_slow diagnostic,
under the fixed agent. Everything else was latch-on and unaffected."""
from dataclasses import replace

from plan_compilation_framework.core import Constant
from plan_compilation_framework.experiment import (
  AgentSpec, EnvSpec, ExperimentSpec, run_experiment,
)
from plan_compilation_framework.labelling import JsdConfig
from plan_compilation_framework.run_gridworld import paper_pcf_config


def main():
  base20 = paper_pcf_config(q0=-300., anneal=300)
  run_experiment(ExperimentSpec(
    name='ablations', env=EnvSpec(size=(20, 20), max_steps=100_000),
    agents={'A4_no_latch': AgentSpec('pcf', planner='astar',
                                     config=replace(base20, label_rule=('jsd', JsdConfig(latch=False))))},
    episodes=1000, seeds=5), workers=5)

  base50 = paper_pcf_config(q0=-1250., anneal=800)
  run_experiment(ExperimentSpec(
    name='diag_50x50', env=EnvSpec(size=(50, 50), max_steps=1_000_000),
    agents={'no_latch_slow': AgentSpec('pcf', planner='astar',
                                       config=replace(base50, label_rule=('jsd', JsdConfig(
                                         tau_l=Constant(0.95), latch=False))))},
    episodes=10_000, seeds=4), workers=4)


if __name__ == '__main__':
  main()
