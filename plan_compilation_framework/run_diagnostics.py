"""Parameter-sensitivity sweep on the 50x50 grid (thesis requirement C4).

One-factor variants around the paper-faithful base configuration, so the
thesis can state which parameters the method is sensitive to. The first
version of this sweep (17 Aug 2026) ran against a base that mis-set the
learnt-phase Q rate to 0.4 (the l-value rate) and per-episode alpha; those
two variants are kept here as sensitivity points because they showed a
reward plateau at planner level.

  ./run.sh -m plan_compilation_framework.run_diagnostics
"""
from __future__ import annotations

from dataclasses import replace

from plan_compilation_framework.core import Constant, Harmonic, Linear
from plan_compilation_framework.experiment import (
  AgentSpec, EnvSpec, ExperimentSpec, run_experiment,
)
from plan_compilation_framework.labelling import JsdConfig
from plan_compilation_framework.run_gridworld import paper_pcf_config


def main():
  env = EnvSpec(size=(50, 50), max_steps=1_000_000)
  base = paper_pcf_config(q0=-1250., anneal=800)

  variants = {
    'base': base,
    # the two mis-set knobs of the first sweep, kept as sensitivity points
    'alpha_learnt_0.4': replace(base, alpha_learnt=Constant(0.4)),
    'alpha_per_episode': replace(base, alpha_per_visit=False),
    # exploration
    'anneal_eps': replace(base, eps_learner=Linear(0.1, 0.01, 5000)),
    'longer_explore': paper_pcf_config(q0=-1250., anneal=5000),
    'no_explorer': replace(base, use_explorer=False),
    # labelling thresholds
    'tau_l_0.95': replace(base, label_rule=('jsd', JsdConfig(tau_l=Constant(0.95)))),
    'tau_d_0.3': replace(base, label_rule=('jsd', JsdConfig(tau_d=Constant(0.3)))),
    'alpha_l_0.1': replace(base, label_rule=('jsd', JsdConfig(alpha_l=Constant(0.1)))),
    # updates
    'td_everywhere': replace(base, mc_until_learnt=False),
    'alpha_const_0.1': replace(base, alpha=Constant(0.1), alpha_per_visit=False),
    'q0_-300': replace(base, q0_learner=-300.),
    'q0_-5000': replace(base, q0_learner=-5000.),
    'no_latch_slow': replace(base, label_rule=('jsd', JsdConfig(
      tau_l=Constant(0.95), latch=False))),
  }

  agents = {name: AgentSpec('pcf', planner='astar', config=cfg)
            for name, cfg in variants.items()}
  agents['qlearning'] = AgentSpec('qlearning')
  agents['astar'] = AgentSpec('planner', planner='astar')

  spec = ExperimentSpec(name='sensitivity_50x50', env=env, agents=agents,
                        episodes=10_000, seeds=5)
  run_experiment(spec, workers=48)


if __name__ == '__main__':
  main()
