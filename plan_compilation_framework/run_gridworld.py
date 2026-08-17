"""GridWorld experiment entry point.

  uv run python -m plan_compilation_framework.run_gridworld --preset smoke
  uv run python -m plan_compilation_framework.run_gridworld --preset validate
  uv run python -m plan_compilation_framework.run_gridworld --preset labelling
  uv run python -m plan_compilation_framework.run_gridworld --preset ablations
  uv run python -m plan_compilation_framework.run_gridworld --preset paper

Presets encode the resubmission's experiment matrix. `paper` reproduces the
ECML'22 configuration (50x50, harmonic alpha, q0=-1250, no step limit).
"""
from __future__ import annotations

import argparse
from dataclasses import replace

from plan_compilation_framework.core import Constant, Harmonic, Linear
from plan_compilation_framework.agents import PCFConfig, QLearningConfig
from plan_compilation_framework.experiment import (
  AgentSpec, EnvSpec, ExperimentSpec, run_experiment,
)
from plan_compilation_framework.labelling import JsdConfig


def paper_pcf_config(q0: float, anneal: int) -> PCFConfig:
  """The ECML'22 hyperparameters (from the 2022 experiments.json).

  alpha is the per-(state, action) harmonic of the 2022 code (first update a
  full replace, decaying to 0.1); the 0.4 in that config was the l-value rate
  alpha_l, not a Q learning rate.
  """
  return PCFConfig(
    gamma=1.0,
    alpha=Harmonic(1.0, 0.1),
    alpha_per_visit=True,
    alpha_learnt=None,
    alpha_explorer=Constant(0.1),
    eps_learner=Constant(0.1),
    eps_explorer=Constant(0.1),
    explore_prob=Linear(0.1, 0.001, anneal),
    quota_frac=Linear(0.2, 0.05, anneal),
    q0_learner=q0,
    q0_explorer=0.,
    label_rule=('jsd', JsdConfig(tau_d=Constant(0.1), tau_l=Constant(0.85),
                                 alpha_l=Constant(0.4))),
  )


def build_preset(name: str) -> ExperimentSpec:
  if name == 'smoke':
    env = EnvSpec(size=(12, 12), max_steps=50_000)
    cfg = paper_pcf_config(q0=-300., anneal=100)
    return ExperimentSpec(
      name='smoke', env=env, episodes=300, seeds=2,
      agents={
        'qlearning': AgentSpec('qlearning'),
        'astar': AgentSpec('planner', planner='astar'),
        'pcf_astar': AgentSpec('pcf', planner='astar', config=cfg),
      })

  if name == 'validate':
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    cfg = paper_pcf_config(q0=-300., anneal=300)
    return ExperimentSpec(
      name='validate', env=env, episodes=1000, seeds=3,
      agents={
        'qlearning': AgentSpec('qlearning'),
        'astar': AgentSpec('planner', planner='astar'),
        'rtdp': AgentSpec('planner', planner='rtdp'),
        'lrtdp': AgentSpec('planner', planner='lrtdp'),
        'pcf_astar': AgentSpec('pcf', planner='astar', config=cfg),
        'pcf_rtdp': AgentSpec('pcf', planner='rtdp', config=cfg),
      })

  if name == 'labelling':
    # L1-L5: swap only the learnt-state rule, hold everything else fixed
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    base = paper_pcf_config(q0=-300., anneal=300)
    rules = {
      'L1_jsd': ('jsd', JsdConfig()),
      'L2_argmax_k10': ('argmax', 10, True),
      'L3_residual_e1_k10': ('residual', 1.0, 10, True),
      'L5_counts_n20': ('counts', 20),
    }
    agents = {name_: AgentSpec('pcf', planner='astar',
                               config=replace(base, label_rule=rule))
              for name_, rule in rules.items()}
    agents['astar'] = AgentSpec('planner', planner='astar')
    return ExperimentSpec(name='labelling', env=env, episodes=1000, seeds=5,
                          agents=agents)

  if name == 'ablations':
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    base = paper_pcf_config(q0=-300., anneal=300)
    agents = {
      'base': AgentSpec('pcf', planner='astar', config=base),
      'A1_optimistic_q0': AgentSpec('pcf', planner='astar',
                                    config=replace(base, q0_learner=0.)),
      'A2_td_everywhere': AgentSpec('pcf', planner='astar',
                                    config=replace(base, mc_until_learnt=False)),
      'A3_no_horizon_reduction': AgentSpec('pcf', planner='astar',
                                           config=replace(base, reduce_horizon=False)),
      'A4_no_latch': AgentSpec('pcf', planner='astar',
                               config=replace(base, label_rule=('jsd', JsdConfig(latch=False)))),
      'A5_no_explorer': AgentSpec('pcf', planner='astar',
                                  config=replace(base, use_explorer=False)),
      'A5_unbounded_quota': AgentSpec('pcf', planner='astar',
                                      config=replace(base, bound_quota=False)),
      'A6_no_value_ordering': AgentSpec('pcf', planner='astar',
                                        config=replace(base, value_ordered_goals=False)),
      'R1_plan_cache': AgentSpec('pcf', planner='astar',
                                 config=replace(base, use_plan_cache=True)),
    }
    return ExperimentSpec(name='ablations', env=env, episodes=1000, seeds=5,
                          agents=agents)

  if name == 'paper':
    env = EnvSpec(size=(50, 50), max_steps=1_000_000)
    cfg = paper_pcf_config(q0=-1250., anneal=800)
    return ExperimentSpec(
      name='paper', env=env, episodes=10_000, seeds=10,
      agents={
        'qlearning': AgentSpec('qlearning'),
        'astar': AgentSpec('planner', planner='astar'),
        'rtdp': AgentSpec('planner', planner='rtdp'),
        'lrtdp': AgentSpec('planner', planner='lrtdp'),
        'pcf_astar': AgentSpec('pcf', planner='astar', config=cfg),
        'pcf_rtdp': AgentSpec('pcf', planner='rtdp', config=cfg),
      })

  raise SystemExit(f'unknown preset {name!r}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--preset', required=True,
                  choices=['smoke', 'validate', 'labelling', 'ablations', 'paper'])
  ap.add_argument('--episodes', type=int, default=None)
  ap.add_argument('--seeds', type=int, default=None)
  ap.add_argument('--workers', type=int, default=None)
  ap.add_argument('--out', default='results')
  args = ap.parse_args()

  spec = build_preset(args.preset)
  if args.episodes:
    spec.episodes = args.episodes
  if args.seeds:
    spec.seeds = args.seeds
  spec.out_dir = args.out

  run_experiment(spec, workers=args.workers)


if __name__ == '__main__':
  main()
