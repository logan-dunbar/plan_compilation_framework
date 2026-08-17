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
from plan_compilation_framework.agents import (
  PCFConfig, QLearningConfig, LearnedHeuristicConfig, DynaQConfig, ShapedQConfig,
)
from plan_compilation_framework.experiment import (
  AgentSpec, EnvSpec, ExperimentSpec, run_experiment,
)
from plan_compilation_framework.labelling import JsdConfig


def eps_schedule(episodes: int):
  """Learner epsilon: 0.1 -> 0.01 over the first half of the run.

  Adopted for the resubmission after the C4 sensitivity sweep (constant 0.1
  was the ECML'22 setting; annealing gave the best tail reward at identical
  search cost). Applied identically to the Q-learning baseline for fairness.
  """
  return Linear(0.1, 0.01, max(1, episodes // 2))


def qlearning_config(episodes: int) -> QLearningConfig:
  return QLearningConfig(gamma=1.0, alpha=Constant(0.1), eps=eps_schedule(episodes), q0=0.)


def paper_pcf_config(q0: float, anneal: int, episodes: int = 10_000) -> PCFConfig:
  """The ECML'22 hyperparameters (from the 2022 experiments.json), with the
  resubmission's annealed learner epsilon.

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
    eps_learner=eps_schedule(episodes),
    eps_explorer=Constant(0.1),
    explore_prob=Linear(0.1, 0.001, anneal),
    quota_frac=Linear(0.2, 0.05, anneal),
    q0_learner=q0,
    q0_explorer=0.,
    label_rule=('jsd', JsdConfig(tau_d=Constant(0.1), tau_l=Constant(0.85),
                                 alpha_l=Constant(0.4))),
  )


def baseline_agents(episodes: int) -> dict:
  """The comparison set: model-free, model-tied planners, value-in-search
  baselines in both groundings, and the model-based / shaping RL baselines."""
  eps = eps_schedule(episodes)
  return {
    'qlearning': AgentSpec('qlearning', config=qlearning_config(episodes)),
    'astar': AgentSpec('planner', planner='astar'),
    'rtdp': AgentSpec('planner', planner='rtdp'),
    'lrtdp': AgentSpec('planner', planner='lrtdp'),
    'lh_model': AgentSpec('learned_heuristic', config=LearnedHeuristicConfig(grounding='model')),
    'lh_env': AgentSpec('learned_heuristic', config=LearnedHeuristicConfig(grounding='env')),
    'dynaq': AgentSpec('dynaq', config=DynaQConfig(eps=eps, n_planning=10)),
    'shaped_q': AgentSpec('shaped_q', planner='astar', config=ShapedQConfig(eps=eps)),
  }


def build_preset(name: str) -> ExperimentSpec:
  if name == 'smoke':
    env = EnvSpec(size=(12, 12), max_steps=50_000)
    cfg = paper_pcf_config(q0=-300., anneal=100, episodes=300)
    return ExperimentSpec(
      name='smoke', env=env, episodes=300, seeds=2,
      agents={
        'qlearning': AgentSpec('qlearning', config=qlearning_config(300)),
        'astar': AgentSpec('planner', planner='astar'),
        'pcf_astar': AgentSpec('pcf', planner='astar', config=cfg),
      })

  if name == 'validate':
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    cfg = paper_pcf_config(q0=-300., anneal=300, episodes=1000)
    return ExperimentSpec(
      name='validate', env=env, episodes=1000, seeds=3,
      agents={
        'qlearning': AgentSpec('qlearning', config=qlearning_config(1000)),
        'astar': AgentSpec('planner', planner='astar'),
        'rtdp': AgentSpec('planner', planner='rtdp'),
        'lrtdp': AgentSpec('planner', planner='lrtdp'),
        'pcf_astar': AgentSpec('pcf', planner='astar', config=cfg),
        'pcf_rtdp': AgentSpec('pcf', planner='rtdp', config=cfg),
      })

  if name == 'labelling':
    # L1-L5: swap only the learnt-state rule, hold everything else fixed
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    base = paper_pcf_config(q0=-300., anneal=300, episodes=1000)
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
    base = paper_pcf_config(q0=-300., anneal=300, episodes=1000)
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

  if name == 'baselines':
    # quick 20x20 check of the full comparison set
    env = EnvSpec(size=(20, 20), max_steps=100_000)
    cfg = paper_pcf_config(q0=-300., anneal=300, episodes=1000)
    agents = baseline_agents(1000)
    agents['pcf_astar'] = AgentSpec('pcf', planner='astar', config=cfg)
    return ExperimentSpec(name='baselines', env=env, episodes=1000, seeds=5, agents=agents)

  if name == 'paper':
    env = EnvSpec(size=(50, 50), max_steps=1_000_000)
    cfg = paper_pcf_config(q0=-1250., anneal=800, episodes=10_000)
    agents = baseline_agents(10_000)
    agents['pcf_astar'] = AgentSpec('pcf', planner='astar', config=cfg)
    agents['pcf_rtdp'] = AgentSpec('pcf', planner='rtdp', config=cfg)
    return ExperimentSpec(name='paper', env=env, episodes=10_000, seeds=10, agents=agents)

  raise SystemExit(f'unknown preset {name!r}')


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--preset', required=True,
                  choices=['smoke', 'validate', 'labelling', 'ablations', 'baselines', 'paper'])
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
