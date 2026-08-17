"""Regression tests for the latch-off plan-memo cycle (the 17 Aug 2026 OOM).

Before the fix, PlanCompiler with a non-latching label rule could get two
unlearnt states pointing at each other through the intra-episode plan memo,
looping forever with no planner call and no learning update. These tests run
that configuration with a generous per-episode cap and require every episode
to terminate at the goal, and the pending trajectory to stay bounded.
"""
from dataclasses import replace

import pytest

from plan_compilation_framework.core import Constant, Transition
from plan_compilation_framework.environments import GridWorld
from plan_compilation_framework.agents import PlanCompiler
from plan_compilation_framework.labelling import JsdConfig
from plan_compilation_framework.planners import AStar
from plan_compilation_framework.run_gridworld import paper_pcf_config

STEP_CAP = 20_000  # a 20x20 episode legitimately takes tens to hundreds of steps


def run_episodes(agent, env, episodes):
  worst_steps, worst_traj = 0, 0
  for ep in range(episodes):
    s = env.reset()
    agent.begin_episode(ep)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated):
      a = agent.act(s)
      s2, r, terminated, truncated = env.step(a)
      agent.observe(Transition(s, a, r, s2, terminated, truncated))
      s = s2
      steps += 1
      worst_traj = max(worst_traj, len(agent.traj))
    worst_steps = max(worst_steps, steps)
    assert terminated, f'episode {ep} truncated at {steps} steps: cycle?'
    agent.end_episode()
  return worst_steps, worst_traj


@pytest.mark.parametrize('seed', [0, 1, 2, 3])
def test_no_latch_terminates_every_episode(seed):
  env = GridWorld.random(size=(20, 20), seed=seed, max_steps=STEP_CAP)
  cfg = replace(paper_pcf_config(q0=-300., anneal=300),
                label_rule=('jsd', JsdConfig(tau_l=Constant(0.95), latch=False)))
  agent = PlanCompiler(env, AStar(env.model(), seed=seed), cfg, seed=seed)
  worst_steps, worst_traj = run_episodes(agent, env, episodes=600)
  assert worst_steps < STEP_CAP
  assert worst_traj < STEP_CAP


@pytest.mark.parametrize('seed', [0, 1])
def test_no_latch_with_cross_episode_cache(seed):
  env = GridWorld.random(size=(20, 20), seed=seed, max_steps=STEP_CAP)
  cfg = replace(paper_pcf_config(q0=-300., anneal=300),
                label_rule=('jsd', JsdConfig(latch=False)), use_plan_cache=True)
  agent = PlanCompiler(env, AStar(env.model(), seed=seed), cfg, seed=seed)
  worst_steps, _ = run_episodes(agent, env, episodes=600)
  assert worst_steps < STEP_CAP


def test_latched_base_still_amortises():
  """The fix must not break the headline behaviour: expansions go to zero."""
  env = GridWorld.random(size=(12, 12), seed=0, max_steps=STEP_CAP)
  agent = PlanCompiler(env, AStar(env.model(), seed=0),
                       paper_pcf_config(q0=-300., anneal=100), seed=0)
  run_episodes(agent, env, episodes=250)
  tail = []
  for ep in range(250, 300):
    s = env.reset()
    agent.begin_episode(ep)
    terminated = truncated = False
    while not (terminated or truncated):
      a = agent.act(s)
      s2, r, terminated, truncated = env.step(a)
      agent.observe(Transition(s, a, r, s2, terminated, truncated))
      s = s2
    tail.append(agent.end_episode()['expansions'])
  assert sum(tail) / len(tail) < 1.0
