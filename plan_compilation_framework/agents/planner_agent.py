"""Plan-and-enact baseline: a planner, closed-loop plan following, no learning.

Follows the plan by state lookup, so stochastic drift off the plan corridor
triggers an honest replan (counted). `use_cache=True` gives the baseline the
same cross-episode plan cache the paper's PC agents had, so the cache's
contribution can be measured on both sides (R1).
"""
from __future__ import annotations

import time

import numpy as np

from plan_compilation_framework.core import Agent, Transition
from plan_compilation_framework.planners.base import Planner


class PlannerAgent(Agent):
  def __init__(self, env, planner: Planner, use_cache: bool = False,
               seed: int | None = None):
    self.env = env
    self.planner = planner
    self.use_cache = use_cache
    self.rng = np.random.default_rng(seed)
    self.n_actions = env.n_actions

    self.episode_plan: dict = {}
    self.plan_cache: dict = {}
    self._stats = {}

  def begin_episode(self, episode: int):
    self.episode_plan.clear()
    self.planner.begin_episode()
    self._stats = dict(expansions=0, cache_hits=0, plans_computed=0,
                       planner_steps=0, learner_steps=0, explorer_steps=0,
                       plan_time=0., n_learnt=0)

  def act(self, s) -> int:
    self._stats['planner_steps'] += 1

    if s in self.episode_plan:
      return self.episode_plan[s]

    if self.use_cache and s in self.plan_cache:
      votes = self.plan_cache[s]
      self._stats['cache_hits'] += 1
      return max(votes, key=votes.get)

    t0 = time.perf_counter()
    result = self.planner.plan(s)
    self._stats['plan_time'] += time.perf_counter() - t0
    self._stats['expansions'] += result.expansions
    self._stats['plans_computed'] += 1

    if not result.found:
      return int(self.rng.integers(self.n_actions))

    self.episode_plan.update(result.plan.actions)
    if self.use_cache:
      for state, action in result.plan.actions.items():
        votes = self.plan_cache.setdefault(state, {})
        votes[action] = votes.get(action, 0) + 1
    return result.plan.actions[s]

  def observe(self, t: Transition):
    pass

  def end_episode(self) -> dict:
    return dict(self._stats)
