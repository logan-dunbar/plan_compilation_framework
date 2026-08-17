"""Learned-heuristic search: a value function used *inside* search (the
AlphaZero-shaped baseline the examiners asked for), in two groundings.

The agent plans with A* every time it needs an action, using
h(s) = c_hat(s), a learned cost-to-go, as the heuristic. As c_hat tightens the
search expands fewer states — but search never leaves the loop, which is the
structural difference from PCF. Two variants differ only in what c_hat is
regressed onto:

  grounding='model'  targets are the *model's* remaining plan cost from each
                     state on the found plan (the search's own outcome under
                     the model, as AlphaZero learns from self-play outcomes
                     under an exact game model). Value stays tied to the
                     model: with a wrong model, reward stays at model level.
  grounding='env'    targets are the *realised* environment cost-to-go from
                     each visited state (Monte-Carlo, like PCF's learner).
                     Value reflects the true costs, but is only used to order
                     search over the model's unit-cost graph.

Both use the per-(state) visit-count harmonic learning rate of PCF's learner.
Expansions are honest A* expansions per plan call; the plan is followed by
state lookup with a replan on drift (same policy as PlannerAgent).
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import Agent, Harmonic, Schedule, Transition
from plan_compilation_framework.planners.astar import AStar


@dataclass
class LearnedHeuristicConfig:
  grounding: str = 'model'            # 'model' | 'env'
  alpha: Schedule = Harmonic(1.0, 0.1)
  weight: float = 1.0                 # h = weight * c_hat  (weight > 1: weighted A*)


class _LearnedHeuristicAStar(AStar):
  """A* whose heuristic reads a mutable cost-to-go table."""

  def __init__(self, model, table: dict, weight: float, seed=None):
    super().__init__(model, heuristic='zero', seed=seed)
    self.table = table
    self.weight = weight

  def _h(self, s):
    return self.weight * self.table.get(s, 0.)


class LearnedHeuristic(Agent):
  def __init__(self, env, config: LearnedHeuristicConfig | None = None,
               model_constant_cost: bool = True, seed: int | None = None):
    self.env = env
    self.config = config or LearnedHeuristicConfig()
    assert self.config.grounding in ('model', 'env')
    self.rng = np.random.default_rng(seed)
    self.n_actions = env.n_actions

    self.model = env.model(constant_cost=model_constant_cost)
    self.c_hat: dict = {}   # state -> estimated cost-to-go (>= 0)
    self.n_s: dict = {}
    self.planner = _LearnedHeuristicAStar(self.model, self.c_hat, self.config.weight, seed=seed)

    self.plan: dict = {}
    self.traj: list[Transition] = []
    self._stats = {}

  # -- learning -------------------------------------------------------------
  def _update(self, s, target):
    n = self.n_s.get(s, 0)
    self.n_s[s] = n + 1
    a = self.config.alpha(n)
    self.c_hat[s] = self.c_hat.get(s, 0.) + a * (max(target, 0.) - self.c_hat.get(s, 0.))

  def _learn_from_plan(self, plan):
    """Model grounding: remaining model cost along the plan path."""
    path = plan.path
    remaining = 0.
    for i in range(len(path) - 1, -1, -1):
      s = path[i]
      if i < len(path) - 1:
        a = plan.actions[s]
        remaining += self.model.cost(s, a, path[i + 1])
      self._update(s, remaining)

  # -- lifecycle ------------------------------------------------------------
  def begin_episode(self, episode: int):
    self.plan.clear()
    self.traj.clear()
    self._stats = dict(expansions=0, cache_hits=0, plans_computed=0,
                       planner_steps=0, learner_steps=0, explorer_steps=0,
                       plan_time=0., n_learnt=0)

  def act(self, s) -> int:
    self._stats['planner_steps'] += 1
    if s in self.plan:
      return self.plan[s]

    t0 = time.perf_counter()
    result = self.planner.plan(s)
    self._stats['plan_time'] += time.perf_counter() - t0
    self._stats['expansions'] += result.expansions
    self._stats['plans_computed'] += 1
    if not result.found:
      return int(self.rng.integers(self.n_actions))

    self.plan = dict(result.plan.actions)
    if self.config.grounding == 'model':
      self._learn_from_plan(result.plan)
    return self.plan[s]

  def observe(self, t: Transition):
    if self.config.grounding == 'env':
      self.traj.append(t)
      if t.terminated or t.truncated:
        g = 0.
        for tr in reversed(self.traj):
          g += -tr.r          # cost = negated reward
          self._update(tr.s, g)
        self.traj.clear()

  def end_episode(self) -> dict:
    self._stats['n_learnt'] = len(self.c_hat)
    return dict(self._stats)
