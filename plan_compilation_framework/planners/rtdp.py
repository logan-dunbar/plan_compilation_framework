"""RTDP (Barto, Bradtke & Singh 1995) as an online planner.

Faithful port of the 2022 planner: each plan() call runs a single greedy
trial from the start state, performing Bellman backups on a value table that
persists across calls and episodes. The returned plan is the trial's
trajectory. Values are in reward space (all costs negative), initialised to 0
which upper-bounds the optimal values, so greedy trials remain directed.

Expansions are counted per state expanded (one successor generation), not per
action as in the 2022 code.
"""
from __future__ import annotations

import numpy as np

from plan_compilation_framework.planners.base import Plan, Planner, SearchResult

TRIAL_CAP = 200_000  # safety net against pathological greedy cycles


class RTDP(Planner):
  def __init__(self, model, seed: int | None = None):
    super().__init__(model, seed)
    self.values: dict = {}

  def _q_values(self, s):
    model = self.model
    q = np.empty(model.n_actions)
    succs = []
    for a in model.actions(s):
      s_next = model.next_state(s, a)
      q[a] = model.reward(s, a, s_next) + (0. if model.is_goal(s_next)
                                           else self.values.get(s_next, 0.))
      succs.append(s_next)
    return q, succs

  def _greedy(self, q):
    max_actions = np.flatnonzero(q == q.max())
    return int(self.rng.choice(max_actions))

  def plan(self, start, extra_goal=None) -> SearchResult:
    model = self.model
    expansions = 0
    actions, path = {}, [start]
    curr = start

    for _ in range(TRIAL_CAP):
      if model.is_goal(curr) or (extra_goal is not None and extra_goal(curr)):
        return SearchResult(Plan(actions=actions, path=path, cost=float(len(actions))),
                            expansions)
      q, succs = self._q_values(curr)
      expansions += 1
      self.values[curr] = float(q.max())
      a = self._greedy(q)
      actions[curr] = a
      curr = succs[a]
      path.append(curr)

    return SearchResult(None, expansions)
