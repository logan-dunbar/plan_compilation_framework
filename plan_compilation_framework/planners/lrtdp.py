"""Labelled RTDP (Bonet & Geffner 2003) — the examiner-named baseline.

RTDP plus the CheckSolved labelling procedure: a state is labelled *solved*
when the Bellman residuals over its greedy envelope are below epsilon, and
trials stop at solved states. Each plan() call runs trials from the start
until the start is solved (or the trial budget is exhausted), then returns
the greedy plan. Values and solved labels persist across calls and episodes,
which is LRTDP's own amortisation over repeated queries — the model-grounded
mirror of PCF's learnt set.

Deterministic-model specialisation: the greedy envelope of a state is the
single greedy successor chain.
"""
from __future__ import annotations

import numpy as np

from plan_compilation_framework.planners.base import Plan, Planner, SearchResult

TRIAL_CAP = 200_000


class LRTDP(Planner):
  def __init__(self, model, epsilon: float = 1e-3, max_trials_per_call: int = 1000,
               seed: int | None = None):
    super().__init__(model, seed)
    self.epsilon = epsilon
    self.max_trials_per_call = max_trials_per_call
    self.values: dict = {}
    self.solved: set = set()
    self._expansions = 0

  # -- value helpers ---------------------------------------------------------
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

  def _residual_backup(self, s):
    """Backup s; return (residual, greedy successor)."""
    q, succs = self._q_values(s)
    self._expansions += 1
    best = float(q.max())
    residual = abs(best - self.values.get(s, 0.))
    self.values[s] = best
    a = int(self.rng.choice(np.flatnonzero(q == q.max())))
    return residual, succs[a], a

  def _is_terminal(self, s, extra_goal):
    return self.model.is_goal(s) or (extra_goal is not None and extra_goal(s))

  # -- CheckSolved (Bonet & Geffner 2003, deterministic case) ----------------
  def _check_solved(self, s, extra_goal):
    rv = True
    open_stack, closed = [], []
    seen = set()
    if s not in self.solved:
      open_stack.append(s)
      seen.add(s)

    while open_stack:
      node = open_stack.pop()
      closed.append(node)

      q, succs = self._q_values(node)
      self._expansions += 1
      residual = abs(float(q.max()) - self.values.get(node, 0.))
      if residual > self.epsilon:
        rv = False
        continue

      greedy_next = succs[int(np.argmax(q))]
      if (not self._is_terminal(greedy_next, extra_goal)
          and greedy_next not in self.solved and greedy_next not in seen):
        open_stack.append(greedy_next)
        seen.add(greedy_next)

    if rv:
      self.solved.update(closed)
    else:
      for node in reversed(closed):
        self._residual_backup(node)
    return rv

  # -- trials ----------------------------------------------------------------
  def _trial(self, start, extra_goal):
    visited = []
    curr = start
    for _ in range(TRIAL_CAP):
      if curr in self.solved or self._is_terminal(curr, extra_goal):
        break
      visited.append(curr)
      _, greedy_next, _ = self._residual_backup(curr)
      curr = greedy_next

    while visited:
      s = visited.pop()
      if not self._check_solved(s, extra_goal):
        break

  def plan(self, start, extra_goal=None) -> SearchResult:
    self._expansions = 0

    trials = 0
    while start not in self.solved and not self._is_terminal(start, extra_goal):
      if trials >= self.max_trials_per_call:
        break
      self._trial(start, extra_goal)
      trials += 1

    # extract the greedy plan
    actions, path = {}, [start]
    curr = start
    for _ in range(TRIAL_CAP):
      if self._is_terminal(curr, extra_goal):
        return SearchResult(Plan(actions=actions, path=path, cost=float(len(actions))),
                            self._expansions)
      q, succs = self._q_values(curr)
      a = int(np.argmax(q))
      actions[curr] = a
      curr = succs[a]
      path.append(curr)

    return SearchResult(None, self._expansions)
