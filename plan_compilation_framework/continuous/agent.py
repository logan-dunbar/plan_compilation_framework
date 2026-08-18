"""The Plan Compilation Framework over a continuous state space.

Identical control flow to agents/plan_compiler.py: planner while the state is
not learnt, learner once it is, bounded exploration from an optimistically
initialised explorer, Monte-Carlo updates until learnt and TD between learnt
states. Two things change. The planner works over cells and answers "which
action from this cell"; its surrogate-goal test asks the learner at the
centre of each cell. And the value function is a pluggable representation
(continuous/learners.py), which is the variable under study.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from plan_compilation_framework.continuous.env import ContinuousMaze
from plan_compilation_framework.continuous.learners import LabelConfig, Learner, make_learner
from plan_compilation_framework.core import epsilon_greedy_probabilities
from plan_compilation_framework.planners.astar import AStar

FOLLOW_CAP = 200


@dataclass
class ContinuousPCFConfig:
  learner: str = 'tabular'
  learner_kwargs: dict = field(default_factory=dict)
  q0: float = -300.
  label: LabelConfig = field(default_factory=LabelConfig)
  eps_start: float = 0.1
  eps_end: float = 0.01
  eps_anneal: int = 500
  explore_prob: float = 0.1
  explore_prob_end: float = 0.001
  quota: float = 0.2
  quota_end: float = 0.05
  anneal: int = 300
  alpha_explorer: float = 0.1
  gamma: float = 1.0
  reduce_horizon: bool = True
  value_ordered_goals: bool = True


class ContinuousPlanCompiler:
  def __init__(self, env: ContinuousMaze, cfg: ContinuousPCFConfig, seed: int = 0):
    self.env, self.cfg = env, cfg
    self.rng = np.random.default_rng(seed)
    size = env.grid.shape[0]
    self.learner: Learner = make_learner(cfg.learner, cfg.q0, cfg.label, size, seed, **cfg.learner_kwargs)
    self.planner = AStar(env.model(), seed=seed)
    # explorer: optimistic tabular-over-cells value function (fixed across variants:
    # it is not the object under study)
    self.q_exp: dict = {}
    self.episode = 0
    self.plan: dict = {}
    self.plan_goal = None
    self.plan_goal_true = False
    self.plan_follows = 0
    self.quota = 0.
    self.traj: list = []
    self.stats = {}

  # -- helpers -----------------------------------------------------------------
  def _lin(self, a, b, over):
    if self.episode >= over:
      return b
    return a + (b - a) * self.episode / over

  def _eps(self):
    return self._lin(self.cfg.eps_start, self.cfg.eps_end, self.cfg.eps_anneal)

  def _q_exp(self, cell):
    return self.q_exp.setdefault(cell, np.zeros(4))

  def _greedy_eps(self, q, eps):
    return int(self.rng.choice(4, p=epsilon_greedy_probabilities(q, eps)))

  def _learnt_cell(self, cell):
    return self.learner.learnt(self.env.centre(cell))

  # -- episode -----------------------------------------------------------------
  def begin_episode(self, episode):
    self.episode = episode
    self.plan, self.plan_goal, self.plan_valid = {}, None, False
    self.quota = 0.
    self.traj.clear()
    self.stats = dict(expansions=0, plans_computed=0, planner_steps=0, learner_steps=0,
                      explorer_steps=0, plan_time=0.)

  def act(self, s):
    cell = self.env.cell(s)
    if self.quota > 0.:
      self.stats['explorer_steps'] += 1
      return self._greedy_eps(self._q_exp(cell), 0.1)
    if not self.learner.learnt(s):
      self.stats['planner_steps'] += 1
      return self._planner_action(cell)
    if self.rng.random() < self._lin(self.cfg.explore_prob, self.cfg.explore_prob_end, self.cfg.anneal):
      self.quota = self._lin(self.cfg.quota, self.cfg.quota_end, self.cfg.anneal) * abs(self.learner.value(s))
      self.stats['explorer_steps'] += 1
      return self._greedy_eps(self._q_exp(cell), 0.1)
    self.stats['learner_steps'] += 1
    return self._greedy_eps(self.learner.q(s), self._eps())

  def _plan_valid(self):
    if not self.plan_valid or self.plan_follows > FOLLOW_CAP:
      return False
    return self.plan_goal_true or self._learnt_cell(self.plan_goal)

  def _planner_action(self, cell):
    if cell in self.plan and self._plan_valid():
      self.plan_follows += 1
      return self.plan[cell]
    extra = None
    if self.cfg.reduce_horizon:
      start_v = self.learner.value(self.env.centre(cell))
      ordered = self.cfg.value_ordered_goals

      def extra(c):
        if not self._learnt_cell(c):
          return False
        return (not ordered) or self.learner.value(self.env.centre(c)) > start_v
    t0 = time.perf_counter()
    res = self.planner.plan(cell, extra_goal=extra)
    self.stats['plan_time'] += time.perf_counter() - t0
    self.stats['expansions'] += res.expansions
    self.stats['plans_computed'] += 1
    if not res.found:
      self.plan_valid = False
      return int(self.rng.integers(4))
    self.plan = dict(res.plan.actions)
    self.plan_goal = res.plan.path[-1]
    self.plan_goal_true = self.env.model().is_goal(self.plan_goal)
    self.plan_valid, self.plan_follows = True, 1
    return self.plan[cell]

  def observe(self, s, a, r, s2, terminated):
    cfg = self.cfg
    ls, ln = self.learner.learnt(s), self.learner.learnt(s2)
    if ls and ln:
      target = r + (0. if terminated else cfg.gamma * self.learner.value(s2))
      self.learner.observe_update(s, a, target)
    else:
      self.traj.append((s, a, r))
    if terminated or ln:
      g = 0. if terminated else self.learner.value(s2)
      for (ss, aa, rr) in reversed(self.traj):
        g = rr + cfg.gamma * g
        self.learner.observe_update(ss, aa, g)
      self.traj.clear()
    # explorer
    c, c2 = self.env.cell(s), self.env.cell(s2)
    qe = self._q_exp(c)
    qe[a] += cfg.alpha_explorer * (r + (0. if terminated else self._q_exp(c2).max()) - qe[a])
    if self.quota > 0.:
      self.quota -= abs(r)

  def end_episode(self):
    self.learner.end_episode()
    return dict(self.stats)
