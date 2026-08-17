"""Plan-based potential shaping (Grzes & Kudenko 2008) on Q-learning.

At the start of each episode the planner produces one plan from the initial
state to the goal. The plan defines a potential over its states,
Phi(s_i) = i (position along the plan, higher nearer the goal); the agent's
current potential is that of the last plan state it visited (0 before it
reaches any), so leaving the plan neither rewards nor punishes until it
rejoins. Q-learning then receives the shaped reward
r + gamma * Phi(s') - Phi(s) (Ng et al. 1999). The plan is advice, not
control: the agent still chooses epsilon-greedy actions from Q, so it can
drift into pure Q-learning behaviour when shaping and environment disagree
— the point made in the thesis' related work.

Search-work accounting: one A* per episode (from the initial state), counted
honestly; unlike PCF the cost never shrinks.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import (
  Agent, Constant, Schedule, Transition, epsilon_greedy_probabilities,
)
from plan_compilation_framework.planners.base import Planner


@dataclass
class ShapedQConfig:
  gamma: float = 1.0
  alpha: Schedule = Constant(0.1)
  eps: Schedule = Constant(0.1)
  q0: float = 0.
  scale: float = 1.0        # potential per plan step


class ShapedQLearning(Agent):
  def __init__(self, env, planner: Planner, config: ShapedQConfig | None = None,
               seed: int | None = None):
    self.env = env
    self.planner = planner
    self.config = config or ShapedQConfig()
    self.rng = np.random.default_rng(seed)
    self.n_actions = env.n_actions

    self.q: dict = {}
    self.potential: dict = {}
    self.current_phi = 0.
    self.episode = 0
    self._stats = {}

  def _q(self, s):
    if s not in self.q:
      self.q[s] = np.full(self.n_actions, self.config.q0)
    return self.q[s]

  def begin_episode(self, episode: int):
    self.episode = episode
    self.potential = {}
    self.current_phi = 0.
    self._planned = False
    self._stats = dict(expansions=0, cache_hits=0, plans_computed=0,
                       planner_steps=0, learner_steps=0, explorer_steps=0,
                       plan_time=0., n_learnt=0)

  def _plan_from(self, s):
    t0 = time.perf_counter()
    result = self.planner.plan(s)
    self._stats['plan_time'] += time.perf_counter() - t0
    self._stats['expansions'] += result.expansions
    self._stats['plans_computed'] += 1
    if result.found:
      path = result.plan.path
      self.potential = {st: self.config.scale * i for i, st in enumerate(path)}
    self._planned = True

  def _phi(self, s):
    if s in self.potential:
      self.current_phi = self.potential[s]
    return self.current_phi

  def act(self, s) -> int:
    if not self._planned:
      self._plan_from(s)
    self._stats['learner_steps'] += 1
    probs = epsilon_greedy_probabilities(self._q(s), self.config.eps(self.episode))
    return int(self.rng.choice(self.n_actions, p=probs))

  def observe(self, t: Transition):
    cfg = self.config
    phi_s = self._phi(t.s)
    phi_next = 0. if t.terminated else self._phi(t.s_next)
    shaped = t.r + cfg.gamma * phi_next - phi_s
    q = self._q(t.s)
    q_next = 0. if t.terminated else float(self._q(t.s_next).max())
    q[t.a] += cfg.alpha(self.episode) * (shaped + cfg.gamma * q_next - q[t.a])

  def end_episode(self) -> dict:
    return dict(self._stats)
