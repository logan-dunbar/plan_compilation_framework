"""Tabular Q-learning baseline (the model-free column of the comparison)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import (
  Agent, Constant, Schedule, Transition, epsilon_greedy_probabilities,
)


@dataclass
class QLearningConfig:
  gamma: float = 1.0
  alpha: Schedule = Constant(0.1)
  eps: Schedule = Constant(0.1)
  q0: float = 0.


class QLearning(Agent):
  def __init__(self, env, config: QLearningConfig | None = None,
               seed: int | None = None):
    self.env = env
    self.config = config or QLearningConfig()
    self.rng = np.random.default_rng(seed)
    self.n_actions = env.n_actions
    self.q: dict = {}
    self.episode = 0

  def _q(self, s):
    if s not in self.q:
      self.q[s] = np.full(self.n_actions, self.config.q0)
    return self.q[s]

  def begin_episode(self, episode: int):
    self.episode = episode

  def act(self, s) -> int:
    probs = epsilon_greedy_probabilities(self._q(s), self.config.eps(self.episode))
    return int(self.rng.choice(self.n_actions, p=probs))

  def observe(self, t: Transition):
    cfg = self.config
    q = self._q(t.s)
    q_next = 0. if t.terminated else float(self._q(t.s_next).max())
    target = t.r + cfg.gamma * q_next
    q[t.a] += cfg.alpha(self.episode) * (target - q[t.a])
