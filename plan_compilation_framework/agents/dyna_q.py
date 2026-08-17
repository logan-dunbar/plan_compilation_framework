"""Dyna-Q (Sutton 1990) given the same planning model PCF uses.

Q-learning on real experience plus `n_planning` simulated one-step backups
per real step, drawn from previously visited (state, action) pairs and
answered by the *given* model — deterministic, and with the model's reward
(unit cost by default). Because the model does not know about quicksand or
stochasticity, the simulated backups pull Q towards the model's optimum;
this is the "DYNA needs an accurate reward model" point made in the thesis'
related work, now as an empirical comparison.

Search-work accounting: each simulated backup generates one successor, i.e.
1/|A| of a full state expansion; `expansions` reports planning_steps/|A| so
the number is comparable with the search planners.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import (
  Agent, Constant, Schedule, Transition, epsilon_greedy_probabilities,
)


@dataclass
class DynaQConfig:
  gamma: float = 1.0
  alpha: Schedule = Constant(0.1)
  eps: Schedule = Constant(0.1)
  q0: float = 0.
  n_planning: int = 10


class DynaQ(Agent):
  def __init__(self, env, config: DynaQConfig | None = None,
               model_constant_cost: bool = True, seed: int | None = None):
    self.env = env
    self.config = config or DynaQConfig()
    self.rng = np.random.default_rng(seed)
    self.n_actions = env.n_actions
    self.model = env.model(constant_cost=model_constant_cost)

    self.q: dict = {}
    self.visited: list = []       # (s, a) pairs seen, for planning draws
    self._seen: set = set()
    self.episode = 0
    self._stats = {}

  def _q(self, s):
    if s not in self.q:
      self.q[s] = np.full(self.n_actions, self.config.q0)
    return self.q[s]

  def begin_episode(self, episode: int):
    self.episode = episode
    self._stats = dict(expansions=0., cache_hits=0, plans_computed=0,
                       planner_steps=0, learner_steps=0, explorer_steps=0,
                       plan_time=0., n_learnt=0)

  def act(self, s) -> int:
    self._stats['learner_steps'] += 1
    probs = epsilon_greedy_probabilities(self._q(s), self.config.eps(self.episode))
    return int(self.rng.choice(self.n_actions, p=probs))

  def _backup(self, s, a, r, s_next, terminal):
    q = self._q(s)
    q_next = 0. if terminal else float(self._q(s_next).max())
    q[a] += self.config.alpha(self.episode) * (r + self.config.gamma * q_next - q[a])

  def observe(self, t: Transition):
    self._backup(t.s, t.a, t.r, t.s_next, t.terminated)
    if (t.s, t.a) not in self._seen:
      self._seen.add((t.s, t.a))
      self.visited.append((t.s, t.a))

    n = self.config.n_planning
    if n and self.visited:
      idx = self.rng.integers(len(self.visited), size=n)
      for i in idx:
        s, a = self.visited[i]
        s_next = self.model.next_state(s, a)
        self._backup(s, a, self.model.reward(s, a, s_next), s_next, self.model.is_goal(s_next))
      self._stats['expansions'] += n / self.n_actions

  def end_episode(self) -> dict:
    return dict(self._stats)
