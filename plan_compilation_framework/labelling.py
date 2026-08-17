"""Pluggable learnt-state labelling rules — the L1-L5 study.

The examiners asked for the solved-labelling contribution to be compared
against alternatives (counts, residual value changes, LRTDP-style
epsilon-convergence). Every rule implements the same interface so the Plan
Compilation Framework can swap them with everything else held fixed:

  record_update(s, q_before, q_after, delta, eps) -- called after each
      Q-value update to state s. `delta` is the applied update on the updated
      action; `eps` is the learner's current epsilon (for policy
      distributions).
  is_learnt(s) -> bool
  count() -> number of currently learnt states

Rules:
  L1 jsd       -- policy-stability via Jensen-Shannon distance between the
                  epsilon-greedy policies before/after the update, tracked
                  with an exponential recency-weighted average (the thesis
                  method).
  L2 argmax    -- greedy action unchanged for k consecutive updates.
  L3 residual  -- |update delta| below epsilon for k consecutive updates
                  (LRTDP-style epsilon-convergence, measured on the learner).
  L5 counts    -- state updated at least n0 times (E^3-style known states).

All rules support the latch (a learnt state never reverts), which is itself
ablation A4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import Schedule, Constant, epsilon_greedy_probabilities


def _kl(p, q):
  p = np.where(p == 0., 1e-50, p)
  q = np.where(q == 0., 1e-50, q)
  return max(float(np.sum(p * np.log(p / q))), 0.)


def js_distance(p, q):
  """Jensen-Shannon *distance* (the 2022 code took the square root)."""
  m = 0.5 * (p + q)
  return float(np.sqrt(0.5 * _kl(p, m) + 0.5 * _kl(q, m)))


class LabelRule:
  latch = True

  def begin_episode(self, episode: int):
    self._episode = episode

  def record_update(self, s, q_before, q_after, delta, eps):
    raise NotImplementedError

  def is_learnt(self, s) -> bool:
    raise NotImplementedError

  def count(self) -> int:
    raise NotImplementedError


@dataclass
class JsdConfig:
  tau_d: Schedule = Constant(0.1)     # divergence threshold
  tau_l: Schedule = Constant(0.85)    # learnt threshold on the l-value
  alpha_l: Schedule = Constant(0.4)   # l-value learning rate
  latch: bool = True


class JsdStability(LabelRule):
  """L1: the thesis rule."""

  def __init__(self, config: JsdConfig | None = None):
    self.config = config or JsdConfig()
    self.latch = self.config.latch
    self.l: dict = {}
    self._learnt: set = set()
    self._episode = 0

  def record_update(self, s, q_before, q_after, delta, eps):
    cfg = self.config
    if self.latch and s in self._learnt:
      target = 1.
    else:
      p_pre = epsilon_greedy_probabilities(q_before, eps)
      p_post = epsilon_greedy_probabilities(q_after, eps)
      target = 1. if js_distance(p_pre, p_post) < cfg.tau_d(self._episode) else 0.

    l_val = self.l.get(s, 0.)
    l_val += cfg.alpha_l(self._episode) * (target - l_val)
    self.l[s] = l_val

    if l_val > cfg.tau_l(self._episode):
      self._learnt.add(s)
    elif not self.latch:
      self._learnt.discard(s)

  def is_learnt(self, s):
    return s in self._learnt

  def count(self):
    return len(self._learnt)


class ArgmaxStability(LabelRule):
  """L2: greedy action unchanged for k consecutive updates."""

  def __init__(self, k: int = 10, latch: bool = True):
    self.k = k
    self.latch = latch
    self.streak: dict = {}
    self._learnt: set = set()

  def record_update(self, s, q_before, q_after, delta, eps):
    if self.latch and s in self._learnt:
      return
    stable = int(np.argmax(q_before)) == int(np.argmax(q_after))
    streak = self.streak.get(s, 0) + 1 if stable else 0
    self.streak[s] = streak
    if streak >= self.k:
      self._learnt.add(s)
    elif not self.latch:
      self._learnt.discard(s)

  def is_learnt(self, s):
    return s in self._learnt

  def count(self):
    return len(self._learnt)


class ResidualStability(LabelRule):
  """L3: |applied update| < epsilon for k consecutive updates."""

  def __init__(self, epsilon: float = 1.0, k: int = 10, latch: bool = True):
    self.epsilon = epsilon
    self.k = k
    self.latch = latch
    self.streak: dict = {}
    self._learnt: set = set()

  def record_update(self, s, q_before, q_after, delta, eps):
    if self.latch and s in self._learnt:
      return
    streak = self.streak.get(s, 0) + 1 if abs(delta) < self.epsilon else 0
    self.streak[s] = streak
    if streak >= self.k:
      self._learnt.add(s)
    elif not self.latch:
      self._learnt.discard(s)

  def is_learnt(self, s):
    return s in self._learnt

  def count(self):
    return len(self._learnt)


class VisitCount(LabelRule):
  """L5: learnt after n0 updates (E^3-style known states)."""

  def __init__(self, n0: int = 20):
    self.n0 = n0
    self.counts: dict = {}
    self._learnt: set = set()

  def record_update(self, s, q_before, q_after, delta, eps):
    n = self.counts.get(s, 0) + 1
    self.counts[s] = n
    if n >= self.n0:
      self._learnt.add(s)

  def is_learnt(self, s):
    return s in self._learnt

  def count(self):
    return len(self._learnt)


def make_label_rule(spec) -> LabelRule:
  """spec: ('jsd', JsdConfig) | ('argmax', k, latch) | ('residual', eps, k, latch) | ('counts', n0)"""
  kind, *args = spec
  if kind == 'jsd':
    return JsdStability(*args)
  if kind == 'argmax':
    return ArgmaxStability(*args)
  if kind == 'residual':
    return ResidualStability(*args)
  if kind == 'counts':
    return VisitCount(*args)
  raise ValueError(f'unknown labelling rule {kind!r}')
