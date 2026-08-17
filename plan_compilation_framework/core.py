"""Core abstractions: transitions, agents, schedules.

Every hyperparameter that varies over training is a Schedule: a pure function
of the episode index. This replaces the mutable Schedule.update() objects of
the 2022 codebase so that configs are picklable, loggable and reproducible.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

State = Any  # any hashable state


@dataclass(frozen=True)
class Transition:
  s: State
  a: int
  r: float
  s_next: State
  terminated: bool  # reached a true terminal state (the goal)
  truncated: bool   # hit the step limit


class Agent:
  """Base agent interface used by the experiment runner."""

  def begin_episode(self, episode: int):
    pass

  def act(self, s: State) -> int:
    raise NotImplementedError

  def observe(self, t: Transition):
    pass

  def end_episode(self) -> dict:
    """Return per-episode diagnostics to merge into the episode record."""
    return {}


# ------------------------------------------------------------------ schedules

class Schedule:
  def __call__(self, episode: int) -> float:
    raise NotImplementedError


@dataclass(frozen=True)
class Constant(Schedule):
  value: float

  def __call__(self, episode: int) -> float:
    return self.value


@dataclass(frozen=True)
class Linear(Schedule):
  """Linear interpolation from start to stop over `steps` episodes, then flat."""
  start: float
  stop: float
  steps: int

  def __call__(self, episode: int) -> float:
    if episode >= self.steps:
      return self.stop
    frac = episode / self.steps
    return self.start + (self.stop - self.start) * frac


@dataclass(frozen=True)
class Harmonic(Schedule):
  """start / (episode + 1), floored at stop.

  Matches HarmonicSchedule from the 2022 code, whose counter advanced once
  per episode.
  """
  start: float
  stop: float = 0.

  def __call__(self, episode: int) -> float:
    return max(self.start / (episode + 1), self.stop)


def epsilon_greedy_probabilities(q_values, eps: float):
  """Probability vector of the epsilon-greedy policy over q_values.

  Ties on the maximum share the greedy mass, as in the 2022 implementation.
  """
  import numpy as np
  q_values = np.asarray(q_values, dtype=float)
  i_max = q_values == q_values.max()
  probs = np.full(q_values.shape, eps / q_values.size)
  probs[i_max] += (1. - eps) / np.count_nonzero(i_max)
  return probs
