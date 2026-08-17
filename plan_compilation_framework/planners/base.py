"""Planner interfaces and the uniform expansion-accounting contract.

One *expansion* = generating the successors of one state, for every planner.
The 2022 code counted A* by nodes popped but RTDP by per-action model steps,
inflating RTDP by a factor of |A|; the resubmission counts state expansions
uniformly so the "states expanded per episode" figures are comparable across
planners.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from plan_compilation_framework.core import State


@dataclass
class Plan:
  """Closed-loop plan: a state->action mapping along the found path."""
  actions: dict = field(default_factory=dict)
  path: list = field(default_factory=list)
  cost: float = 0.

  def __contains__(self, s):
    return s in self.actions

  def __getitem__(self, s):
    return self.actions[s]

  def __len__(self):
    return len(self.actions)


@dataclass
class SearchResult:
  plan: Plan | None
  expansions: int

  @property
  def found(self):
    return self.plan is not None


class Planner:
  """A planner owns a model and answers plan(start) queries.

  `extra_goal` is an optional predicate: states satisfying it are treated as
  goals in addition to the model's true goal. This is the hook the Plan
  Compilation Framework uses to shorten the planning horizon.
  """

  def __init__(self, model, seed: int | None = None):
    import numpy as np
    self.model = model
    self.rng = np.random.default_rng(seed)

  def plan(self, start: State, extra_goal=None) -> SearchResult:
    raise NotImplementedError

  def begin_episode(self):
    """Hook for per-episode bookkeeping (persistent planners keep their state)."""
