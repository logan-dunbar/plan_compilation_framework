"""Continuous-state maze with discrete actions (the Chapter 4 test bed).

Same maze layout and reward structure as the GridWorld of Chapter 3, but the
agent's state is a real-valued position and every move is a fixed-length
step in one of four directions plus Gaussian noise. Keeping the actions
discrete keeps the argmax exact, so any failure of the learnt-state
mechanism is due to how the *state* is represented -- which is the property
the chapter examines. The planner sees the same deterministic unit-cost
grid model as before, over cells.
"""
from __future__ import annotations

import numpy as np

from plan_compilation_framework.environments.gridworld import (
  ACTIONS, N_ACTIONS, Grid, GridModel, generate_grid,
)


class ContinuousMaze:
  n_actions = N_ACTIONS

  def __init__(self, grid: Grid, step_len=0.8, noise=0.2, move_reward=-1., bump_reward=-5.,
               sand_reward=-100., max_steps=20_000, seed=None):
    self.grid = grid
    self.step_len = step_len
    self.noise = noise
    self.move_reward, self.bump_reward, self.sand_reward = move_reward, bump_reward, sand_reward
    self.max_steps = max_steps
    self.rng = np.random.default_rng(seed)
    self.pos = np.zeros(2)
    self._steps = 0
    self._starts = np.array(grid.start_states, dtype=float)

  @classmethod
  def random(cls, size=(20, 20), seed=None, **kw):
    rng = np.random.default_rng(seed)
    grid = generate_grid(size, rng=rng)
    return cls(grid, seed=None if seed is None else seed + 1, **kw)

  # -- geometry ------------------------------------------------------------
  @staticmethod
  def cell(pos):
    return (int(np.floor(pos[0])), int(np.floor(pos[1])))

  @staticmethod
  def centre(cell):
    return np.array([cell[0] + 0.5, cell[1] + 0.5])

  def in_wall(self, pos):
    r, c = self.cell(pos)
    rows, cols = self.grid.shape
    return not (0 <= r < rows and 0 <= c < cols) or (r, c) in self.grid.walls

  # -- episodes ------------------------------------------------------------
  def reset(self):
    self._steps = 0
    cell = self._starts[self.rng.integers(len(self._starts))]
    self.pos = cell + self.rng.uniform(0.05, 0.95, size=2)
    return self.pos.copy()

  def step(self, action: int):
    dr, dc = ACTIONS[action]
    delta = np.array([dr, dc], dtype=float) * self.step_len + self.rng.normal(0., self.noise, size=2)
    proposed = self.pos + delta
    bumped = self.in_wall(proposed)
    new = self.pos if bumped else proposed
    reward = self.move_reward + (self.bump_reward if bumped else 0.)
    c = self.cell(new)
    if c in self.grid.sand:
      reward += self.sand_reward
    terminated = c == self.grid.goal
    self.pos = new
    self._steps += 1
    truncated = not terminated and self._steps >= self.max_steps
    return new.copy(), reward, terminated, truncated

  def model(self):
    return GridModel(self.grid, constant_cost=True)

  def probe_points(self):
    """Centres of all free cells: the fixed lattice on which labels are audited."""
    return np.array([self.centre(c) for c in self.grid.free_states()])
