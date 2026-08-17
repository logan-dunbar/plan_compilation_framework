"""Stochastic maze-like GridWorld and its (deliberately wrong) planning model.

Environment semantics follow the ECML'22 experiments exactly:
  - actions are the four cardinal directions;
  - with `stochastic=True` the executed action is the intended one with
    p=0.8, or either neighbouring direction with p=0.1 each (never reversed);
  - moving into a wall leaves the agent in place;
  - rewards are additive: every step earns `move_reward`; bumping a wall adds
    `bump_reward`; landing on quicksand adds `sand_reward`; reaching the goal
    adds `terminal_reward` and terminates. With the defaults a bump is
    -1 + -5 = -6 total and quicksand -1 + -100 = -101 total. (The thesis text
    describes these as -5 and -100; the implementation has always been
    additive, which the resubmission documents explicitly.)

The planning model view (`GridWorld.model()`) is deterministic and, by
default, constant unit cost: it knows the maze connectivity but neither the
transition stochasticity nor the true reward structure. This is the
"inaccurate model" of the thesis. `constant_cost=False` yields a model with
accurate (deterministic-case) costs, giving the model-fidelity axis a second
setting.
"""
from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass, field

import numpy as np

ACTIONS = {
  0: (-1, 0),  # up
  1: (0, 1),   # right
  2: (1, 0),   # down
  3: (0, -1),  # left
}
ACTION_NAMES = {0: 'UP', 1: 'RIGHT', 2: 'DOWN', 3: 'LEFT'}
# executed-action distribution for each intended action (index = executed)
STOCHASTIC_PROBS = {
  0: (0.8, 0.1, 0.0, 0.1),
  1: (0.1, 0.8, 0.1, 0.0),
  2: (0.0, 0.1, 0.8, 0.1),
  3: (0.1, 0.0, 0.1, 0.8),
}
N_ACTIONS = 4


@dataclass
class Grid:
  shape: tuple[int, int]
  walls: frozenset
  sand: frozenset
  goal: tuple[int, int]
  start_states: tuple  # states reachable from the goal (may include sand)
  start: tuple[int, int]

  def free_states(self):
    r, c = self.shape
    return [(i, j) for i in range(r) for j in range(c) if (i, j) not in self.walls]


def _bfs_reachable(shape, walls, source):
  """All non-wall cells connected to `source` under 4-connectivity."""
  seen = {source}
  frontier = deque([source])
  rows, cols = shape
  while frontier:
    r, c = frontier.popleft()
    for dr, dc in ACTIONS.values():
      nb = (r + dr, c + dc)
      if 0 <= nb[0] < rows and 0 <= nb[1] < cols and nb not in walls and nb not in seen:
        seen.add(nb)
        frontier.append(nb)
  return seen


def generate_grid(size=(50, 50), wall_frac=0.2, sand_frac=0.25,
                  min_reachable_frac=0.5, rng: np.random.Generator | None = None) -> Grid:
  """Random maze generator, semantics ported from the 2022 experiment code.

  Walls: `wall_frac` of all cells drawn uniformly with replacement, plus the
  border. Quicksand: `sand_frac` of the remaining free cells, drawn with
  weight (row%2 + col%2 + 1) which favours odd rows/columns and creates
  maze-like corridors. The goal gets five placement attempts, each requiring
  that more than `min_reachable_frac` of all cells reach it; otherwise the
  whole grid is regenerated. Start states are every cell reachable from the
  goal (quicksand included).
  """
  rng = rng or np.random.default_rng()
  rows, cols = size
  total = rows * cols

  while True:
    all_inds = np.arange(total)
    wall_inds = set(rng.choice(all_inds, int(total * wall_frac)).tolist())
    for r in range(rows):
      wall_inds.add(r * cols + 0)
      wall_inds.add(r * cols + (cols - 1))
    for c in range(cols):
      wall_inds.add(0 * cols + c)
      wall_inds.add((rows - 1) * cols + c)

    free_inds = sorted(set(range(total)) - wall_inds)
    free_cells = [(i // cols, i % cols) for i in free_inds]

    weights = np.array([r % 2 + c % 2 + 1 for r, c in free_cells], dtype=float)
    weights /= weights.sum()
    sand_pick = rng.choice(len(free_cells), int(len(free_cells) * sand_frac), p=weights)
    sand = {free_cells[i] for i in sand_pick.tolist()}

    walls = frozenset((i // cols, i % cols) for i in wall_inds)
    placeable = [c for c in free_cells if c not in sand]

    placed = False
    for _ in range(5):
      goal = placeable[rng.integers(len(placeable))]
      reachable = _bfs_reachable(size, walls, goal)
      if len(reachable) > min_reachable_frac * total:
        placed = True
        break

    if placed:
      while (start := placeable[rng.integers(len(placeable))]) == goal:
        pass
      start_states = tuple(sorted(reachable - {goal}))
      return Grid(shape=size, walls=walls, sand=frozenset(sand), goal=goal,
                  start_states=start_states, start=start)


CHAR_MAP = {'W': 'wall', 'G': 'goal', 'B': 'sand', 'S': 'start'}


def load_grid(file_name: str) -> Grid:
  """Load a map file (chars: W wall, G goal, B quicksand, S start)."""
  path = file_name
  if not os.path.isabs(path):
    path = os.path.join(os.path.dirname(__file__), file_name)
  with open(path) as f:
    lines = f.read().splitlines()
  cells = [list(line) for line in lines]
  shape = (len(cells), len(cells[0]))

  walls, sand, goal, start = set(), set(), None, None
  for r, row in enumerate(cells):
    for c, ch in enumerate(row):
      if ch == 'W':
        walls.add((r, c))
      elif ch == 'B':
        sand.add((r, c))
      elif ch == 'G':
        goal = (r, c)
      elif ch == 'S':
        start = (r, c)
  assert goal is not None, f'no goal in {file_name}'
  start = start or goal

  reachable = _bfs_reachable(shape, walls, goal)
  return Grid(shape=shape, walls=frozenset(walls), sand=frozenset(sand), goal=goal,
              start_states=tuple(sorted(reachable - {goal})), start=start)


class GridWorld:
  """The real environment: stochastic transitions, true rewards."""

  n_actions = N_ACTIONS

  def __init__(self, grid: Grid, move_reward=-1., bump_reward=-5., sand_reward=-100.,
               terminal_reward=0., stochastic=True, random_start=True,
               max_steps: int | None = None, seed: int | None = None):
    self.grid = grid
    self.move_reward = move_reward
    self.bump_reward = bump_reward
    self.sand_reward = sand_reward
    self.terminal_reward = terminal_reward
    self.stochastic = stochastic
    self.random_start = random_start
    self.max_steps = max_steps
    self.rng = np.random.default_rng(seed)

    self.state = grid.start
    self._steps = 0
    self._done = False

  # -- factory helpers -------------------------------------------------------
  @classmethod
  def random(cls, size=(50, 50), wall_frac=0.2, sand_frac=0.25,
             min_reachable_frac=0.5, seed: int | None = None, **kwargs):
    rng = np.random.default_rng(seed)
    grid = generate_grid(size, wall_frac, sand_frac, min_reachable_frac, rng)
    return cls(grid, seed=None if seed is None else seed + 1, **kwargs)

  @classmethod
  def from_file(cls, file_name: str, seed: int | None = None, **kwargs):
    return cls(load_grid(file_name), seed=seed, **kwargs)

  # -- environment interface -------------------------------------------------
  def reset(self):
    self._done = False
    self._steps = 0
    if self.random_start:
      states = self.grid.start_states
      self.state = states[self.rng.integers(len(states))]
    else:
      self.state = self.grid.start
    return self.state

  def step(self, action: int):
    assert not self._done, 'step() after episode end'
    executed = action
    if self.stochastic:
      executed = int(self.rng.choice(N_ACTIONS, p=STOCHASTIC_PROBS[action]))

    dr, dc = ACTIONS[executed]
    proposed = (self.state[0] + dr, self.state[1] + dc)
    bumped = proposed in self.grid.walls
    new_state = self.state if bumped else proposed

    reward = self.move_reward
    if bumped:
      reward += self.bump_reward
    if new_state in self.grid.sand:
      reward += self.sand_reward

    terminated = new_state == self.grid.goal
    if terminated:
      reward += self.terminal_reward

    self.state = new_state
    self._steps += 1
    truncated = (not terminated and self.max_steps is not None
                 and self._steps >= self.max_steps)
    self._done = terminated or truncated
    return new_state, reward, terminated, truncated

  # -- planning model --------------------------------------------------------
  def model(self, constant_cost=True):
    return GridModel(self.grid, constant_cost=constant_cost,
                     move_reward=self.move_reward, bump_reward=self.bump_reward,
                     sand_reward=self.sand_reward)


class GridModel:
  """Deterministic planning model over the same maze.

  With `constant_cost=True` (the thesis setting) every action costs 1: the
  model is ignorant of both the stochasticity and the true rewards. With
  `constant_cost=False` costs equal the negated deterministic-case rewards,
  i.e. the model knows about quicksand and bump penalties but still not about
  stochasticity.
  """

  n_actions = N_ACTIONS

  def __init__(self, grid: Grid, constant_cost=True,
               move_reward=-1., bump_reward=-5., sand_reward=-100.):
    self.grid = grid
    self.constant_cost = constant_cost
    self.move_reward = move_reward
    self.bump_reward = bump_reward
    self.sand_reward = sand_reward

  def actions(self, s):
    return range(N_ACTIONS)

  def next_state(self, s, a):
    dr, dc = ACTIONS[a]
    proposed = (s[0] + dr, s[1] + dc)
    return s if proposed in self.grid.walls else proposed

  def cost(self, s, a, s_next):
    if self.constant_cost:
      return 1.
    c = -self.move_reward
    if s_next == s:
      c += -self.bump_reward
    if s_next in self.grid.sand:
      c += -self.sand_reward
    return c

  def reward(self, s, a, s_next):
    return -self.cost(s, a, s_next)

  def is_goal(self, s):
    return s == self.grid.goal

  def heuristic(self, s):
    """Manhattan distance to the true goal; admissible for unit costs.

    Note: when the goal set is augmented with learnt states this is no longer
    admissible for the nearest augmented goal. The 2022 experiments used a
    zero heuristic (Dijkstra); planners take the heuristic as an option so
    both settings are available.
    """
    return abs(s[0] - self.grid.goal[0]) + abs(s[1] - self.grid.goal[1])
