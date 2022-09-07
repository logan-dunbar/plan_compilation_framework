from copy import deepcopy

import os
from typing import Any, TypedDict

import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.envs.classic_control.rendering import SimpleImageViewer

import numpy as np
import networkx as nx

CHAR_MAP = {
  ' ': 0,  # empty
  'W': 1,  # wall
  'G': 2,  # goal
  'B': 3,  # bomb
  'S': 4,  # start
}
ACTION_MAP = {
  0: np.array([-1, 0]),
  1: np.array([0, 1]),
  2: np.array([1, 0]),
  3: np.array([0, -1]),
}
ACTION_NAME_MAP = {
  0: 'UP',
  1: 'RIGHT',
  2: 'DOWN',
  3: 'LEFT',
}
STOCHASTIC_PROBS_MAP = {
  0: [0.8, 0.1, 0, 0.1],
  1: [0.1, 0.8, 0.1, 0],
  2: [0, 0.1, 0.8, 0.1],
  3: [0.1, 0, 0.1, 0.8],
}


class SimpleGridWorld(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 15}

  def __init__(self,
               seed=None,
               random_start=False,
               file_name=None,
               random_graph=False,
               stochastic=False,
               terminal_reward=10.0,
               move_reward=-1.0,
               bump_reward=-5.0,
               bomb_reward=-50.0,
               size=(20, 20)):
    self.viewer = None
    self.display_grid = None
    self.np_random = np.random.RandomState(seed=seed)

    if random_graph:
      self.walls, self.goals, self.bombs, self.start_states, self.start, self.shape = self.generate_random_graph(size)
    else:
      assert file_name is not None
      self.walls, self.goals, self.bombs, self.start_states, self.start, self.shape = self.load_from_file(file_name)

    self.terminal_reward = terminal_reward
    self.move_reward = move_reward
    self.bump_reward = bump_reward
    self.bomb_reward = bomb_reward
    self.random_start = random_start
    self.stochastic = stochastic

    self.n_actions = 4
    self.action_space = spaces.Discrete(self.n_actions)
    self.observation_space = spaces.Box(low=np.array([0, 0]), high=np.array(self.shape), dtype=np.int)

    self.state = deepcopy(self.start)
    self.done = False

  def generate_random_graph(self, size, wall_prop=0.2, bomb_prop=0.25, min_allowed_prop=0.5):
    while True:
      total_size = size[0] * size[1]

      border_inds = []
      for h in range(size[0]):
        border_inds.append((h, 0))
        border_inds.append((h, size[1] - 1))
      for w in range(size[1]):
        border_inds.append((0, w))
        border_inds.append((size[0] - 1, w))
      border_inds = np.ravel_multi_index(list(zip(*border_inds)), dims=size)

      grid_inds = range(total_size)
      wall_inds = self.np_random.choice(grid_inds, int(total_size * wall_prop))
      wall_inds = set(wall_inds).union(border_inds)
      free_inds = list(set(grid_inds) - wall_inds)

      # slightly favour corridor like configurations
      p = np.array([h % 2 + w % 2 + 1 for h, w in zip(*np.unravel_index(free_inds, shape=size))])
      p = p / np.sum(p)

      bomb_inds = self.np_random.choice(free_inds, int(len(free_inds) * bomb_prop), p=p)
      bomb_nodes = list(zip(*np.unravel_index(list(bomb_inds), shape=size)))

      graph = nx.grid_graph(size)
      wall_nodes = list(zip(*np.unravel_index(list(wall_inds), shape=size)))
      graph.remove_nodes_from(wall_nodes)

      placed = False
      for i in range(5):  # 5 tries to place goal
        goal_node = np.unravel_index(self.np_random.choice(free_inds), shape=size)
        u = graph.to_undirected()
        nodes = nx.shortest_path(u, goal_node)
        subgraph = graph.subgraph(nodes)
        allowed_start_nodes = subgraph.nodes
        if len(allowed_start_nodes) > min_allowed_prop * total_size:
          placed = True
          break

      if placed:
        if goal_node in bomb_nodes:
          bomb_nodes.remove(goal_node)

        while (start_node := np.unravel_index(self.np_random.choice(free_inds), shape=size)) == goal_node:
          pass
        break

    return set(wall_nodes), {goal_node}, set(bomb_nodes), list(allowed_start_nodes), start_node, size

  def load_from_file(self, file_name):
    with open(os.path.join(os.path.dirname(__file__), file_name), 'r') as f:
      grid = np.array([list(l) for l in f.read().splitlines()], dtype=np.str)

    walls = set(zip(*np.nonzero(grid == 'W')))
    goals = set(zip(*np.nonzero(grid == 'G')))
    bombs = set(zip(*np.nonzero(grid == 'B')))
    start_states = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1]) if (r, c) not in walls]
    start = list(zip(*np.nonzero(grid == 'S')))[0]

    return walls, goals, bombs, start_states, start, grid.shape

  def reset(self):
    self.done = False
    if self.random_start:
      while (start := self.start_states[self.np_random.randint(len(self.start_states))]) in self.goals:
        pass
      self.state = start
    else:
      self.state = deepcopy(self.start)

    return self.state

  def step(self, action):
    assert self.action_space.contains(action)

    self.state = deepcopy(self.state)

    if self.done:
      return self.state, 0.0, self.done, {}
    else:
      new_state = self.take_action(action)
      reward = self.get_reward(new_state)
      self.state = new_state
      return self.state, reward, self.done, {}

  def take_action(self, action):
    if self.stochastic:
      action = self.np_random.choice(range(self.n_actions), p=STOCHASTIC_PROBS_MAP[action])
    new_state = tuple(np.array(self.state) + ACTION_MAP[action])
    return new_state if new_state not in self.walls else self.state

  def get_reward(self, new_agent):
    reward = self.move_reward

    if new_agent in self.goals:
      self.done = True
      reward += self.terminal_reward
    if new_agent in self.bombs:
      reward += self.bomb_reward
    if new_agent == self.state:
      reward += self.bump_reward

    return reward

  def set_state(self, state):
    self.state = deepcopy(state)

  def render(self, mode='human', close=False, agent=None):
    if close:
      self.close()
      return

    if self.display_grid is None:
      self.display_grid = np.multiply(np.ones((*self.shape, 3), dtype=np.uint8),
                                      np.array([0, 255, 0], dtype=np.uint8))
      for b in self.bombs:
        self.display_grid[b] = np.array([255, 255, 0])
      for w in self.walls:
        self.display_grid[w] = np.array([0, 0, 0])
      for g in self.goals:
        self.display_grid[g] = np.array([255, 0, 0])
      self.display_grid = self.display_grid.repeat(30, axis=0).repeat(30, axis=1)

    grid = self.display_grid.copy()

    agent_color = [0, 0, 255]
    grid[self.state[0] * 30:self.state[0] * 30 + 30, self.state[1] * 30:self.state[1] * 30 + 30] = np.array(agent_color)

    if mode == 'human':
      if not self.viewer:
        self.viewer = SimpleImageViewer(maxwidth=1000)
      self.viewer.imshow(grid)
      return self.viewer.isopen
    elif mode == 'rgb_array':
      return grid
    else:
      return

  def close(self):
    if self.viewer is not None:
      self.viewer.close()
      self.viewer = None
