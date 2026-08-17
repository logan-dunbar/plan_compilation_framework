from collections import defaultdict

import numpy as np
import networkx as nx

from plan_compilation_framework.planners.plan import Plan
from plan_compilation_framework.planners.planner import Planner


class RrtEx(Planner):
  def __init__(self, env, n_bins, budget=1e5, **kwargs):
    super().__init__(env, **kwargs)

    self.n_bins = n_bins
    self.budget = int(budget)

    obs_sp = self.model.observation_space
    scale = n_bins / (obs_sp.high - obs_sp.low)
    self.get_bins = lambda s: tuple(np.round((np.array(s) - obs_sp.low) * scale, 6).astype(int))

    # annoying.
    self.W_b = None
    self.B_s = None
    self.B_m = None
    self.m = None
    self.g = None

    self.reset()

  def reset(self):
    self.W_b = defaultdict(lambda: [])
    self.B_s = defaultdict(lambda: defaultdict(lambda: []))
    self.B_m = {}
    self.m = 0
    self.g = nx.DiGraph()

  def get_plan(self, start, is_goal=None):
    is_goal = is_goal if is_goal else lambda x: False

    self.reset()

    start = tuple(start)
    self.ex_insert(start)
    for i in range(self.budget):
      s = self.ex_select()
      a = self.model.action_space.sample()
      self.model.reset()
      self.model.set_state(s)
      
      s_p, rew, done, trunc, info = self.model.step(a)
      assert not trunc

      s_p = tuple(s_p)
      self.ex_insert(s_p)

      self.g.add_edge(s, s_p, reward=-rew, action=a)

      if done or is_goal(s_p):
        path = nx.astar_path(self.g, start, s_p, weight='reward')
        plan = {s1: self.g.edges[s1, s2]['action'] for s1, s2 in zip(path[:-1], path[1:])}
        return Plan(plan)

    raise Exception('No plan found')

  def ex_insert(self, s):
    b = self.get_bins(s)
    if b not in self.B_m:
      self.m = 0
      self.W_b[0].append(b)
    self.bin_insert(b, s)

  def bin_insert(self, b, s):
    self.B_s[b][0].append(s)
    self.B_m[b] = 0

  def ex_select(self):
    b = self.W_b[self.m][self.np_random.choice(len(self.W_b[self.m]))]
    self.W_b[self.m].remove(b)
    self.W_b[self.m + 1].append(b)
    if len(self.W_b[self.m]) == 0:
      self.m += 1
    return self.bin_select(b)

  def bin_select(self, b):
    m_b = self.B_m[b]
    s = self.B_s[b][m_b][self.np_random.choice(len(self.B_s[b][m_b]))]
    self.B_s[b][m_b].remove(s)
    self.B_s[b][m_b + 1].append(s)
    if len(self.B_s[b][m_b]) == 0:
      self.B_m[b] += 1
    return s
