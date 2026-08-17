from collections import defaultdict
from typing import Optional

import numpy as np

from plan_compilation_framework.agents import Agent
from plan_compilation_framework.agents.tc_q_learning import TCQFunc, TCLFunc
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.divergence import js_divergence
from plan_compilation_framework.planners.plan import Plan
from plan_compilation_framework.policies.epsilon_greedy import EpsilonGreedy


class TCPCPlanner(Agent):
  def __init__(self, env, parent, planner, seed=0):
    super().__init__(env)

    self.parent: TCPlanCompiler = parent
    self.planner = planner
    self.plan: Optional[Plan] = None
    self.steps = 0

  def begin_episode(self):
    self.plan = None
    self.steps = 0

  def populate_plan(self, obs):
    if self.plan is None:
      is_goal = self.parent.learner.get_is_goal(obs)

      # TODO: try except planning fail, explore a bit?
      self.plan = self.planner.get_plan(obs, is_goal)

  def plan_action(self, obs):
    action = None
    while action is None:
      try:
        action = self.plan.next_action()
      except IndexError:
        self.plan = None
        self.populate_plan(obs)
    return action

  def get_action(self, obs):
    if self.parent.learner.is_learnt(obs):
      self.parent.current = self.parent.learner
      self.plan = None
      return

    self.populate_plan(obs)

    self.steps += 1
    return self.plan_action(obs)

  def do_update(self, t: Transition):
    pass

  def end_episode(self):
    pass


class TCPCLearner(Agent):
  def __init__(self, env, parent, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.parent: TCPlanCompiler = parent
    self.np_random = np.random.RandomState(seed=seed)
    self.n_actions = n_actions
    self.default_q = default_q

    self.gamma = 0.99
    self.alpha = 0.1
    self.alpha_l = 0.02
    self.thresh_l = 0.9
    self.exp_eps = 0.01
    self.quota_frac = 0.05
    self.policy = EpsilonGreedy(0.1)

    low, high = env.observation_space.low, env.observation_space.high
    dim = low.shape[0]

    start = 1
    depth = 3
    tiles = [[2 ** t] * dim for t in range(start, depth + 1)]
    tilings = [np.power(2, int(np.ceil(np.log2(4 * dim))))] * len(tiles)
    # tilings = [np.power(2, int(np.floor(np.log2(4 * dim))))] * len(tiles)

    self.q = TCQFunc(n_actions, low, high, tiles, tilings, init_q=self.default_q)
    self.l = TCLFunc(self.q.tc)

    self.traj = []
    self.steps = 0

  def begin_episode(self):
    self.traj.clear()
    self.steps = 0

  def get_q_values(self, obs):
    assert np.ndim(obs) == 2
    q_values = self.q.values(obs)
    return q_values

  def get_values(self, obs):
    return np.max(self.get_q_values(obs), axis=1)

  def get_action(self, obs):
    if not self.is_learnt(obs):
      self.parent.current = self.parent.planner
      self.parent.planner.plan = None
      return

    q_values = self.get_q_values(obs[None, :]).squeeze()
    value = np.max(q_values)
    action = np.random.choice(np.nonzero(q_values == value)[0])

    if self.np_random.random() < self.exp_eps:
      self.parent.explorer.quota = self.quota_frac * abs(value)
      self.parent.current = self.parent.explorer
      return

    self.steps += 1
    return action

  def do_update(self, t: Transition):
    if self.is_learnt(t.s) and self.is_learnt(t.s_p):
      self.bootstrap(t)
    else:
      self.traj.append(t)

    if t.a_d or self.is_learnt(t.s_p):
      G_init = 0. if t.a_d else self.get_values(t.s_p[None, :]).item()
      self.monte_carlo(G_init)

  def end_episode(self):
    pass

  def bootstrap(self, t: Transition):
    q_next = self.get_values(t.s_p[None, :]).item()
    q_curr = self.get_q_values(t.s[None, :])[0, t.a]

    target = t.r_p + (1. - t.a_d) * self.gamma * q_next - q_curr
    self.q.update([t.s], [t.a], [target], self.alpha)

  def monte_carlo(self, G_init):
    G = G_init
    for t in reversed(self.traj):
      # if self.is_learnt(t.s_p):
      #   feat_p = self.env.state_features(t.s_p)
      #   _, q_p = self.max_action_value(feat_p)
      #   G = t.r_p + np.max([G, q_p])
      # else:
      #   G = t.r_p + G

      G = t.r_p + self.gamma * G

      q_values_pre = self.get_q_values(t.s[None, :]).squeeze()
      probs_pre = self.policy.probabilities(q_values_pre)

      target = G - q_values_pre[t.a]
      self.q.update([t.s], [t.a], [target], self.alpha)

      q_values_post = self.get_q_values(t.s[None, :]).squeeze()
      probs_post = self.policy.probabilities(q_values_post)

      divergence = js_divergence(probs_pre, probs_post)
      l_curr = self.l.values(t.s[None, :]).item()
      l_target = (1. if divergence < 0.01 or self.is_learnt(t.s) else 0.) - l_curr
      self.l.update([t.s], [l_target], self.alpha_l)

    self.traj.clear()

  def is_learnt(self, obs):
    return self.l.values(obs[None, :]).item() > self.thresh_l

  def get_is_goal(self, start):
    start_value = self.get_values(start[None, :]).item()

    def is_goal(state):
      state = np.array(state)
      if self.is_learnt(state):
        state_value = self.get_values(state[None, :]).item()
        return state_value > start_value

      return False

    return is_goal


class TCPCExplorer(Agent):
  def __init__(self, env, parent, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.parent: TCPlanCompiler = parent
    self.np_random = np.random.RandomState(seed=seed)
    self.actions = list(range(n_actions))
    self.default_q = default_q

    self.gamma = 1.
    self.alpha = 0.1
    self.policy = EpsilonGreedy(0.1)

    low, high = env.observation_space.low, env.observation_space.high
    dim = low.shape[0]

    start = 1
    depth = 3
    tiles = [[2 ** t] * dim for t in range(start, depth + 1)]
    tilings = [np.power(2, int(np.ceil(np.log2(4 * dim))))] * len(tiles)

    self.q = TCQFunc(n_actions, low, high, tiles, tilings, init_q=self.default_q)

    self.quota = 0
    self.steps = 0

  def begin_episode(self):
    self.quota = 0
    self.steps = 0

  def get_q_values(self, obs):
    assert np.ndim(obs) == 2
    q_values = self.q.values(obs)
    return q_values

  def get_values(self, obs):
    return np.max(self.get_q_values(obs), axis=1)

  def get_action(self, obs):
    if self.quota <= 0:
      self.parent.current = self.parent.planner
      self.parent.planner.plan = None
      return

    q_values = self.get_q_values(obs[None, :]).squeeze()
    probs = self.policy.probabilities(q_values)
    action = self.np_random.choice(self.actions, p=probs)
    self.steps += 1
    return action

  def do_update(self, t: Transition):
    q_next = self.get_values(t.s_p[None, :]).item()
    q_curr = self.get_q_values(t.s[None, :])[0, t.a]

    target = t.r_p + (1. - t.a_d) * self.gamma * q_next - q_curr
    self.q.update([t.s], [t.a], [target], self.alpha)

    if self.quota > 0:
      self.quota -= abs(t.r_p)

  def end_episode(self):
    pass


class TCPlanCompiler(Agent):
  def __init__(self, env, planner, n_actions, learner_default_q=0., explorer_default_q=0., seed=0):
    super().__init__(env)
    self.planner = TCPCPlanner(env, self, planner, seed)
    self.learner = TCPCLearner(env, self, n_actions, learner_default_q, seed)
    self.explorer = TCPCExplorer(env, self, n_actions, explorer_default_q, seed)

    self.current = None

  def begin_episode(self):
    self.current = self.planner

    self.planner.begin_episode()
    self.learner.begin_episode()
    self.explorer.begin_episode()

  def get_action(self, obs):
    action = None
    while action is None:
      action = self.current.get_action(obs)
    return action

  def do_update(self, t: Transition):
    self.planner.do_update(t)
    self.learner.do_update(t)
    self.explorer.do_update(t)

  def end_episode(self):
    print(f'planner: {self.planner.steps: 5}, learner: {self.learner.steps: 5}, explorer: {self.explorer.steps: 5}')

  def get_values(self, states):
    return self.learner.get_values(states)
    # return self.explorer.get_values(states)
