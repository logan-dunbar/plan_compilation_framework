from collections import defaultdict
from typing import Optional

import numpy as np

from plan_compilation_framework.agents import Agent
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.divergence import js_divergence
from plan_compilation_framework.planners.plan import Plan
from plan_compilation_framework.policies.epsilon_greedy import EpsilonGreedy


class PCPlanner(Agent):
  def __init__(self, env, parent, planner, seed=0):
    super().__init__(env)

    self.parent: PlanCompiler = parent
    self.planner = planner
    self.plan: Optional[Plan] = None
    self.steps = 0

  def begin_episode(self):
    self.plan = None
    self.steps = 0

  def get_action(self, obs):
    if self.parent.learner.is_learnt(obs):
      self.parent.current = self.parent.learner
      return

    if self.plan is None:
      is_goal = self.parent.learner.get_is_goal(obs)
      self.plan = self.planner.get_plan(obs, is_goal)

    self.steps += 1
    return self.plan.next_action()

  def do_update(self, t: Transition):
    pass

  def end_episode(self):
    pass


class PCLearner(Agent):
  def __init__(self, env, parent, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.parent: PlanCompiler = parent
    self.np_random = np.random.RandomState(seed=seed)
    self.n_actions = n_actions
    self.default_q = default_q

    self.gamma = 1.
    self.alpha = 0.1
    self.alpha_l = 0.1
    self.thresh_l = 0.9
    self.exp_eps = 0.05
    self.quota_frac = 0.1
    self.policy = EpsilonGreedy(0.1)

    self.q = {a: defaultdict(lambda: self.default_q) for a in range(self.n_actions)}
    self.l = defaultdict(lambda: 0.)
    self.traj = []
    self.steps = 0

  def begin_episode(self):
    self.traj.clear()
    self.steps = 0

  def get_action(self, obs):
    if not self.is_learnt(obs):
      self.parent.current = self.parent.planner
      self.parent.planner.plan = None
      return

    feat = self.env.state_features(obs)
    action, value = self.max_action_value(feat)
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
      feat_p = self.env.state_features(t.s_p)
      G_init = 0. if t.a_d else self.max_action_value(feat_p)[1]
      self.monte_carlo(G_init)

  def end_episode(self):
    pass

  def bootstrap(self, t: Transition):
    feat, feat_p = self.env.state_features(t.s), self.env.state_features(t.s_p)

    q_next = 0. if t.a_d else self.max_action_value(feat_p)[1]
    target = t.r_p + self.gamma * q_next
    self.q[t.a][feat] += self.alpha * (target - self.q[t.a][feat])

  def monte_carlo(self, G_init):
    G = G_init
    for t in reversed(self.traj):
      # if self.is_learnt(t.s_p):
      #   feat_p = self.env.state_features(t.s_p)
      #   _, q_p = self.max_action_value(feat_p)
      #   G = t.r_p + np.max([G, q_p])
      # else:
      #   G = t.r_p + G

      G = t.r_p + G

      feat = self.env.state_features(t.s)

      q_values_pre = self.q_values(feat)
      probs_pre = self.policy.probabilities(q_values_pre)

      self.q[t.a][feat] += self.alpha * (G - self.q[t.a][feat])

      q_values_post = self.q_values(feat)
      probs_post = self.policy.probabilities(q_values_post)

      divergence = js_divergence(probs_pre, probs_post)
      l_target = 1. if divergence < 0.01 or self.is_learnt(t.s) else 0.
      self.l[feat] += self.alpha_l * (l_target - self.l[feat])

    self.traj.clear()

  def max_action_value(self, feat):
    q_values = self.q_values(feat)
    max_actions = np.nonzero(q_values == np.max(q_values))[0]
    action = self.np_random.choice(max_actions)
    return action, q_values[action]

  def q_values(self, feat):
    return np.array([self.q[a][feat] for a in range(self.n_actions)])

  def get_values(self, states):
    values = np.zeros(states.shape[0])
    for i, s in enumerate(states):
      feat = self.env.state_features(tuple(s))
      q_values = self.q_values(feat)
      values[i] = np.max(q_values)
    return values

  def is_learnt(self, obs):
    feat = self.env.state_features(obs)
    return self.l[feat] > self.thresh_l

  def get_is_goal(self, start):
    feat = self.env.state_features(start)
    _, start_value = self.max_action_value(feat)

    def is_goal(state):
      if self.is_learnt(state):
        feat_p = self.env.state_features(state)
        _, state_value = self.max_action_value(feat_p)
        return state_value > start_value

      return False

    return is_goal


class PCExplorer(Agent):
  def __init__(self, env, parent, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.parent: PlanCompiler = parent
    self.np_random = np.random.RandomState(seed=seed)
    self.n_actions = n_actions
    self.default_q = default_q

    self.gamma = 1.
    self.alpha = 0.1
    self.policy = EpsilonGreedy(0.1)

    self.q = {a: defaultdict(lambda: self.default_q) for a in range(self.n_actions)}
    self.quota = 0
    self.steps = 0

  def begin_episode(self):
    self.quota = 0
    self.steps = 0

  def get_action(self, obs):
    if self.quota <= 0:
      self.parent.current = self.parent.planner
      self.parent.planner.plan = None
      return

    feat = self.env.state_features(obs)
    q_values = self.q_values(feat)
    probs = self.policy.probabilities(q_values)
    action = self.np_random.choice(range(self.n_actions), p=probs)
    self.steps += 1
    return action

  def do_update(self, t: Transition):
    feat, feat_p = self.env.state_features(t.s), self.env.state_features(t.s_p)

    q_next = 0. if t.a_d else self.max_action_value(feat_p)[1]
    target = t.r_p + self.gamma * q_next
    self.q[t.a][feat] += self.alpha * (target - self.q[t.a][feat])

    if self.quota > 0:
      self.quota -= abs(t.r_p)

  def end_episode(self):
    pass

  def max_action_value(self, feat):
    q_values = self.q_values(feat)
    max_actions = np.nonzero(q_values == np.max(q_values))[0]
    action = self.np_random.choice(max_actions)
    return action, q_values[action]

  def q_values(self, feat):
    return np.array([self.q[a][feat] for a in range(self.n_actions)])

  def get_values(self, states):
    values = np.zeros(states.shape[0])
    for i, s in enumerate(states):
      feat = self.env.state_features(tuple(s))
      q_values = self.q_values(feat)
      values[i] = np.max(q_values)
    return values


class PlanCompiler(Agent):
  def __init__(self, env, planner, n_actions, learner_default_q=0., explorer_default_q=0., seed=0):
    super().__init__(env)
    self.planner = PCPlanner(env, self, planner, seed)
    self.learner = PCLearner(env, self, n_actions, learner_default_q, seed)
    self.explorer = PCExplorer(env, self, n_actions, explorer_default_q, seed)

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
