from collections import defaultdict
import numpy as np

from plan_compilation_framework.agents import Agent
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.policies.epsilon_greedy import EpsilonGreedy


class QLearning(Agent):
  def __init__(self, env, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.np_random = np.random.RandomState(seed=seed)
    self.n_actions = n_actions
    self.default_q = default_q

    self.gamma = 1.
    self.alpha = 0.1
    self.policy = EpsilonGreedy(0.1)

    self.q = {a: defaultdict(lambda: self.default_q) for a in range(self.n_actions)}

  def begin_episode(self):
    pass

  def get_action(self, obs):
    feat = self.env.state_features(obs)

    q_values = self.q_values(feat)
    probs = self.policy.probabilities(q_values)
    action = self.np_random.choice(range(self.n_actions), p=probs)
    return action

  def do_update(self, t: Transition):
    feat, feat_p = self.env.state_features(t.s), self.env.state_features(t.s_p)

    q_next = 0. if t.a_d else self.max_action_value(feat_p)[1]
    target = t.r_p + self.gamma * q_next
    self.q[t.a][feat] += self.alpha * (target - self.q[t.a][feat])

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
