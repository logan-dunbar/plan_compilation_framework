from collections import defaultdict
import numpy as np

from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.policies.epsilon_greedy import EpsilonGreedy


class QLearning:
  def __init__(self, n_actions, default_q=0., seed=0):
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
    q_values = self.q_values(obs)
    probs = self.policy.probabilities(q_values)
    action = self.np_random.choice(range(self.n_actions), p=probs)
    return action

  def do_update(self, t: Transition):
    q_next = 0. if t.a_d else self.max_action_value(t.s_p)[1]
    target = t.r_p + self.gamma * q_next
    self.q[t.a][t.s] += self.alpha * (target - self.q[t.a][t.s])

  def end_episode(self):
    pass

  def max_action_value(self, obs):
    q_values = self.q_values(obs)
    max_actions = np.nonzero(q_values == np.max(q_values))[0]
    action = self.np_random.choice(max_actions)
    return action, q_values[action]

  def q_values(self, obs):
    return np.array([self.q[a][obs] for a in range(self.n_actions)])

