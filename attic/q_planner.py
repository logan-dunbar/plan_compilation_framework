from collections import defaultdict
import numpy as np

from plan_compilation_framework.agents import Agent
from plan_compilation_framework.agents.qlearning import QLearning
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.policies.epsilon_greedy import EpsilonGreedy


class QPlanner(Agent):
  def __init__(self, env, planner, n_actions, default_q=0., seed=0):
    super().__init__(env)

    self.planner = planner
    self.learner = QLearning(env, n_actions, default_q, seed=seed)

    self.plan = None

  def begin_episode(self):
    self.plan = None
    self.learner.begin_episode()

  def get_action(self, obs):
    if self.plan is None:
      self.plan = self.planner.get_plan(obs)

    return self.plan.next_action()

  def do_update(self, t: Transition):
    self.learner.do_update(t)

  def end_episode(self):
    self.learner.end_episode()

  def get_values(self, states):
    return self.learner.get_values(states)
