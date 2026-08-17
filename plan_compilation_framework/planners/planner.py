from copy import deepcopy
import numpy as np

from plan_compilation_framework.environments import GymPlannerWrapper


class Planner:
  def __init__(self, env, constant_reward=True, set_state=None, seed=0):
    self.np_random = np.random.RandomState(seed=seed)
    self.model = deepcopy(env)
    self.constant_reward = constant_reward

    if hasattr(self.model.unwrapped, 'stochastic'):
      setattr(self.model.unwrapped, 'stochastic', False)

    if hasattr(self.model, 'random_start'):
      setattr(self.model, 'random_start', False)

    if 'TimeLimit' in str(self.model):
      tmp_model = self.model
      while not str(tmp_model).startswith('<TimeLimit'):
        tmp_model = tmp_model.env
      tmp_model.spec.max_episode_steps = tmp_model._max_episode_steps = np.Inf

    # only some envs, might need custom wrappers for specific envs
    self.model = GymPlannerWrapper(self.model, set_state=set_state)
