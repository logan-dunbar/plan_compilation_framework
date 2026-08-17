import numpy as np
import gymnasium as gym

from plan_compilation_framework.func_approx import run_experiment
from plan_compilation_framework.helpers import Transition


class RandomAgent:
  def __init__(self, act_sp):
    self.act_sp = act_sp
    self.step = 0

    actions = [[0.25, 0., 0., 0., 0., 0., 0.]] * 20
    actions += [[0., 0.25, 0., 0., 0., 0., 0.]] * 50
    actions += [[0., 0., -0.25, 0., 0., 0., 0.]] * 20
    actions += [[0., 0., 0., -0.5, 0., 0., 0.]] * 10
    actions += [[0., 0., 0., 0., 0.5, 0., 0.]] * 10
    actions += [[0., 0., 0., 0., 0., 0.5, 0.]] * 10
    actions += [[0., 0., 0., 0., 0., 0., 0.5]] * 10
    actions += [[0.5, 0., 0., 0., 0., 0., 0.]] * 10
    actions += [[0., 0.5, 0., 0., 0., 0., 0.]] * 10
    actions += [[0., 0., 0.5, 0., 0., 0., 0.]] * 11
    self.actions = actions

  def begin_episode(self):
    self.step = 0

  def end_episode(self):
    pass

  def get_action(self, obs):
    return self.actions.pop(0)
    # return self.act_sp.sample() * 0.5
    # return np.zeros(self.act_sp.low.shape)

  def do_update(self, t: Transition):
    pass


def runner():
  episodes = 100000
  eval_freq = None
  eval_num = None
  obs_scaler = None
  rew_scaler = None

  # Inv Pendulum
  # env = gym.make('InvertedPendulum-v4')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1.5, -0.5, -4.5, -8.]),
  #   np.array([1.5, 0.5, 4.5, 8.]),
  #   dtype=env.observation_space.dtype
  # )

  # Reacher
  # env = gym.make('Reacher-v4')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]),
  #   np.array([1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]),
  #   dtype=env.observation_space.dtype
  # )
  # env = ReacherRewardWrapper(env, ctrl_gain=0.1)  # makes it similar to Pusher env

  # Pusher
  env = gym.make('Pusher-v4')
  env.unwrapped.observation_space = gym.spaces.Box(
    np.array([-2.5, -2., -2.5, -3., -2., -2., -2.5, -2., -2.5, -15., -10.,
              -15., -10., -15., -1., -2., -1., -1., -1., -0.5, -0.45, -0.5, -0.5]),
    np.array([2.5, 2., 2.5, 1., 2., 1., 2.5, 2., 2.5, 15., 10., 15., 10.,
              15., 1.5, 1., 1., 1.5, 1., 0.5, 0.45, 0.5, 0.5]),
    dtype=env.observation_space.dtype
  )

  env.unwrapped.render_mode = 'human'
  agent = RandomAgent(env.action_space)

  t_plotter = None
  e_plotter = None

  run_experiment(env, agent, episodes,
                 train_plotter=t_plotter, eval_plotter=e_plotter,
                 obs_scaler=obs_scaler, rew_scaler=rew_scaler,
                 eval_freq=eval_freq, eval_num=eval_num)
  # plt.show()
  debug = 0


if __name__ == '__main__':
  runner()
