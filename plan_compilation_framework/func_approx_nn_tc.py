import jax
import numpy as np
import gymnasium as gym
from gymnasium import RewardWrapper
import wandb

from plan_compilation_framework.agents.nn_tc_agent import MyLayeredTileCoder, NNTCContQAgent
from plan_compilation_framework.func_approx import run_experiment
from plan_compilation_framework.helpers.buffer import ReplayBuffer, NStepReplayBuffer
from plan_compilation_framework.helpers.plotter import Plotter
from plan_compilation_framework.helpers.schedules import LinearSchedule, CosineAnnealingSchedule, CompositeSchedule
from plan_compilation_framework.helpers.wrappers import ReacherRewardWrapper


def jax_runner():
  config = dict(
    episodes=100000,
    gamma=0.9,
    init_q=0.,
    buffer_sz=1e5,
    batch_sz=64,
    min_updates=100,
    update_freq=64,
    eval_freq=100,
    eval_num=5,
  )

  # run = wandb.init(project='sample-project', config={
  #
  # })

  episodes = 100000
  gamma = 0.9
  init_q = 0.
  buffer_sz = 1e5
  batch_sz = 64
  min_updates = 100
  update_freq = 64
  plot_freq_ep = 10
  eval_freq = 100
  eval_num = 5
  obs_scaler = None
  rew_scaler = None
  disable_jit = False
  width, height = 640, 480

  eps_init = 1.0
  # eps = ConstantSchedule(eps_init)
  # eps = LinearSchedule(eps_init, 0.01, 10000)
  # eps = CosineAnnealingSchedule(0.01, eps_init, 67, 150, end_min=True)

  eps1 = CosineAnnealingSchedule(0.1, 0.8, 100, 100, end_min=True)
  eps2 = LinearSchedule(1., 0.5, 10000)
  eps = CompositeSchedule([eps1, eps2])

  alpha_init = 0.001
  alpha = LinearSchedule(alpha_init, alpha_init / 10., 5000)
  # alpha = CosineAnnealingSchedule(alpha_init / 20., alpha_init, 47, 150, end_min=True)

  # alpha1 = CosineAnnealingSchedule(alpha_init / 20., alpha_init, 47, 150, end_min=True)
  # alpha2 = LinearSchedule(1., 0.1, 10000)
  # alpha = CompositeSchedule([alpha1, alpha2])

  # Inv Pendulum
  # env = gym.make('InvertedPendulum-v4', width=width, height=height)
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1.5, -0.5, -4.5, -8.]),
  #   np.array([1.5, 0.5, 4.5, 8.]),
  #   dtype=env.observation_space.dtype
  # )

  # Reacher
  # env = gym.make('Reacher-v4', width=width, height=height)
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]),
  #   np.array([1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]),
  #   dtype=env.observation_space.dtype
  # )
  # env = ReacherRewardWrapper(env, ctrl_gain=0.1)  # makes it similar to Pusher env

  # Pusher
  env = gym.make('Pusher-v4', width=width, height=height)
  env.unwrapped.observation_space = gym.spaces.Box(
    np.array([-2.5, -2., -2.5, -3., -2., -2., -2.5, -2., -2.5, -15., -10.,
              -15., -10., -15., -1., -2., -1., -1., -1., -0.5, -0.45, -0.5, -0.5]),
    np.array([2.5, 2., 2.5, 1., 2., 1., 2.5, 2., 2.5, 15., 10., 15., 10.,
              15., 1.5, 1., 1., 1.5, 1., 0.5, 0.45, 0.5, 0.5]),
    dtype=env.observation_space.dtype
  )

  e_low, e_high = env.observation_space.low, env.observation_space.high
  a_low, a_high = env.action_space.low, env.action_space.high
  e_dim, a_dim = e_low.shape[0], a_low.shape[0]

  # tiles
  tiles = [[t] * (e_dim + a_dim) for t in [2, 3, 5, 7, 11]]
  tilings = [np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim)))))] * len(tiles)

  e_tiles = [t[:e_dim] for t in tiles]
  a_tiles = [t[-a_dim:] for t in tiles]

  obs_tc = MyLayeredTileCoder(e_tiles, tilings, e_low, e_high)
  act_tc = MyLayeredTileCoder(a_tiles, tilings, a_low, a_high)

  # buffer = ReplayBuffer(max_size=buffer_sz, dtype=np.double)
  buffer = NStepReplayBuffer(max_size=buffer_sz, dtype=np.double, n=4)

  with jax.disable_jit(disable=disable_jit):
    agent = NNTCContQAgent(env.action_space, a_low, a_high,
                           env.observation_space, e_low, e_high,
                           eps, gamma, alpha,
                           buffer, batch_sz, min_updates, update_freq,
                           obs_tc, act_tc,
                           q_dims=(256, 128), p_dims=(128, 64))

    t_plotter = Plotter(plot_freq_ep, title='Train')
    e_plotter = Plotter(1, title='Eval')
    # t_plotter = None
    # e_plotter = None

    # env.unwrapped.render_mode = 'human'

    run_experiment(env, agent, episodes,
                   train_plotter=t_plotter, eval_plotter=e_plotter,
                   obs_scaler=obs_scaler, rew_scaler=rew_scaler,
                   eval_freq=eval_freq, eval_num=eval_num)
  # plt.show()
  debug = 0


if __name__ == '__main__':
  jax_runner()
