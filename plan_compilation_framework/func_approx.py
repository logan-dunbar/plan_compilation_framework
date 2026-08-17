import numpy as np
import gymnasium as gym

import matplotlib
import matplotlib.pyplot as plt

from plan_compilation_framework.agents.jax_agent import JaxLayeredTileCoder, JaxContTCQFunc, JaxContQAgent, \
  MultiJaxContTCQFunc, JaxContTCTabQFunc, ProperTCQAgent, ProperTCQFunc, ProperTCPFunc
from plan_compilation_framework.agents.tc_plan_compiler import TCPlanCompiler
from plan_compilation_framework.agents.tc_q_learning import TCQFunc, QAgent, ContQAgent, ContTCQFunc, \
  BufferAgent
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.buffer import ReplayBuffer
from plan_compilation_framework.helpers.plotter import Plotter
from plan_compilation_framework.helpers.schedules import LinearSchedule, ConstantSchedule, CosineAnnealingSchedule, \
  StepSchedule
from plan_compilation_framework.helpers.tile_coder import TileCoder, LayeredTileCoder
from plan_compilation_framework.planners.rrt_ex import RrtEx

matplotlib.use('TkAgg')


def run_episode(env, agent, obs_scaler=None, rew_scaler=None, plotter=None, train=True):
  obs_scaler = (lambda o: o) if obs_scaler is None else obs_scaler
  rew_scaler = (lambda r: r) if rew_scaler is None else rew_scaler

  term = trunc = False
  ep_rew = 0.
  i = 0

  obs, _ = env.reset()
  obs = obs_scaler(obs)

  obs_min = obs
  obs_max = obs

  agent.begin_episode()

  while not (term or trunc):
    act = agent.get_action(obs)

    obs_n, rew, term, trunc, info = env.step(act)
    obs_n, rew = obs_scaler(obs_n), rew_scaler(rew)
    obs_min = np.min(np.vstack([obs_min, obs_n]), axis=0)
    obs_max = np.max(np.vstack([obs_max, obs_n]), axis=0)

    if train:
      t = Transition(s=obs, a=act, r_p=rew, s_p=obs_n, d=term or trunc, a_d=term, te=term, tr=trunc)
      agent.do_update(t)

    ep_rew += rew
    obs = obs_n
    i += 1

    if plotter is not None:
      plotter.plot_step(i, agent)

  agent.end_episode()

  return ep_rew, i, obs_min, obs_max


def run_experiment(env, agent, episodes,
                   obs_scaler=None, rew_scaler=None,
                   train_plotter=None, eval_plotter=None, eval_freq=None, eval_num=5):
  ep_rews = []
  ep_lens = []
  eval_rews = []
  eval_lens = []
  obs_min = np.array([np.inf] * env.observation_space.low.shape[0])
  obs_max = np.array([-np.inf] * env.observation_space.low.shape[0])
  for e in range(episodes):
    ep_rew, t, o_min, o_max = run_episode(env, agent, obs_scaler, rew_scaler, train_plotter)
    ep_rews.append(ep_rew)
    ep_lens.append(t)

    obs_min = np.min(np.vstack([obs_min, o_min]), axis=0)
    obs_max = np.max(np.vstack([obs_max, o_max]), axis=0)

    if train_plotter is not None:
      train_plotter.plot_episode(e, ep_rews, ep_lens, agent)

    print(f'train: ep: {e}, rew: {ep_rew}')

    if eval_freq is not None and e % eval_freq == 0:
      old_eps = agent.eps
      agent.eps = ConstantSchedule(0.)
      env.unwrapped.render_mode = 'human'
      for ev in range(eval_num):
        ep_rew, t, _, _ = run_episode(env, agent, obs_scaler, rew_scaler, eval_plotter, train=False)
        eval_rews.append(ep_rew)
        eval_lens.append(t)

        if eval_plotter is not None:
          eval_plotter.plot_episode(e, eval_rews, eval_lens, agent)

        print(f'eval: ep: {ev}, rew: {ep_rew}')
      agent.eps = old_eps
      env.unwrapped.render_mode = None


def main():
  episodes = 100000
  gamma = 0.99
  alpha = 0.1
  buffer_sz = 1e5
  batch_sz = 64
  min_updates = 1000
  update_freq = 16
  plot_freq_ep = 10

  eps = 0.1
  # eps_sch = ConstantSchedule(eps)
  eps_sch = LinearSchedule(eps, 0.005, 10000)

  # # Mountain car
  # env = gym.make('MountainCar-v0')
  # # env = gym.make('MountainCar-v0', render_mode='human')
  # low, high = env.observation_space.low, env.observation_space.high

  # Cart pole
  env = gym.make('CartPole-v1')
  # env = gym.make('CartPole-v1', render_mode='human')
  low, high = env.observation_space.low, env.observation_space.high
  low[[1, 3]], high[[1, 3]] = -4., 4.
  env.unwrapped.observation_space = gym.spaces.Box(low, high, dtype=env.observation_space.dtype)
  low, high = env.observation_space.low, env.observation_space.high

  # Mountain car
  # tiles = [[2, 2], [4, 4], [8, 8], [16, 16]]
  # tilings = [8, 8, 8, 8]

  # Cart pole
  tiles = [[2, 2, 2, 2], [4, 4, 4, 4], [8, 8, 8, 8], [16, 16, 16, 16]]
  tilings = [16, 16, 16, 16]

  # obs_scaler = lambda o: (o - low) / (high - low)
  # rew_scaler = lambda r: r / 1.
  obs_scaler, rew_scaler = None, None

  # sc_low = np.array([0] * low.shape[0])
  # sc_high = np.array([1] * high.shape[0])
  sc_low, sc_high = low, high

  n_actions = env.action_space.n

  # q_func = TileCoderQFunc(n_actions, sc_low, sc_high, tiles, tilings)
  q_func = TCQFunc(n_actions, sc_low, sc_high, tiles, tilings)

  agent = QAgent(n_actions, eps_sch, gamma, alpha, q_func)

  buffer = ReplayBuffer(max_size=buffer_sz, dtype=np.double)
  agent = BufferAgent(agent, buffer, batch_sz, min_updates, update_freq)

  plotter = Plotter(plot_freq_ep)

  run_experiment(env, agent, episodes, obs_scaler, rew_scaler, plotter)
  plt.show()


def tc_planner_run():
  episodes = 100000
  plot_freq_ep = 10
  budget = 1e6
  gamma = 0.99
  alpha = 0.1

  eps = 0.1
  eps_sch = ConstantSchedule(eps)
  # eps_sch = LinearSchedule(eps, 0.005, 10000)

  # Mountain car
  # env = gym.make('MountainCar-v0')
  # env = gym.make('MountainCar-v0', render_mode='human')

  env = gym.make('Acrobot-v1')

  # env = gym.make('Acrobot-v1', render_mode='human')

  def set_state(model, s):
    model.unwrapped.state = np.hstack([np.arctan2(s[1], s[0]), np.arctan2(s[3], s[2]), s[4:]])

  env = env.env  # Remove time limit TODO: need something here

  n_actions = env.action_space.n
  low, high = env.observation_space.low, env.observation_space.high

  dim = low.shape[0]
  start = 1
  depth = 3
  tiles = [[2 ** t] * dim for t in range(start, depth + 1)]
  tilings = [np.power(2, int(np.ceil(np.log2(4 * dim))))] * len(tiles)
  q_func = TCQFunc(n_actions, low, high, tiles, tilings)
  agent = QAgent(n_actions, eps_sch, gamma, alpha, q_func)

  # n_bins = np.array([30] * env.observation_space.low.shape[0])
  # planner = RrtEx(env, n_bins, budget=budget, constant_reward=True, set_state=set_state)
  # agent = TCPlanCompiler(env, planner, n_actions, -200., 0.)

  plotter = Plotter(plot_freq_ep)

  # env.unwrapped.render_mode = 'human'
  run_experiment(env, agent, episodes, plotter=plotter)

  debug = 0


def acrobot():
  env = gym.make('Acrobot-v1', render_mode='human')

  term = trunc = False
  obs, _ = env.reset()

  while not (term or trunc):
    action = env.action_space.sample()
    obs_n, rew, term, trunc, info = env.step(action)

    obs = obs_n

  debug = 0


def runner():
  episodes = 100000
  gamma = 0.99
  alpha = 0.1
  init_q = 0.
  plot_freq_ep = 10
  buffer_sz = 1e5
  batch_sz = 64
  min_updates = 200
  update_freq = 16
  plot_freq_ep = 10
  eval_freq = 100

  eps = 0.5
  # eps_sch = ConstantSchedule(eps)
  eps_sch = LinearSchedule(eps, 0.005, 5000)

  # Inv Pendulum
  env = gym.make('InvertedPendulum-v4')
  # env = gym.make('InvertedPendulum-v4', render_mode='human')
  # env = env.env  # Remove time limit TODO: need something here
  env.unwrapped.observation_space = gym.spaces.Box(
    np.array([-1.5, -0.5, -4.5, -8.]),
    np.array([1.5, 0.5, 4.5, 8.]),
    dtype=env.observation_space.dtype
  )

  # Reacher
  # env = gym.make('Reacher-v4')
  # # env = gym.make('Reacher-v4', render_mode='human')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]),
  #   np.array([1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]),
  #   dtype=env.observation_space.dtype
  # )

  # Pusher
  # env = gym.make('Pusher-v4')
  # # env = gym.make('Pusher-v4', render_mode='human')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-2.5, -2., -2.5, -3., -2., -2., -2.5, -2., -2.5, -15., -10.,
  #             -15., -10., -15., -1., -2., -1., -1., -1., -0.5, -0.45, -0.5, -0.5]),
  #   np.array([2.5, 2., 2.5, 1., 2., 1., 2.5, 2., 2.5, 15., 10., 15., 10.,
  #             15., 1.5, 1., 1., 1.5, 1., 0.5, 0.45, 0.5, 0.5]),
  #   dtype=env.observation_space.dtype
  # )

  e_low, e_high = env.observation_space.low, env.observation_space.high
  a_low, a_high = env.action_space.low, env.action_space.high
  e_dim, a_dim = e_low.shape[0], a_low.shape[0]

  # obs_max = np.array([-np.inf] * e_dim)
  # obs_min = np.array([np.inf] * e_dim)
  # for _ in range(5000):
  #   term = trunc = False
  #   obs, _ = env.reset()
  #
  #   while not (term or trunc):
  #     action = env.action_space.sample()
  #     obs_n, rew, term, trunc, info = env.step(action)
  #
  #     obs_max = np.max(np.vstack([obs_max, obs_n]), axis=0)
  #     obs_min = np.min(np.vstack([obs_min, obs_n]), axis=0)
  #
  #     obs = obs_n
  #
  # debug = 0

  # inv-pendulum
  start = 1
  depth = 3
  # tiles = [[2 ** t] * (e_dim + a_dim) for t in range(start, depth + 1)]  # TODO: different size for actions
  tiles = [[t] * (e_dim + a_dim) for t in [1, 2, 4, 6]]  # the TC adds 1, so need prime - 1 for true count
  tilings = [min(np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim))))), 64)] * len(tiles)

  # reacher
  # start = 1
  # depth = 3
  # # tiles = [([2 ** t] * (e_dim - 1) + [0] + [2 ** t] * a_dim) for t in
  # #          range(start, depth + 1)]  # TODO: different size for actions
  # tiles = [([t] * (e_dim - 1) + [0] + [t] * a_dim) for t in [1, 2, 4, 6]]  # the TC adds 1, so need prime - 1
  # tilings = [np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim)))))] * len(tiles)

  # pusher
  # start = 0
  # depth = 2
  # tiles = [[2 ** t] * (e_dim + a_dim) for t in range(start, depth + 1)]
  # tilings = [min(np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim))))), 32)] * len(tiles)

  e_lims = np.vstack([e_low, e_high]).T
  e_tiles = [t[:e_dim] for t in tiles]
  o_tc = LayeredTileCoder(e_tiles, e_lims, tilings)
  q_func = ContTCQFunc(e_low, e_high, a_low, a_high, tiles, tilings, init_q=init_q)
  agent = ContQAgent(e_dim, env.action_space, a_low, a_high, eps_sch, gamma, alpha, q_func, o_tc)

  # buffer = ReplayBuffer(max_size=buffer_sz, dtype=np.double)
  # agent = BufferAgent(agent, buffer, batch_sz, min_updates, update_freq)

  t_plotter = Plotter(plot_freq_ep, title='Train')
  e_plotter = Plotter(1, title='Eval')

  # env.unwrapped.render_mode = 'human'
  run_experiment(env, agent, episodes,
                 train_plotter=t_plotter, eval_plotter=e_plotter,
                 eval_freq=eval_freq)
  plt.show()
  debug = 0


def jax_runner():
  episodes = 100000
  gamma = 0.99
  init_q = 0.
  plot_freq_ep = 10
  buffer_sz = 1e5
  batch_sz = 64
  min_updates = 200
  update_freq = 16
  plot_freq_ep = 10
  eval_freq = 100
  obs_scaler = None
  rew_scaler = None

  eps_init = 0.1
  # eps = ConstantSchedule(eps_init)
  # eps = LinearSchedule(eps_init, 0.01, 10000)
  # eps = CosineAnnealingSchedule(0.001, eps_init, 67, 100, end_min=True)
  eps = StepSchedule(eps_init, 0.01, 60, 10, end_second=True)

  alpha_init = 0.1
  alpha = ConstantSchedule(alpha_init)
  # alpha = CosineAnnealingSchedule(0.01, alpha_init, 31, 100, end_min=True)
  # alpha = StepSchedule(0.03, alpha_init, 60, 10, end_second=False)

  # Inv Pendulum
  env = gym.make('InvertedPendulum-v4')
  # env = gym.make('InvertedPendulum-v4', render_mode='human')
  # env = env.env  # Remove time limit TODO: need something here
  env.unwrapped.observation_space = gym.spaces.Box(
    np.array([-1.5, -0.5, -4.5, -8.]),
    np.array([1.5, 0.5, 4.5, 8.]),
    dtype=env.observation_space.dtype
  )

  # Reacher
  # env = gym.make('Reacher-v4')
  # # env = gym.make('Reacher-v4', render_mode='human')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]),
  #   np.array([1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]),
  #   dtype=env.observation_space.dtype
  # )

  # # remove target x, y
  # e_low, e_high = env.observation_space.low, env.observation_space.high
  # e_low = np.hstack([e_low[:4], e_low[6:]])
  # e_high = np.hstack([e_high[:4], e_high[6:]])
  # obs_scaler = lambda o: np.hstack([o[:4], o[6:]])
  # env.observation_space.low, env.observation_space.high = e_low, e_high

  # Pusher
  # env = gym.make('Pusher-v4')
  # # env = gym.make('Pusher-v4', render_mode='human')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-2.5, -2., -2.5, -3., -2., -2., -2.5, -2., -2.5, -15., -10.,
  #             -15., -10., -15., -1., -2., -1., -1., -1., -0.5, -0.45, -0.5, -0.5]),
  #   np.array([2.5, 2., 2.5, 1., 2., 1., 2.5, 2., 2.5, 15., 10., 15., 10.,
  #             15., 1.5, 1., 1., 1.5, 1., 0.5, 0.45, 0.5, 0.5]),
  #   dtype=env.observation_space.dtype
  # )

  e_low, e_high = env.observation_space.low, env.observation_space.high
  a_low, a_high = env.action_space.low, env.action_space.high
  e_dim, a_dim = e_low.shape[0], a_low.shape[0]

  # inv-pendulum
  start = 1
  depth = 3
  # tiles = [[2 ** t] * (e_dim + a_dim) for t in range(start, depth + 1)]  # TODO: different size for actions
  tiles = [[t] * (e_dim + a_dim) for t in [1, 2, 4, 6]]  # the TC adds 1, so need prime - 1 for true count
  tilings = [min(np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim))))), 64)] * len(tiles)

  # reacher
  # start = 1
  # depth = 3
  # # tiles = [([2 ** t] * (e_dim - 1) + [0] + [2 ** t] * a_dim) for t in
  # #          range(start, depth + 1)]  # TODO: different size for actions
  # tiles = [([t] * (e_dim - 1) + [0] + [t] * a_dim) for t in [1, 2, 4, 6]]  # the TC adds 1, so need prime - 1
  # tilings = [np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim)))))] * len(tiles)

  # pusher
  # start = 1
  # depth = 3
  # # tiles = [[2 ** t] * (e_dim + a_dim) for t in range(start, depth + 1)]
  # tiles = [[t] * (e_dim + a_dim) for t in [1, 2, 4, 6]]  # the TC adds 1, so need prime - 1
  # tilings = [np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim)))))] * len(tiles)

  e_tiles = [t[:e_dim] for t in tiles]
  a_tiles = [t[-a_dim:] for t in tiles]

  obs_tc = JaxLayeredTileCoder(e_tiles, tilings, e_low, e_high)
  act_tc = JaxLayeredTileCoder(a_tiles, tilings, a_low, a_high)

  # with JaxContTCTabQFunc(e_low, e_high, a_low, a_high, tiles, tilings, init_q=init_q) as q_func:
  # with JaxContTCQFunc(e_low, e_high, a_low, a_high, tiles, tilings, init_q=init_q) as q_func:
  # with MultiJaxContTCQFunc(e_low, e_high, a_low, a_high, tiles, tilings, init_q=init_q) as q_func:

  # agent = JaxContQAgent(env.action_space, a_low, a_high, e_low, e_high, eps, gamma, alpha, q_func, obs_tc)

  n_acts = 10
  init_p = lambda: np.vstack([env.action_space.sample() for _ in range(n_acts)])
  q_func = ProperTCQFunc(obs_tc, act_tc, init_q=init_q)
  p_func = ProperTCPFunc(obs_tc, n_acts, a_dim, init_p=init_p)
  agent = ProperTCQAgent(env.action_space, a_low, a_high,
                         env.observation_space, e_low, e_high,
                         eps, gamma, alpha,
                         q_func, p_func,
                         obs_tc, act_tc)

  buffer = ReplayBuffer(max_size=buffer_sz, dtype=np.double)
  agent = BufferAgent(agent, buffer, batch_sz, min_updates, update_freq)

  t_plotter = Plotter(plot_freq_ep, title='Train')
  e_plotter = Plotter(1, title='Eval')
  # t_plotter = None
  # e_plotter = None

  # env.unwrapped.render_mode = 'human'

  run_experiment(env, agent, episodes,
                 train_plotter=t_plotter, eval_plotter=e_plotter,
                 obs_scaler=obs_scaler, rew_scaler=rew_scaler,
                 eval_freq=eval_freq)
  plt.show()
  debug = 0


def obs_test():
  # env = gym.make('Reacher-v4')
  # # env = gym.make('Reacher-v4', render_mode='human')
  # e_low, e_high = env.observation_space.low, env.observation_space.high
  # e_low[:] = [-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]
  # e_high[:] = [1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]

  env = gym.make('HalfCheetah-v4')
  env.unwrapped.observation_space = gym.spaces.Box(
    np.array([-2., -6., -2., -2., -2., -2., -2., -2., -6., -6., -15., -50., -50., -50., -50., -50., -50.]),
    np.array([2., 6., 2., 2., 2., 2., 2., 2., 6., 6., 15., 50., 50., 50., 50., 50., 50.]),
    dtype=env.observation_space.dtype
  )

  # env = gym.make('Walker2d-v4')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-1., -5., -5., -5., -5., -5., -5., -5., -10., -10., -10., -10., -10., -10., -10., -10., -10.]),
  #   np.array([5., 5., 5., 5., 5., 5., 5., 5., 10., 10., 10., 10., 10., 10., 10., 10., 10.]),
  #   dtype=env.observation_space.dtype
  # )

  # env = gym.make('Pusher-v4')
  # env = gym.make('Pusher-v4', render_mode='human')
  # env.unwrapped.observation_space = gym.spaces.Box(
  #   np.array([-2.5, -2., -2.5, -3., -2., -2., -2.5, -2., -2.5, -15., -10.,
  #             -15., -10., -15., -1., -2., -1., -1., -1., -0.5, -0.45, -0.5, -0.5]),
  #   np.array([2.5, 2., 2.5, 1., 2., 1., 2.5, 2., 2.5, 15., 10., 15., 10.,
  #             15., 1.5, 1., 1., 1.5, 1., 0.5, 0.45, 0.5, 0.5]),
  #   dtype=env.observation_space.dtype
  # )
  e_low, e_high = env.observation_space.low, env.observation_space.high

  # e_low[:] = [-1., -1., -1., -1., -1., -1., -100., -100., -1., -1., -0.1]
  # e_high[:] = [1., 1., 1., 1., 1., 1., 100., 100., 1., 1., 0.1]

  a_low, a_high = env.action_space.low, env.action_space.high
  e_dim, a_dim = e_low.shape[0], a_low.shape[0]

  obs_min = np.array([np.inf] * e_dim)
  obs_max = np.array([-np.inf] * e_dim)
  for _ in range(1000):
    term = trunc = False
    obs, _ = env.reset()

    while not (term or trunc):
      action = env.action_space.sample()
      obs_n, rew, term, trunc, info = env.step(action)

      obs_min = np.min(np.vstack([obs_min, obs_n]), axis=0)
      obs_max = np.max(np.vstack([obs_max, obs_n]), axis=0)

      obs = obs_n

  print(f'obs_min: {obs_min}, obs_max: {obs_max}')

  debug = 0


def tc_playing():
  tiles_per_dim = [1, 1, 1]
  tilings = 8

  low = np.array([0., 0., 0.])
  high = np.array([1., 1., 1.])
  lims = np.vstack([low, high]).T

  tc = TileCoder(tiles_per_dim, lims, tilings)

  x_y = np.meshgrid(np.linspace(-0.5, 1.5, 2001), np.linspace(-0.5, 1.5, 2001), indexing='ij')
  x = np.dstack(x_y).reshape(-1, 2)

  tiles = np.array([tc[o] for o in x])
  u_tiles = np.unique(tiles, axis=0)

  centers = np.array([(np.max(y, axis=0) + np.min(y, axis=0)) / 2. for y in
                      [x[np.where(np.all(tiles == t, axis=1))[0]] for t in u_tiles]])
  plt.figure()
  plt.scatter(centers[:, 0], centers[:, 1])

  corners = np.array([[np.min(y[:, 0]), np.max(y[:, 0]), np.min(y[:, 1]), np.max(y[:, 1])] for y in
                      [x[np.where(np.all(tiles == t, axis=1))[0]] for t in u_tiles]])
  plt.figure()
  c = {(0, 0): 'r', (0, 1): 'g', (1, 0): 'b', (1, 1): 'y'}
  for c_x in range(2):
    for c_y in range(2):
      plt.scatter(corners[:, c_x], corners[:, c_y + 2], c=c[(c_x, c_y)], s=6)

  plt.show()
  debug = 0


if __name__ == '__main__':
  # main()
  # tc_planner_run()
  # acrobot()
  # runner()
  # jax_runner()
  obs_test()
  # tc_playing()
