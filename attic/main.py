import gym
import imageio
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from gym.wrappers import Monitor, RecordVideo
from matplotlib import cm
import plan_compilation_framework.environments
from plan_compilation_framework.agents.plan_compiler import PlanCompiler
from plan_compilation_framework.agents.q_planner import QPlanner
from plan_compilation_framework.agents.qlearning import QLearning
from plan_compilation_framework.environments import StateFeaturesWrapper
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.planners.astar import AStar
from plan_compilation_framework.planners.rrt_ex import RrtEx

matplotlib.use('TkAgg')

FOLDER_MAP = {
  'images/q_optimistic'         : 'Q-Learning (Optimistic)',
  'images/q_pessimistic'        : 'Q-Learning (Pessimistic)',
  'images/q_planner'            : 'Q-Planner (Optimistic)',
  'images/q_planner_pessimistic': 'Q-Planner (Pessimistic)',
  'images/pc_learner'           : 'PLan Compiler',
  'images/pc_explorer'          : 'Plan Compiler ($Q_{exp}$)',
  'images/temp'                 : 'Temp',
}


class ValuePlotter:
  def __init__(self, shape, zlim=None, fig_sz=(10, 10), folder='images/value'):
    assert zlim is not None, 'Need zlim for plotting'
    self.folder = folder
    self.zlim = zlim
    self.fig, self.ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=fig_sz)
    self.i, self.j = np.meshgrid(np.arange(1, shape[0] - 1), np.arange(1, shape[1] - 1), indexing='ij')
    self.states = np.stack([self.i.reshape(-1), self.j.reshape(-1)]).T

    self.scale = 10
    self.I, self.J = np.meshgrid(np.arange(*np.array([1, shape[0] - 1]) * self.scale),
                                 np.arange(*np.array([1, shape[1] - 1]) * self.scale),
                                 indexing='ij')

  def plot(self, agent, epoch):
    assert hasattr(agent, 'get_values'), "Can't get_values() from Agent"

    values = agent.get_values(self.states)
    values = values.reshape(self.i.shape)

    r, c = self.i.shape
    values_ = np.empty((r, self.scale, c, self.scale), values.dtype)
    values_[...] = values[:, None, :, None]
    values_ = values_.reshape(r * self.scale, c * self.scale)

    self.ax.clear()
    self.ax.plot_surface(self.I, self.J, values_,
                         cmap=cm.coolwarm, linewidth=0, antialiased=False,
                         rcount=np.min([self.I.shape[0], 100]), ccount=np.min([self.I.shape[1], 100]))
    self.ax.view_init(elev=29, azim=-33)
    self.ax.set_xlabel('Y')
    self.ax.set_ylabel('X')
    self.ax.set_xticklabels([])
    self.ax.set_yticklabels([])
    self.ax.set_zlim(self.zlim)
    self.ax.set_title(f'Value Function - {FOLDER_MAP[self.folder]}\nEpisode: {epoch: 4}')
    self.fig.savefig(f'{self.folder}/value_{epoch:04}.png')


def run_episodes(env, agent, episodes=10000, render=True, plot_value=False, zlim=None, folder=None):
  ep_rewards = []

  val_plotter = None

  for e in range(episodes):
    obs = env.reset()

    done = False
    agent.begin_episode()
    ep_rew = 0.
    step = 0

    # plan = planner.get_plan(obs)

    while not done:

      if render:
        env.render()
        if e < 20:
          rgb = env.render(mode='rgb_array')
          imageio.imwrite(f'{folder}/e_{e:03}_step_{step:03}.png', rgb)

      action = agent.get_action(obs)
      # action = plan.next_action()

      n_obs, rew, done, info = env.step(action)
      actually_done = done and not info.get('TimeLimit.truncated', False)

      t = Transition(s=obs, a=action, r_p=rew, s_p=n_obs, d=done, a_d=actually_done)
      agent.do_update(t)

      if actually_done:
        t = Transition(s=n_obs, a=0, r_p=0, s_p=n_obs, d=done, a_d=actually_done)
        agent.do_update(t)

      obs = n_obs
      ep_rew += rew
      step += 1

    agent.end_episode()
    ep_rewards.append(ep_rew)
    print(f'e: {e: 5}, rew: {ep_rew}')

    if plot_value:  # and e % 20 == 0:
      if val_plotter is None:
        val_plotter = ValuePlotter(env.observation_space.high, zlim=zlim, folder=folder)
      val_plotter.plot(agent, epoch=e)
      plt.show(block=False)
      plt.pause(0.1)

  plt.plot(ep_rewards)
  plt.show()
  debug = 0


def gridworld():
  # env = gym.make('gridworld-default-v0', seed=0)
  # env = gym.make('gridworld-bombs-v0', seed=0)
  # env = gym.make('gridworld-bombs-lots-v0', seed=0)
  # env = gym.make('gridworld-random-20x20-v0', seed=18)
  # env = gym.make('gridworld-random-20x20-stoch-v0', seed=0)
  # env = gym.make('gridworld-random-50x50-v0', seed=0)
  # env = gym.make('gridworld-random-50x50-stoch-v0', seed=0)
  env = gym.make('gridworld-custom-v0', seed=0)

  env = StateFeaturesWrapper(env)
  # env = RecordVideo(env, 'videos', name_prefix='gridworld-q')

  # folder = 'images/q_optimistic'
  # folder = 'images/q_pessimistic'
  # folder = 'images/q_planner'
  # folder = 'images/q_planner_pessimistic'
  folder = 'images/pc_learner'
  # folder = 'images/pc_explorer'
  # folder = 'images/temp'

  planner = AStar(env, constant_reward=True)
  # agent = QLearning(env, env.action_space.n, default_q=0.)
  # agent = QLearning(env, env.action_space.n, default_q=-20.)
  # agent = QPlanner(env, planner, env.action_space.n, default_q=0.)
  # agent = QPlanner(env, planner, env.action_space.n, default_q=-20.)
  agent = PlanCompiler(env, planner, env.action_space.n, -20., 0.)

  run_episodes(env, agent, episodes=1000, render=True, plot_value=True, zlim=[-20, 0], folder=folder)

  debug = 0


def mountain_car():
  env = gym.make('MountainCar-v0')
  env = env.unwrapped  # no time limit
  env.seed(0)

  n_bins = np.array([100, 100])
  obs_sp = env.observation_space
  scale = n_bins / (obs_sp.high - obs_sp.low)

  def get_bins(s):
    return tuple(np.round((np.array(s) - obs_sp.low) * scale, 6).astype(int))

  env = StateFeaturesWrapper(env, state_features=get_bins)

  agent = QLearning(env, env.action_space.n)
  # planner = RrtEx(env, n_bins, constant_reward=True)
  # agent = PlanCompiler(env, planner, env.action_space.n, -1000., 0.)

  run_episodes(env, agent, episodes=10000, render=False)

  debug = 0


if __name__ == '__main__':
  gridworld()
  # mountain_car()
