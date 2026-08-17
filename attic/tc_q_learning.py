from collections import defaultdict

import functools
from time import perf_counter

import numpy as np
from scipy.stats import qmc

from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.schedules import LinearSchedule, Schedule
from plan_compilation_framework.helpers.tile_coder import TileCoder, LayeredTileCoder


class TCQFunc:
  def __init__(self, n_actions, low, high, tiles, tilings, init_q=0.):
    self.actions = list(range(n_actions))

    lims = np.vstack([low, high]).T
    self.tc = LayeredTileCoder(tiles, lims, tilings)

    init_q /= self.tc.n_layers

    self.q = {
      a: [np.ones(n_t) * init_q * t_w for n_t, t_w in zip(self.tc.n_tiles, self.tc.t_weights)] for a in self.actions
    }

  def values(self, obs):
    q_values = np.zeros((obs.shape[0], len(self.actions)))

    for i, o in enumerate(obs):
      for a in self.actions:
        for l, (tiles, tw, tl) in enumerate(self.tc[o]):
          q_values[i][a] += self.q[a][l][tiles].sum()

    return q_values

  def update(self, obs, act, target, lr):
    for i, (o, a, t) in enumerate(zip(obs, act, target)):
      for l, (tiles, tw, tl) in enumerate(self.tc[o]):
        self.q[a][l][tiles] += lr * t * tw * tl


class ContTCQFunc:
  def __init__(self, env_low, env_high, act_low, act_high, tiles, tilings, init_q=0.):
    lims = np.vstack([np.hstack([env_low, act_low]), np.hstack([env_high, act_high])]).T
    self.tc = LayeredTileCoder(tiles, lims, tilings)

    self.q = {
      l: defaultdict(lambda: defaultdict(lambda: init_q * t_w * l_w))
      for l, t_w, l_w in zip(range(self.tc.n_layers), self.tc.t_weights, self.tc.l_weights)}

    debug = 0

  def values(self, obs_act):
    q_values = np.zeros(obs_act.shape[0])

    t1 = perf_counter()

    for i, o_a in enumerate(obs_act):
      for l, (tiles, tw, lw) in enumerate(self.tc[o_a]):
        for i_t, t in enumerate(tiles):
          q_values[i] += self.q[l][i_t][t]

    t2 = perf_counter()

    first = t2 - t1

    return q_values

  def update(self, obs_act, target, lr):
    for i, (o_a, tgt) in enumerate(zip(obs_act, target)):
      for l, (tiles, t_w, l_w) in enumerate(self.tc[o_a]):
        for i_t, t in enumerate(tiles):
          self.q[l][i_t][t] += lr * tgt * t_w * l_w

    debug = 1

class TCLFunc:
  def __init__(self, tc, init_l=0.):
    self.tc = tc

    init_l /= self.tc.n_layers

    self.l = [np.ones(n_t) * init_l for n_t in self.tc.n_tiles]

  def values(self, obs):
    l_values = np.zeros(obs.shape[0])

    for i, o in enumerate(obs):
      for l, (tiles, tw, tl) in enumerate(self.tc[o]):
        l_values[i] += self.l[l][tiles].sum()

    return l_values

  def update(self, obs, target, lr):
    for i, (o, t) in enumerate(zip(obs, target)):
      for l, (tiles, tw, tl) in enumerate(self.tc[o]):
        self.l[l][tiles] += lr * t * tw * tl


class QAgent:
  def __init__(self, n_actions, eps: Schedule, gamma, alpha, q_func):
    self.actions = list(range(n_actions))
    self.eps = eps
    self.gamma = gamma
    self.alpha = alpha

    self.q = q_func

  def begin_episode(self):
    pass

  def end_episode(self):
    self.eps.update()

  def get_q_values(self, obs):
    q_values = self.q.values(obs)
    return q_values

  def get_values(self, obs):
    return np.max(self.get_q_values(obs), axis=1)

  def get_action(self, obs):
    # inefficient - used for debugging
    q_values = self.get_q_values(obs[None, :]).squeeze()
    action = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])

    if np.random.random() < self.eps.value:
      action = np.random.choice(self.actions)

    return action.item()

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done = t.s, int(t.a), t.r_p, t.s_p, t.a_d

    q_next = self.get_values(obs_n[None, :]).item()
    q_curr = self.get_q_values(obs[None, :])[0, act]

    target = rew + (1. - done) * self.gamma * q_next - q_curr
    self.q.update([obs], [act], [target], self.alpha)


class ContQAgent:
  def __init__(self, obs_dim, act_sp, act_low, act_high, eps: Schedule, gamma, alpha, q_func, o_tc):
    self.obs_dim = obs_dim
    self.act_dim = act_low.shape[0]
    self.act_sp = act_sp
    self.act_low = act_low
    self.act_high = act_high

    self.eps = eps

    self.gamma = gamma
    self.alpha = alpha

    self.q = q_func

    self.samples = 100
    self.sampler = qmc.LatinHypercube(d=self.act_dim)

    self.o_tc = o_tc
    self.o_acts = {
      l: defaultdict(lambda: None)
      for l in range(o_tc.n_layers)
    }

    self.n_max = 50
    self.n_l = [int(self.n_max / (l + 1)) for l in range(self.o_tc.n_layers)]
    # self.o_acts = {
    #   l: defaultdict(lambda: np.vstack([self.act_sp.sample() for _ in range(self.n_max)]))
    #   for l in range(o_tc.n_layers)
    # }
    # self.o_acts = defaultdict(lambda: np.vstack([self.act_sp.sample() for _ in range(self.n_max)]))

  def begin_episode(self):
    pass

  def end_episode(self):
    self.eps.update()

  def get_q_values(self, obs, act=None):
    if act is None:
      sample = self.sampler.random(n=self.samples)
      sample = qmc.scale(sample, self.act_low, self.act_high)

      obs_act = np.hstack([np.tile(obs, (sample.shape[0], 1)), sample])

      o_tiles = [tuple(t) for t, _, _ in self.o_tc[obs]]
      for l in range(self.o_tc.n_layers):
        o_t = sum(o_tiles[:l+1], ())
        o_acts = self.o_acts[l][o_t]
        if o_acts is None:
          self.o_acts[l][o_t] = o_acts = np.hstack([
            np.tile(obs, (self.n_l[l], 1)),
            np.vstack([self.act_sp.sample() for _ in range(self.n_l[l])])
          ])

        o_a = np.hstack([np.tile(obs, (o_acts.shape[0], 1)), o_acts[:, -self.act_dim:]])
        obs_act = np.vstack([obs_act, o_a])
        debug = 0

      q_values = self.q.values(obs_act)

      all_q_values = q_values.copy()
      all_obs_act = obs_act.copy()

      for l in reversed(range(self.o_tc.n_layers)):
        o_t = sum(o_tiles[:l + 1], ())
        o_act = self.o_acts[l][o_t]
        old_q_values = self.q.values(o_act)

        all_q_values = np.hstack([all_q_values, old_q_values])
        all_obs_act = np.vstack([all_obs_act, o_act])

        m_inds = np.argpartition(all_q_values, -self.n_l[l])[-self.n_l[l]:]
        self.o_acts[l][o_t] = all_obs_act[m_inds, :]
    else:
      obs_act = np.atleast_2d(np.hstack([obs, act]))
      q_values = self.q.values(obs_act)

    return q_values, obs_act

  # def get_q_values(self, obs, act=None):
  #   if act is None:
  #     sample = self.sampler.random(n=self.samples)
  #     sample = qmc.scale(sample, self.act_low, self.act_high)
  #
  #     o_tiles = [tuple(t) for t, _, _ in self.o_tc[obs]]
  #     for l in range(self.o_tc.n_layers):
  #       o_t = sum(o_tiles[:l+1], ())
  #       o_acts = self.o_acts[l][o_t]
  #       sample = np.vstack([sample, o_acts])
  #
  #     obs_act = np.hstack([np.tile(obs, (sample.shape[0], 1)), sample])
  #
  #     q_values = self.q.values(obs_act)
  #
  #     prop = 1.
  #     for l in reversed(range(self.o_tc.n_layers)):
  #       n_count = int(self.n_max / prop)
  #       m_inds = np.argpartition(q_values, -n_count)[-self.n_max:]
  #       o_t = sum(o_tiles[:l + 1], ())
  #       self.o_acts[l][o_t] = sample[m_inds, :]
  #   else:
  #     obs_act = np.atleast_2d(np.hstack([obs, act]))
  #     q_values = self.q.values(obs_act)
  #
  #   return q_values, obs_act

  def get_action(self, obs):
    q_values, obs_act = self.get_q_values(obs[None, :])
    action_idx = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])
    action = obs_act[action_idx, -self.act_dim:]

    if np.random.random() < self.eps.value:
      action = self.act_sp.sample()

    return action

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done = t.s, t.a, t.r_p, t.s_p, t.a_d

    q_next = np.max(self.get_q_values(obs_n[None, :])[0])
    q_curr, obs_act = self.get_q_values(obs, act)

    target = rew + (1. - done) * self.gamma * q_next - q_curr
    self.q.update(obs_act, target, self.alpha)


class BufferAgent:
  def __init__(self, agent, buffer, batch_sz, min_updates, update_freq):
    self.agent = agent
    self.buffer = buffer
    self.batch_sz = batch_sz
    self.min_updates = min_updates
    self.update_freq = update_freq
    self.updates = 0

  def __getattr__(self, item):
    if 'agent' in self.__dict__ and hasattr(self.__dict__['agent'], item):
      return getattr(self.__dict__['agent'], item)
    return self.__dict__[item]

  def __setattr__(self, key, value):
    if 'agent' in self.__dict__ and hasattr(self.__dict__['agent'], key):
      return setattr(self.__dict__['agent'], key, value)
    self.__dict__[key] = value

  def begin_episode(self):
    self.agent.begin_episode()

  def end_episode(self):
    self.agent.end_episode()

  def get_action(self, obs):
    return self.agent.get_action(obs)

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done = t.s, t.a, t.r_p, t.s_p, t.a_d
    self.buffer.store(dict(obs=obs, act=act, rew=rew, obs_n=obs_n, done=done))

    if self.buffer.curr_size > self.min_updates and self.updates % self.update_freq == 0:
      batch = self.buffer.sample(self.batch_sz)

      for o, a, r, o_n, d in zip(batch['obs'], batch['act'], batch['rew'], batch['obs_n'], batch['done']):
        # TODO: implement transition buffer
        self.agent.do_update(Transition(s=o, a=a, r_p=r, s_p=o_n, d=d, a_d=d))

    self.updates += 1


def multi_test():
  l = 4

  q = defaultdict(lambda: defaultdict(lambda: 0.))

  lims = np.vstack([[0, 0], [1, 1]]).T
  tiles = [4, 4]
  tc = TileCoder(tiles, lims, 4)
  l_tc = TileCoder([tiles, tiles], lims, [4, 6])

  x = np.random.uniform(size=(10, 2))

  tile = tc[np.array([0.2, 0.6])]
  tiles = tc[x]

  l_tile = tc[np.array([0.2])]
  debug = 0


if __name__ == '__main__':
    multi_test()
