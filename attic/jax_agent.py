from multiprocessing import Pool, freeze_support, Process, Queue, Pipe
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Any

# import jax
import numpy as np
# import jax.numpy as jnp
from functools import partial
from collections import defaultdict
from scipy.stats import qmc
from scipy.sparse import dok_array
from time import perf_counter

import matplotlib
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.schedules import Schedule
from plan_compilation_framework.helpers.tile_coder import TileCoder

matplotlib.use('TkAgg')


# @jax.jit
def compute_tiles(x, low, scaler, offsets, base_ind, hash_vec):
  x = np.expand_dims(np.atleast_2d(x), axis=1)
  off_coords = ((x - low) * scaler + offsets).astype(int)
  return base_ind + np.dot(off_coords, hash_vec)


class JaxTileCoder:
  def __init__(self, tiles, tilings, low, high, offset=lambda n: 2 * np.arange(n) + 1):
    n_dims = len(tiles)
    tiles = np.array(tiles)
    padded_tiles = np.array(np.ceil(tiles), dtype=int) + 1

    offsets = offset(n_dims) * np.repeat(np.arange(tilings)[None, :], n_dims, axis=0).T
    self.offsets = (offsets / float(tilings)) % 1

    self.low = np.array(low)
    self.scaler = tiles / (high - low)
    self.base_ind = np.prod(padded_tiles) * np.arange(tilings)
    self.hash_vec = np.array([np.prod(padded_tiles[:i]) for i in range(n_dims)])
    self.n_tiles = tilings.astype(np.uint64) * np.prod(padded_tiles.astype(np.uint64))

  def __getitem__(self, x):
    return compute_tiles(x, self.low, self.scaler, self.offsets, self.base_ind, self.hash_vec)


class JaxLayeredTileCoder:
  def __init__(self, tiles, tilings, low, high, offset=lambda n: 2 * np.arange(n) + 1):
    assert len(tiles) == len(tilings)
    assert len(tiles[0]) == low.shape[0]

    self.tilings = tilings
    self.n_layers = len(tilings)
    self.w_tiles = 1. / np.array(tilings)
    self.w_layers = np.ones(self.n_layers) * 1. / self.n_layers

    self.tc = []
    self.n_tiles = []
    for t, ti in zip(tiles, tilings):
      tc = JaxTileCoder(t, ti, low, high, offset)
      self.tc.append(tc)
      self.n_tiles.append(tc.n_tiles)

  def __getitem__(self, x):
    return [tc[x] for tc in self.tc]


class JaxContTCTabQFunc:
  def __init__(self, env_low, env_high, act_low, act_high, tiles, tilings, init_q=0.):
    low = np.hstack([env_low, act_low])
    high = np.hstack([env_high, act_high])

    self.tc = JaxLayeredTileCoder(tiles, tilings, low, high)

    self.q = [(dok_array((1, n_t)) if n_t * 8 > 1073741824 else np.zeros((1, n_t)))
              for n_t in self.tc.n_tiles]

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def values(self, obs_act):
    q_values = np.zeros(obs_act.shape[0])

    t1 = perf_counter()

    l_tiles = self.tc[obs_act]

    t2 = perf_counter()

    for l, tiles in enumerate(l_tiles):
      q_values += np.sum(self.q[l][[0], tiles], axis=1)

    t3 = perf_counter()

    # print(f'first: {t2 - t1}, second: {t3 - t2}, third: {t3 - t1}')

    return q_values

  def update(self, obs_act, delta, lr):
    l_tiles = self.tc[obs_act]
    for l, (tiles, w_t, w_l) in enumerate(zip(l_tiles, self.tc.w_tiles, self.tc.w_layers)):
      self.q[l][[0], tiles] += lr * delta * w_t * w_l


class JaxContTCQFunc:
  def __init__(self, env_low, env_high, act_low, act_high, tiles, tilings, init_q=0.):
    low = np.hstack([env_low, act_low])
    high = np.hstack([env_high, act_high])

    self.tc = JaxLayeredTileCoder(tiles, tilings, low, high)

    self.q = {
      l: defaultdict(lambda: defaultdict(lambda: init_q * w_t * w_l))
      for l, w_t, w_l in zip(range(self.tc.n_layers), self.tc.w_tiles, self.tc.w_layers)
    }

    # self.pool = Pool(self.tc.n_layers)

  # def compute_value(self, l, tiles):
  #   q_values = np.zeros(tiles.shape[0])
  #   for i, obs_tiles in enumerate(tiles):
  #     for ti, t in enumerate(obs_tiles):
  #       q_values[i] += self.q[l][ti][t]
  #   return q_values

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    pass

  def values(self, obs_act):
    q_values = np.zeros(obs_act.shape[0])

    t1 = perf_counter()

    l_tiles = self.tc[obs_act]

    # with self.pool:
    #   results = self.pool.starmap(self.compute_value, enumerate(l_tiles))

    t2 = perf_counter()

    for l, tiles in enumerate(l_tiles):
      for i, obs_tiles in enumerate(tiles):
        for ti, t in enumerate(obs_tiles):
          q_values[i] += self.q[l][ti][t]

    t3 = perf_counter()

    # print(f'first: {t2 - t1}, second: {t3 - t2}, third: {t3 - t1}')

    return q_values

  def update(self, obs_act, delta, lr):
    l_tiles = self.tc[obs_act]
    for l, (tiles, w_t, w_l) in enumerate(zip(l_tiles, self.tc.w_tiles, self.tc.w_layers)):
      for i, (obs_tiles, d) in enumerate(zip(tiles, delta)):
        for ti, t in enumerate(obs_tiles):
          self.q[l][ti][t] += lr * d * w_t * w_l


class QProcessor(Process):
  def __init__(self, n_t, init_q, weight, send_queue, tiles_mem, buffer_mem, tiles_shape, buffer_shape):
    super().__init__(daemon=True)
    self.queue = Queue(maxsize=1)
    self.send_queue = send_queue

    self.q: Dict[int, Dict[int, float]] = defaultdict(lambda: defaultdict(lambda: init_q * weight))
    # self.q = dok_array((1, n_t)) if n_t * 8 > 1073741824 else np.zeros((1, n_t))

    self.weight = weight

    self.tiles_mem = tiles_mem
    self.buffer_mem = buffer_mem
    self.tiles_shape = tiles_shape
    self.buffer_shape = buffer_shape

  def run(self) -> None:
    tiles = np.ndarray(self.tiles_shape, dtype=np.uint64, buffer=self.tiles_mem.buf)
    buffer = np.ndarray(self.buffer_shape, dtype=np.double, buffer=self.buffer_mem.buf)

    while True:
      update, inds, lr = self.queue.get()

      if update:
        for i, obs_tiles in enumerate(tiles[:inds]):
          for ti, t in enumerate(obs_tiles):
            self.q[ti][t] += lr * buffer[i] * self.weight

        # self.q[[0], tiles[:inds]] += lr * buffer[:inds] * self.weight

      else:
        buffer[:inds] = 0.
        for i, obs_tiles in enumerate(tiles[:inds]):
          for ti, t in enumerate(obs_tiles):
            buffer[i] += self.q[ti][t]

        # buffer[:inds] = np.sum(self.q[[0], tiles[:inds]], axis=1)

      self.send_queue.put(True, block=True)


class MultiJaxContTCQFunc:
  def __init__(self, env_low, env_high, act_low, act_high, tiles, tilings, init_q=0.):
    low = np.hstack([env_low, act_low])
    high = np.hstack([env_high, act_high])

    self.tc = JaxLayeredTileCoder(tiles, tilings, low, high)

    self.layers = {l: {} for l in range(self.tc.n_layers)}
    for l, n_t, ti, w_t, w_l in zip(range(self.tc.n_layers), self.tc.n_tiles,
                                    self.tc.tilings, self.tc.w_tiles, self.tc.w_layers):
      tiles_shape = (1000, ti)
      buffer_shape = (1000,)
      self.layers[l]['tiles_mem'] = SharedMemory(create=True, size=np.prod(tiles_shape) * 8)
      self.layers[l]['buffer_mem'] = SharedMemory(create=True, size=np.prod(tiles_shape) * 8)
      self.layers[l]['tiles'] = np.ndarray(tiles_shape, dtype=np.uint64, buffer=self.layers[l]['tiles_mem'].buf)
      self.layers[l]['buffer'] = np.ndarray(buffer_shape, dtype=np.double, buffer=self.layers[l]['buffer_mem'].buf)
      self.layers[l]['queue'] = Queue(maxsize=1)
      self.layers[l]['proc'] = QProcessor(n_t, init_q, w_t * w_l, self.layers[l]['queue'],
                                          tiles_mem=self.layers[l]['tiles_mem'], tiles_shape=tiles_shape,
                                          buffer_mem=self.layers[l]['buffer_mem'], buffer_shape=buffer_shape)

      self.layers[l]['proc'].start()

  def __enter__(self):
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    for l in range(self.tc.n_layers):
      self.layers[l]['tiles_mem'].close()
      self.layers[l]['tiles_mem'].unlink()
      self.layers[l]['buffer_mem'].close()
      self.layers[l]['buffer_mem'].unlink()

  def values(self, obs_act):
    t1 = perf_counter()

    l_tiles = self.tc[obs_act]

    t2 = perf_counter()

    n_obs = obs_act.shape[0]

    for l, tiles in enumerate(l_tiles):
      self.layers[l]['tiles'][:n_obs, :tiles.shape[1]] = tiles
      self.layers[l]['proc'].queue.put((False, n_obs, None))

    q_values = []
    for l in range(self.tc.n_layers):
      self.layers[l]['queue'].get()
      q_values.append(self.layers[l]['buffer'][:n_obs])

    q_values = np.vstack(q_values).sum(axis=0)

    t3 = perf_counter()

    # print(f'first: {t2 - t1}, second: {t3 - t2}, third: {t3 - t1}')

    return q_values.copy()

  def update(self, obs_act, delta, lr):
    l_tiles = self.tc[obs_act]

    n_obs = obs_act.shape[0]

    for l, tiles in enumerate(l_tiles):
      self.layers[l]['tiles'][:n_obs, :tiles.shape[1]] = tiles
      self.layers[l]['buffer'][:n_obs] = delta
      self.layers[l]['proc'].queue.put((True, n_obs, lr))

    for l in range(self.tc.n_layers):
      self.layers[l]['queue'].get()


class JaxContQAgent:
  def __init__(self, act_sp, act_low, act_high,
               env_low, env_high,
               eps: Schedule, gamma, alpha,
               q_func: JaxContTCQFunc | MultiJaxContTCQFunc,
               obs_tc: JaxLayeredTileCoder):
    self.act_dim = act_low.shape[0]
    self.act_sp = act_sp
    self.act_low = act_low
    self.act_high = act_high

    low = np.hstack([env_low, act_low])
    high = np.hstack([env_high, act_high])

    self.scaler = lambda x: (x - low) / (high - low)

    self.eps = eps
    self.gamma = gamma
    self.alpha = alpha

    self.q = q_func

    self.samples = 50
    self.sampler = qmc.LatinHypercube(d=self.act_dim)

    self.obs_tc = obs_tc
    # self.max_acts = {
    #   l: defaultdict() for l in range(obs_tc.n_layers)
    # }
    self.max_acts: Dict[tuple, Any] = defaultdict(lambda: None)

    n_max = 50
    self.n_max_acts = [int(n_max / (l + 1)) for l in range(self.obs_tc.n_layers)]

  def begin_episode(self):
    pass

  def end_episode(self):
    self.eps.update()

  def get_q_values(self, obs, act=None):
    # TODO: allow multiple obs, only handles 1 at the moment

    obs_l_tiles = self.obs_tc[obs]

    if act is None:
      sample = self.sampler.random(n=self.samples)
      sample = qmc.scale(sample, self.act_low, self.act_high)

      obs_act = np.hstack([np.tile(obs, (sample.shape[0], 1)), sample])

      for l in range(self.obs_tc.n_layers):
        obs_tiles = tuple(map(tuple, np.hstack(obs_l_tiles[:l + 1])))

        for tiles in obs_tiles:  # this just loops over the only tiles, needs to be extracted
          max_acts = self.max_acts[tiles]
          if max_acts is None:
            self.max_acts[tiles] = max_acts = np.hstack([
              np.tile(obs, (self.n_max_acts[l], 1)),
              np.vstack([self.act_sp.sample() for _ in range(self.n_max_acts[l])]),
            ])

          max_acts = np.hstack([np.tile(obs, (max_acts.shape[0], 1)), max_acts[:, -self.act_dim:]])
          obs_act = np.vstack([obs_act, max_acts])
    else:
      obs_act = np.atleast_2d(np.hstack([obs, act]))

    q_values = self.q.values(obs_act)

    all_q_values = q_values.copy()
    all_obs_act = obs_act.copy()

    for l in reversed(range(self.obs_tc.n_layers)):
      obs_tiles = tuple(map(tuple, np.hstack(obs_l_tiles[:l + 1])))

      for tiles in obs_tiles:
        max_acts = self.max_acts[tiles]
        if max_acts is None:
          self.max_acts[tiles] = max_acts = np.hstack([
            np.tile(obs, self.n_max_acts[l], axis=0),
            np.vstack([self.act_sp.sample() for _ in range(self.n_max_acts[l])]),
          ])

        old_q_values = self.q.values(max_acts)

        all_q_values = np.hstack([all_q_values, old_q_values])
        all_obs_act = np.vstack([all_obs_act, max_acts])

        kmeans = KMeans(n_clusters=self.n_max_acts[l], init='random', n_init=1, max_iter=50)
        labels = kmeans.fit_predict(self.scaler(all_obs_act))

        new_max_acts = []
        for c in np.unique(labels):
          c_q_values = all_q_values[labels == c]
          c_obs_act = all_obs_act[labels == c, :]
          idx = np.random.choice(np.nonzero(c_q_values == np.max(c_q_values))[0])
          new_max_acts.append(c_obs_act[idx])
        self.max_acts[tiles] = np.vstack(new_max_acts)

        # m_inds = np.argpartition(all_q_values, -self.n_max_acts[l])[-self.n_max_acts[l]:]
        # self.max_acts[tiles] = all_obs_act[m_inds]

    return q_values, obs_act

  def get_action(self, obs):
    q_values, obs_act = self.get_q_values(obs)
    action_idx = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])
    action = obs_act[action_idx, -self.act_dim:]

    if np.random.random() < self.eps.value:
      action = self.act_sp.sample()

    return action

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done = t.s, t.a, t.r_p, t.s_p, t.a_d

    q_next = np.max(self.get_q_values(obs_n)[0])
    q_curr, obs_act = self.get_q_values(obs, act)

    delta = rew + (1. - done) * self.gamma * q_next - q_curr
    self.q.update(obs_act, delta, self.alpha)


class ProperTCQFunc:
  def __init__(self, obs_tc, act_tc, init_q=0.):
    self.obs_tc = obs_tc
    self.act_tc = act_tc

    self.q = {
      l: defaultdict(lambda: defaultdict(lambda: init_q * w_t * w_l))
      for l, w_t, w_l in zip(range(obs_tc.n_layers), obs_tc.w_tiles, obs_tc.w_layers)
    }

  def values(self, obs, act):
    obs, act = np.atleast_2d(obs), np.atleast_2d(act)

    q_values = np.zeros(obs.shape[0])

    l_obs_tiles = self.obs_tc[obs]
    l_act_tiles = self.act_tc[act]

    for l, (all_obs_tiles, all_act_tiles) in enumerate(zip(l_obs_tiles, l_act_tiles)):
      for i, (obs_tiles, act_tiles) in enumerate(zip(all_obs_tiles, all_act_tiles)):
        for ti, (o_t, a_t) in enumerate(zip(obs_tiles, act_tiles)):
          q_values[i] += self.q[l][ti][(o_t, a_t)]

    return q_values

  def update(self, obs, act, delta, lr):
    obs, act = np.atleast_2d(obs), np.atleast_2d(act)

    l_obs_tiles = self.obs_tc[obs]
    l_act_tiles = self.act_tc[act]

    for l, (all_obs_tiles, all_act_tiles, w_t, w_l) in enumerate(zip(l_obs_tiles, l_act_tiles,
                                                                     self.obs_tc.w_tiles, self.obs_tc.w_layers)):
      for i, (d, obs_tiles, act_tiles) in enumerate(zip(delta, all_obs_tiles, all_act_tiles)):
        for ti, (o_t, a_t) in enumerate(zip(obs_tiles, act_tiles)):
          self.q[l][ti][(o_t, a_t)] += lr * d * w_t * w_l


class ProperTCPFunc:
  def __init__(self, obs_tc, n_acts, act_dim, init_p=None):
    self.obs_tc = obs_tc

    self.act_shape = (n_acts, act_dim)

    init_p = init_p if init_p is not None else lambda: np.zeros((n_acts, act_dim))

    self.p = {
      l: defaultdict(lambda: defaultdict(lambda: init_p() * w_t * w_l))
      for l, w_t, w_l in zip(range(obs_tc.n_layers), obs_tc.w_tiles, obs_tc.w_layers)
    }

  def values(self, obs):
    obs = np.atleast_2d(obs)

    p_values = np.zeros((obs.shape[0], *self.act_shape))

    l_obs_tiles = self.obs_tc[obs]

    for l, all_obs_tiles in enumerate(l_obs_tiles):
      for i, obs_tiles in enumerate(all_obs_tiles):
        for ti, o_t in enumerate(obs_tiles):
          p_values[i] += self.p[l][ti][o_t]

    return p_values

  def update(self, obs, delta, lr):
    obs = np.atleast_2d(obs)

    l_obs_tiles = self.obs_tc[obs]

    for l, (all_obs_tiles, w_t, w_l) in enumerate(zip(l_obs_tiles, self.obs_tc.w_tiles, self.obs_tc.w_layers)):
      for i, (d, obs_tiles) in enumerate(zip(delta, all_obs_tiles)):
        for ti, o_t in enumerate(obs_tiles):
          self.p[l][ti][o_t] += lr * d * w_t * w_l


class ProperTCQAgent:
  def __init__(self, act_sp, act_low, act_high,
               obs_sp, obs_low, obs_high,
               eps: Schedule, gamma, alpha: Schedule,
               q_func: ProperTCQFunc, p_func: ProperTCPFunc,
               obs_tc: JaxLayeredTileCoder, act_tc: JaxLayeredTileCoder):
    self.act_dim = act_low.shape[0]
    self.act_sp = act_sp
    self.act_low = act_low
    self.act_high = act_high
    self.obs_dim = obs_low.shape[0]

    self.eps = eps
    self.gamma = gamma
    self.alpha = alpha

    self.obs_tc = obs_tc
    self.act_tc = act_tc

    self.q_func = q_func
    self.p_func = p_func

  def begin_episode(self):
    pass

  def end_episode(self):
    self.eps.update()
    self.alpha.update()

  def get_action(self, obs):
    obs = np.atleast_2d(obs)

    actions = self.p_func.values(obs).squeeze(0)
    q_values = self.q_func.values(np.repeat(obs, actions.shape[-2], axis=-2), actions)
    action_idx = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])
    action = actions[action_idx, :]

    if np.random.random() < self.eps.value:
      action = self.act_sp.sample()

    return action

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done = t.s, t.a, t.r_p, t.s_p, t.a_d
    obs, obs_n, act = np.atleast_2d(obs), np.atleast_2d(obs_n), np.atleast_2d(act)

    a_next = self.p_func.values(obs_n).squeeze(0)
    q_next = self.q_func.values(np.repeat(obs_n, a_next.shape[-2], axis=-2), a_next)

    q_curr = self.q_func.values(obs, act)

    delta = rew + (1. - done) * self.gamma * q_next - q_curr
    self.q_func.update(obs, act, delta, self.alpha)

    a_curr = self.p_func.values(obs).squeeze(0)
    a_curr_tried = np.concatenate([a_curr, act], axis=-2)
    # (n_act, act_dim)

    q_curr = self.q_func.values(np.repeat(obs, a_curr_tried.shape[-2], axis=-2), a_curr_tried)
    action_idx = np.random.choice(np.nonzero(q_curr == np.max(q_curr))[0])
    a_max = a_curr_tried[action_idx, :]

    act_delta = np.expand_dims(a_max - a_curr, axis=0)

    # act_delta = (a_curr if q_curr_max > q_curr else act) - a_curr
    self.p_func.update(obs, act_delta, self.alpha * 0.5)


def main():
  lims = np.array([[0, 1], [0, 1]])
  tiles = np.array([3, 3])
  tilings = np.array(7)

  tc = TileCoder(tiles, lims, tilings)

  o1 = np.array([0.6, 0.7])
  o2 = np.array([[0.4, 0.45], [0.78, 0.57]])
  inds1 = tc[o1]
  inds2 = [tc[o] for o in o2]

  my_tc = MyTileCoder(tiles, tilings, lims[:, 0], lims[:, 1])

  my_inds1 = my_tc[o1]
  my_inds2 = my_tc[o2]

  x_y = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), indexing='ij')
  x = np.vstack([_x.reshape(-1) for _x in x_y]).T
  my_inds3 = my_tc[x]

  # fig, ax = plt.subplots()
  # for i in range(tilings):
  #   ax.scatter(my_inds3[:, i, 0], my_inds3[:, i, 1])
  # plt.show()

  jtc = JaxTileCoder(tiles, tilings, lims[:, 0], lims[:, 1])

  j_inds1 = jtc[o1]
  j_inds2 = jtc[o2]

  jltc = JaxLayeredTileCoder([tiles, tiles], [tilings, tilings], lims[:, 0], lims[:, 1])

  jl_inds1 = jltc[o1]
  jl_inds2 = jltc[o2]

  q_func = JaxContTCQFunc(lims[:, 0], lims[:, 1], lims[:, 0], lims[:, 1], [tiles * 2] * 2, [tilings] * 2, init_q=-1.)

  obs_act = np.hstack([o2, o2])
  q_vals1 = q_func.values(obs_act)
  q_func.update(obs_act[0], np.array([1]), lr=0.1)
  q_vals2 = q_func.values(obs_act)
  q_func.update(obs_act[1], np.array([5]), lr=0.1)
  q_vals3 = q_func.values(obs_act)

  debug = 0


def main2():
  t = defaultdict(float)
  a = t[0]


if __name__ == '__main__':
  main()
  # main2()
