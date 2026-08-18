"""Value representations for the continuous maze, behind one interface.

Each learner holds an action-value estimate Q(s, .) over a 2-D state and a
learnt-state signal l(s), and exposes:

  q(s)                     -> np.ndarray[n_actions]
  q_update(s, a, target)   -> (q_before, q_after) at s   (one learning step)
  l(s), l_update(s, target)                              (stability estimate)
  learnt(s)                -> bool
  end_episode()                                          (batch work, e.g. GP refits)

The learnt signal is the Chapter 3 rule -- an exponential average of whether
the policy at s moved less than tau_d under the update -- stored in the same
representation family as Q (a table, a tile coder, an MLP), except for the
GP, which uses its own posterior variance as the signal, the idea of the
2022 attempt.

  TabularCells   one entry per grid cell: Chapter 3 exactly (control)
  TileCoding     linear in overlapping tilings: local by construction
  MLP            small network trained by Adam on a replay of targets: global
  GP             exact GP per action on a capped set of recent targets (JAX)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from plan_compilation_framework.core import epsilon_greedy_probabilities
from plan_compilation_framework.labelling import js_distance

N_ACTIONS = 4


@dataclass
class LabelConfig:
  tau_d: float = 0.1
  tau_l: float = 0.85
  alpha_l: float = 0.4
  eps: float = 0.1            # policy epsilon used in the JSD test


class Learner:
  name = 'base'

  def __init__(self, q0: float, label: LabelConfig):
    self.q0 = q0
    self.label = label

  def q(self, s) -> np.ndarray:
    raise NotImplementedError

  def q_update(self, s, a, target):
    raise NotImplementedError

  def l(self, s) -> float:
    raise NotImplementedError

  def l_update(self, s, target):
    raise NotImplementedError

  def learnt(self, s) -> bool:
    return self.l(s) > self.label.tau_l

  def value(self, s) -> float:
    return float(self.q(s).max())

  def q_batch(self, P) -> np.ndarray:
    return np.stack([self.q(p) for p in P])

  def learnt_batch(self, P) -> np.ndarray:
    return np.array([self.learnt(p) for p in P], dtype=bool)

  def end_episode(self):
    pass

  # the Chapter 3 rule, shared by every learner that stores l explicitly
  def stability_target(self, s, q_before, q_after) -> float:
    if self.learnt(s):
      return 1.0                     # latch (soft, through the l estimate)
    d = js_distance(epsilon_greedy_probabilities(q_before, self.label.eps),
                    epsilon_greedy_probabilities(q_after, self.label.eps))
    return 1.0 if d < self.label.tau_d else 0.0

  def observe_update(self, s, a, target):
    """q update + l update in one call; returns (q_before, q_after)."""
    qb, qa = self.q_update(s, a, target)
    self.l_update(s, self.stability_target(s, qb, qa))
    return qb, qa


# ------------------------------------------------------------------ tabular

class TabularCells(Learner):
  name = 'tabular'

  def __init__(self, q0, label, alpha_floor=0.1):
    super().__init__(q0, label)
    self.alpha_floor = alpha_floor
    self.qt, self.lt, self.n = {}, {}, {}

  @staticmethod
  def _cell(s):
    return (int(np.floor(s[0])), int(np.floor(s[1])))

  def q(self, s):
    return self.qt.setdefault(self._cell(s), np.full(N_ACTIONS, self.q0)).copy()

  def q_update(self, s, a, target):
    c = self._cell(s)
    row = self.qt.setdefault(c, np.full(N_ACTIONS, self.q0))
    before = row.copy()
    n = self.n.get((c, a), 0)
    self.n[(c, a)] = n + 1
    alpha = max(1.0 / (n + 1), self.alpha_floor)
    row[a] += alpha * (target - row[a])
    return before, row.copy()

  def l(self, s):
    return self.lt.get(self._cell(s), 0.)

  def l_update(self, s, target):
    c = self._cell(s)
    self.lt[c] = self.lt.get(c, 0.) + self.label.alpha_l * (target - self.lt.get(c, 0.))


# ------------------------------------------------------------------ tile coding

class TileCoder:
  """Grid tilings over [0, size)^2 with one-cell tiles, offset per tiling."""

  def __init__(self, size, n_tilings=8, tile_width=1.0, seed=0):
    self.size, self.n_tilings, self.w = size, n_tilings, tile_width
    rng = np.random.default_rng(seed)
    self.offsets = rng.uniform(0, tile_width, size=(n_tilings, 2))
    self.per_dim = int(np.ceil(size / tile_width)) + 1
    self.n_features = n_tilings * self.per_dim * self.per_dim

  def indices(self, s):
    idx = np.empty(self.n_tilings, dtype=int)
    for t in range(self.n_tilings):
      i = int((s[0] + self.offsets[t, 0]) // self.w)
      j = int((s[1] + self.offsets[t, 1]) // self.w)
      idx[t] = t * self.per_dim * self.per_dim + i * self.per_dim + j
    return idx


class TileCoding(Learner):
  name = 'tilecoding'

  def __init__(self, q0, label, size, n_tilings=8, tile_width=1.0, lr=0.1, seed=0):
    super().__init__(q0, label)
    self.tc = TileCoder(size, n_tilings, tile_width, seed)
    self.lr = lr / n_tilings
    # weights initialised so that Q = q0 and l = 0 everywhere
    self.wq = np.full((N_ACTIONS, self.tc.n_features), q0 / n_tilings)
    self.wl = np.zeros(self.tc.n_features)

  def q(self, s):
    idx = self.tc.indices(s)
    return self.wq[:, idx].sum(axis=1)

  def q_update(self, s, a, target):
    idx = self.tc.indices(s)
    before = self.wq[:, idx].sum(axis=1)
    self.wq[a, idx] += self.lr * (target - before[a])
    return before, self.wq[:, idx].sum(axis=1)

  def l(self, s):
    return float(np.clip(self.wl[self.tc.indices(s)].sum(), 0., 1.))

  def l_update(self, s, target):
    idx = self.tc.indices(s)
    cur = self.wl[idx].sum()
    self.wl[idx] += (self.label.alpha_l / self.tc.n_tilings) * (target - cur)


# ------------------------------------------------------------------ MLP (JAX)

class MLP(Learner):
  """Two small networks (Q and l) trained by Adam on a replay of recent targets.

  Inputs are positions scaled to [0, 1]^2. Q's output is offset by q0 so the
  network starts pessimistic like the table. Each update pushes the new
  target into a buffer and takes one Adam step on a mini-batch that always
  includes it -- the online regime of the 2022 attempt.
  """
  name = 'mlp'

  def __init__(self, q0, label, size, hidden=64, lr=1e-3, batch=64, buffer=20_000, seed=0):
    super().__init__(q0, label)
    import jax, jax.numpy as jnp, optax
    self.jax, self.jnp, self.optax = jax, jnp, optax
    self.size = float(size)
    key = jax.random.PRNGKey(seed)

    def init(key, out):
      k1, k2, k3 = jax.random.split(key, 3)
      return {'w1': jax.random.normal(k1, (2, hidden)) * (2 / 2) ** 0.5 * 0.5, 'b1': jnp.zeros(hidden),
              'w2': jax.random.normal(k2, (hidden, hidden)) * (2 / hidden) ** 0.5 * 0.5, 'b2': jnp.zeros(hidden),
              'w3': jax.random.normal(k3, (hidden, out)) * 0.01, 'b3': jnp.zeros(out)}

    def fwd(p, x):
      h = jax.nn.relu(x @ p['w1'] + p['b1'])
      h = jax.nn.relu(h @ p['w2'] + p['b2'])
      return h @ p['w3'] + p['b3']

    kq, kl = jax.random.split(key)
    self.pq, self.pl = init(kq, N_ACTIONS), init(kl, 1)
    self.opt = optax.adam(lr)
    self.sq, self.sl = self.opt.init(self.pq), self.opt.init(self.pl)
    self.batch = batch
    self.cap = buffer
    self.bq_x = np.zeros((buffer, 2), np.float32); self.bq_a = np.zeros(buffer, np.int32); self.bq_y = np.zeros(buffer, np.float32)
    self.bl_x = np.zeros((buffer, 2), np.float32); self.bl_y = np.zeros(buffer, np.float32)
    self.nq = self.nl = 0          # items written (ring position = n % cap)
    self.rng = np.random.default_rng(seed)

    @jax.jit
    def q_fn(p, x):
      return fwd(p, x) + q0

    @jax.jit
    def l_fn(p, x):
      return jax.nn.sigmoid(fwd(p, x))[..., 0]

    def q_loss(p, x, a, y):
      pred = (fwd(p, x) + q0)[jnp.arange(x.shape[0]), a]
      return jnp.mean((pred - y) ** 2)

    def l_loss(p, x, y):
      logit = fwd(p, x)[..., 0]
      return jnp.mean(optax.sigmoid_binary_cross_entropy(logit, y))

    @jax.jit
    def q_step(p, st, x, a, y):
      g = jax.grad(q_loss)(p, x, a, y)
      upd, st = self.opt.update(g, st, p)
      return optax.apply_updates(p, upd), st

    @jax.jit
    def l_step(p, st, x, y):
      g = jax.grad(l_loss)(p, x, y)
      upd, st = self.opt.update(g, st, p)
      return optax.apply_updates(p, upd), st

    self._q_fn, self._l_fn, self._q_step, self._l_step = q_fn, l_fn, q_step, l_step

  def _x(self, s):
    return self.jnp.asarray(np.asarray(s, dtype=np.float32) / self.size)

  def q(self, s):
    return np.asarray(self._q_fn(self.pq, self._x(s)[None]))[0]

  def _sample(self, n_written):
    n = min(n_written, self.cap)
    k = min(self.batch - 1, n)
    return self.rng.integers(n, size=k) if k else np.zeros(0, dtype=int)

  def q_update(self, s, a, target):
    before = self.q(s)
    idx = self._sample(self.nq)
    pos = self.nq % self.cap
    self.bq_x[pos], self.bq_a[pos], self.bq_y[pos] = s, a, target
    self.nq += 1
    idx = np.append(idx, pos)
    self.pq, self.sq = self._q_step(self.pq, self.sq, self.jnp.asarray(self.bq_x[idx] / self.size),
                                    self.jnp.asarray(self.bq_a[idx]), self.jnp.asarray(self.bq_y[idx]))
    return before, self.q(s)

  def l(self, s):
    return float(self._l_fn(self.pl, self._x(s)[None])[0])

  def l_update(self, s, target):
    idx = self._sample(self.nl)
    pos = self.nl % self.cap
    self.bl_x[pos], self.bl_y[pos] = s, target
    self.nl += 1
    idx = np.append(idx, pos)
    self.pl, self.sl = self._l_step(self.pl, self.sl, self.jnp.asarray(self.bl_x[idx] / self.size),
                                    self.jnp.asarray(self.bl_y[idx]))

  def q_batch(self, P):
    return np.asarray(self._q_fn(self.pq, self.jnp.asarray(np.asarray(P, np.float32) / self.size)))

  def learnt_batch(self, P):
    return np.asarray(self._l_fn(self.pl, self.jnp.asarray(np.asarray(P, np.float32) / self.size))) > self.label.tau_l


# ------------------------------------------------------------------ GP (JAX)

class GP(Learner):
  """Exact Gaussian-process regression per action on a capped set of targets.

  Prior mean q0 (pessimistic far from data), RBF kernel with a fixed
  lengthscale in cells, fixed noise. Targets accumulate during the episode
  and the posterior is refitted once per episode (or every `refit_every`
  updates), so the cost is one Cholesky of at most `cap` points per action
  per refit -- milliseconds in JAX -- rather than one per step as in 2022.
  The learnt signal is the posterior standard deviation of Q(s, greedy)
  falling below tau_std: the 2022 idea of uncertainty as stability.
  """
  name = 'gp'

  def __init__(self, q0, label, lengthscale=1.5, signal_std=50., noise_std=5., cap=1500,
               refit_every=200, tau_std=10., seed=0):
    super().__init__(q0, label)
    import jax, jax.numpy as jnp
    from jax.scipy.linalg import cho_factor, cho_solve
    self.jnp = jnp
    self.ls, self.sig, self.noise, self.cap = lengthscale, signal_std, noise_std, cap
    self.refit_every, self.tau_std = refit_every, tau_std
    self.X = [deque(maxlen=cap) for _ in range(N_ACTIONS)]
    self.Y = [deque(maxlen=cap) for _ in range(N_ACTIONS)]
    self.fit = [None] * N_ACTIONS     # (X, alpha, cho) per action
    self.dirty = [0] * N_ACTIONS

    def kernel(A, B):
      d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
      return signal_std ** 2 * jnp.exp(-0.5 * d2 / lengthscale ** 2)

    @jax.jit
    def fit_fn(X, y):
      K = kernel(X, X) + noise_std ** 2 * jnp.eye(X.shape[0])
      cho = cho_factor(K, lower=True)
      alpha = cho_solve(cho, y - q0)
      return alpha, cho[0]

    @jax.jit
    def predict_fn(X, alpha, L, xs):
      Ks = kernel(xs, X)
      mean = q0 + Ks @ alpha
      v = jax.scipy.linalg.solve_triangular(L, Ks.T, lower=True)
      var = signal_std ** 2 - (v ** 2).sum(0)
      return mean, jnp.sqrt(jnp.maximum(var, 1e-9))

    self._fit, self._predict = fit_fn, predict_fn

  def _refit(self, a):
    if len(self.X[a]) == 0:
      self.fit[a] = None
    else:
      X = self.jnp.asarray(np.array(self.X[a], dtype=np.float32))
      y = self.jnp.asarray(np.array(self.Y[a], dtype=np.float32))
      alpha, L = self._fit(X, y)
      self.fit[a] = (X, alpha, L)
    self.dirty[a] = 0

  def _pred_batch(self, P):
    """means, stds of shape (len(P), n_actions)."""
    xs = self.jnp.asarray(np.asarray(P, dtype=np.float32))
    n = xs.shape[0]
    means, stds = np.full((n, N_ACTIONS), self.q0), np.full((n, N_ACTIONS), self.sig)
    for a in range(N_ACTIONS):
      if self.fit[a] is not None:
        m, sd = self._predict(*self.fit[a], xs)
        means[:, a], stds[:, a] = np.asarray(m), np.asarray(sd)
    return means, stds

  def _pred(self, s):
    m, sd = self._pred_batch(np.asarray(s, dtype=np.float32)[None])
    return m[0], sd[0]

  def q(self, s):
    return self._pred(s)[0]

  def q_batch(self, P):
    return self._pred_batch(P)[0]

  def learnt_batch(self, P):
    m, sd = self._pred_batch(P)
    return sd[np.arange(len(P)), np.argmax(m, axis=1)] < self.tau_std

  def q_update(self, s, a, target):
    before = self.q(s)
    self.X[a].append(np.asarray(s, dtype=np.float32))
    self.Y[a].append(float(target))
    self.dirty[a] += 1
    if self.dirty[a] >= self.refit_every:
      self._refit(a)
    return before, self.q(s)

  # variance is the learnt signal; l is reported on [0,1] for logging
  def l(self, s):
    means, stds = self._pred(s)
    return float(np.clip(1.0 - stds[int(np.argmax(means))] / self.sig, 0., 1.))

  def learnt(self, s):
    means, stds = self._pred(s)
    return bool(stds[int(np.argmax(means))] < self.tau_std)

  def l_update(self, s, target):
    pass  # the GP's learnt signal is its variance; nothing to update

  def end_episode(self):
    for a in range(N_ACTIONS):
      if self.dirty[a]:
        self._refit(a)


def make_learner(kind: str, q0: float, label: LabelConfig, size: int, seed: int, **kw) -> Learner:
  if kind == 'tabular':
    return TabularCells(q0, label, **kw)
  if kind == 'tilecoding':
    return TileCoding(q0, label, size=size, seed=seed, **kw)
  if kind == 'mlp':
    return MLP(q0, label, size=size, seed=seed, **kw)
  if kind == 'gp':
    return GP(q0, label, seed=seed, **kw)
  raise ValueError(kind)
