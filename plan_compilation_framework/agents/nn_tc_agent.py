from functools import partial

import jax
import math
import objax
import numpy as np
import jax.numpy as jnp
import jax.scipy as jsp
from objax import VarCollection
from objax.nn.init import xavier_normal
from jax import config
import gymnasium as gym

from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.helpers.schedules import Schedule
from plan_compilation_framework.policies.boltzmann import Boltzmann

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('TkAgg')
LOG2PI = math.log(2 * math.pi)
config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')


@partial(jnp.vectorize, signature='(m,m)->()')
def logdet_cholesky(L):
  return 2 * jnp.sum(jnp.log(jnp.diag(L)))


@partial(jnp.vectorize, signature='(d),(d)->(d)')
def sq_dist(x1, x2):
  return (x1 - x2) ** 2


@partial(jnp.vectorize, signature='(d),(d)->(d)')
def rbf_dist(sq_d, length):
  return sq_d / length ** 2


# @jax.jit
def rbf(x1, x2, length, scale):
  return scale ** 2 * jnp.exp(-0.5 * jnp.sum(rbf_dist(sq_dist(x1, x2), length), axis=-1))


def rbf_standard(x1, x2):
  return jnp.exp(-0.5 * jnp.sum(sq_dist(x1, x2), axis=-1))


rbf_full = jnp.vectorize(rbf, excluded={3}, signature='(a,b,d),(b,c,d),(d)->(a,c)')
rbf_diag = jnp.vectorize(rbf, excluded={3}, signature='(a,1,d),(a,1,d),(d)->(a,1)')
rbf_standard_full = jnp.vectorize(rbf_standard, signature='(a,b,d),(b,c,d)->(a,c)')
rbf_standard_diag = jnp.vectorize(rbf_standard, signature='(a,1,d),(a,1,d)->(a,1)')

vec_cholesky = jnp.vectorize(jsp.linalg.cholesky,
                             excluded={1}, signature='(d,d)->(d,d)')
vec_solve_tri = jnp.vectorize(jsp.linalg.solve_triangular,
                              excluded={2, 3}, signature='(d,d),(d,e)->(d,e)')
vec_cho_solve = jnp.vectorize(lambda chol, lower, y: jsp.linalg.cho_solve((chol, lower), y),
                              excluded={1}, signature='(d,d),(d,e)->(d,e)')
vec_trace = jnp.vectorize(jnp.trace, signature='(d,d)->()')


def mean_cov_to_nat(m, S, L=None):
  L = vec_cholesky(S, True) if L is None else L
  precision = vec_cho_solve(L, True, jnp.eye(S.shape[-2]))
  n1 = precision @ m
  n2 = -0.5 * precision
  return n1, n2, L


def nat_to_mean_cov(n1, n2, L=None):
  L = vec_cholesky(-n2, True) if L is None else L
  cov = 0.5 * vec_cho_solve(L, True, jnp.eye(n2.shape[-2]))
  mean = cov @ n1
  return mean, cov, L  # , jnp.sqrt(0.5) * L  # might be able to reuse cholesky here


def condition_to_marginal(m_f, k_ff, k_fu, k_uu, m, S, L=None):
  L = vec_cholesky(k_uu, True) if L is None else L
  alpha = vec_cho_solve(L, True, k_fu.swapaxes(-2, -1)).swapaxes(-2, -1)

  mu = m_f + alpha @ m
  var = k_ff - jnp.expand_dims(
    (alpha @ (k_uu - S) @ alpha.swapaxes(-1, -2))
    .diagonal(axis1=-2, axis2=-1), -1
  )

  return mu, var, L, alpha


def variational_expectation(y, q_m, q_v, variance, grads=False):
  e = (-0.5 * LOG2PI
       - 0.5 * jnp.log(variance)
       - 0.5 * ((y - q_m) ** 2 + q_v) / variance)

  if grads is False:
    return e
  else:
    # Compute first derivative:
    de_dm = (y - q_m) / variance
    # Compute second derivative:
    d2e_dm2 = -1 / variance
    return e, de_dm, d2e_dm2


def kl_divergence(m1, m2, L1=None, S1=None, L2=None, S2=None):
  assert not (L1 is None and S1 is None) and not (L2 is None and S2 is None)

  L1 = vec_cholesky(S1, True) if L1 is None else L1
  L2 = vec_cholesky(S2, True) if L2 is None else L2
  S1 = L1 @ L1.swapaxes(-2, -1) if S1 is None else S1

  d = L1.shape[-2]

  logdet1 = logdet_cholesky(L1)
  logdet2 = logdet_cholesky(L2)

  precision2 = vec_cho_solve(L2, True, jnp.eye(L2.shape[-2]))
  trace = vec_trace(precision2 @ S1)
  quad = (m2 - m1).swapaxes(-2, -1) @ precision2 @ (m2 - m1)

  return 0.5 * (logdet2 - logdet1 - d + trace + quad).squeeze()


class DualSVGP(objax.Module):
  def __init__(self, noise, length, scale, z, jitter=1e-6):
    self.noise = objax.StateVar(jnp.log(jnp.array(noise)))
    self.length = objax.TrainVar(jnp.log(jnp.array(length)))
    self.scale = objax.TrainVar(jnp.log(jnp.array(scale)))

    self.m_z = z.shape[-2]

    self.z = objax.TrainVar(jnp.array(z.copy()))
    self.I = jnp.eye(self.m_z)

    self.lam1 = objax.StateVar(jnp.zeros((self.m_z, 1)))
    self.lam2 = objax.StateVar(self.I * -1e-2)

    self.mean = lambda x: jnp.zeros(x.shape[:-1] + (1,))
    self.jitter = jitter

  def marginalise(self, x, p_L=None, q_L=None):
    d_in = x.shape[-1]
    d_out = 1

    x_e = jnp.expand_dims(x, -2)

    z = self.z.reshape((d_out, self.m_z, 1, d_in))
    z_T = self.z.reshape((d_out, 1, self.m_z, d_in))

    k_ff = rbf_diag(x_e, x_e, jnp.exp(self.length.value), jnp.exp(self.scale.value))
    k_fu = rbf_full(x_e, z_T, jnp.exp(self.length.value), jnp.exp(self.scale.value))
    k_uu = rbf_full(z, z_T, jnp.exp(self.length.value), jnp.exp(self.scale.value)) + self.I * self.jitter
    m = self.mean(self.z.value)
    m_f = self.mean(x)

    p_n1, p_n2, p_L = mean_cov_to_nat(m, k_uu, L=p_L)
    q_n1, q_n2 = p_n1 + self.lam1.value, p_n2 + self.lam2.value

    q_m, q_S, q_L = nat_to_mean_cov(q_n1, q_n2, L=q_L)

    mu_f, var_f, L_u, alpha = condition_to_marginal(m_f, k_ff, k_fu, k_uu, q_m, q_S, L=p_L)

    return mu_f, var_f, alpha, q_m, q_S, m, k_uu, p_L, q_L

  def elbo(self, x, y):
    mu_f, var_f, _, q_m, q_S, p_m, p_S, p_L, q_L = self.marginalise(x)

    obs_var = jnp.ones_like(y) * jnp.exp(self.noise.value)
    e = variational_expectation(y, mu_f, var_f, obs_var)

    kl = kl_divergence(q_m, p_m, L1=p_L, S1=p_S, S2=q_S).squeeze()

    return -(jnp.mean(e, axis=0).sum() - kl)

  def inference(self, x, y):
    mu_f, var_f, alpha, *_ = self.marginalise(x)

    obs_var = jnp.ones_like(y) * jnp.exp(self.noise.value)
    e, de_dm, d2e_d2m = variational_expectation(y, mu_f, var_f, obs_var, grads=True)

    de_dv = 0.5 * d2e_d2m

    dell_dmu1 = de_dm - 2. * de_dv * mu_f
    dell_dmu2 = de_dv

    lam1_u = jnp.mean((alpha.swapaxes(-2, -1) @ dell_dmu1), axis=0)
    lam2_u = jnp.mean((alpha.swapaxes(-2, -1) * dell_dmu2.swapaxes(-2, -1)) @ alpha, axis=0)

    return lam1_u, lam2_u

  def update_lams(self, lam1, lam2, lr=0.1):
    self.lam1.value = (1 - lr) * self.lam1 + lr * lam1
    self.lam2.value = (1 - lr) * self.lam2 + lr * lam2

  def predict(self, x, p_L=None, q_L=None):
    obs_var = jnp.ones((x.shape[-2], 1)) * jnp.exp(self.noise.value)
    mu_f, var_f, *_, p_L, q_L = self.marginalise(x, p_L=p_L, q_L=q_L)
    return mu_f, var_f + obs_var, p_L, q_L


@jax.jit
def get_tiles(x, widths, low, scaler, offsets):
  x = jnp.expand_dims(jnp.atleast_2d(x), axis=1)
  return widths * jnp.floor((x - low) * scaler - offsets) + 0.5 * widths + offsets / scaler


@jax.jit
def boltzmann_probabilities(q_values, temp):
  q_values = jnp.array(q_values)
  int_values = q_values - jnp.max(q_values, axis=-1, keepdims=True)
  exp_values = jnp.exp(int_values / temp)
  probs = exp_values / jnp.sum(exp_values, axis=-1, keepdims=True)
  return probs


class MyTileCoder:
  def __init__(self, tiles, tilings, low, high, offset=lambda n: 2 * jnp.arange(n) + 1):
    n_dims = len(tiles)
    tiles = jnp.array(tiles)

    offsets = offset(n_dims) * jnp.repeat(jnp.arange(tilings)[None, :], n_dims, axis=0).T
    self.offsets = (offsets / float(tilings)) % 1
    self.widths = (high - low) / tiles
    self.low = jnp.array(low)
    self.scaler = tiles / (high - low)

  def __getitem__(self, x):
    return get_tiles(x, self.widths, self.low, self.scaler, self.offsets)


class MyLayeredTileCoder:
  def __init__(self, tiles, tilings, low, high, offset=lambda n: 2 * jnp.arange(n) + 1):
    assert len(tiles) == len(tilings)
    assert len(tiles[0]) == low.shape[0]

    self.tilings = tilings
    self.n_layers = len(tilings)
    self.w_tiles = 1. / jnp.array(tilings)
    self.w_layers = jnp.ones(self.n_layers) * 1. / self.n_layers

    self.tc = []
    for t, ti in zip(tiles, tilings):
      tc = MyTileCoder(t, ti, low, high, offset)
      self.tc.append(tc)

  def __getitem__(self, x):
    return [tc[x] for tc in self.tc]


def get_ops(nets, include_last=False):
  def predict(tiles, x):
    values = []
    for l in range(len(tiles)):
      l_tiles = jnp.concatenate([t for t in tiles[:l+1]], axis=-1)
      values.append(nets[l](l_tiles))
    if include_last:
      values.append(nets[len(tiles)](x))
    return jnp.mean(jnp.dstack(values), axis=-1)
    # return jnp.sum(jnp.dstack(values), axis=-1)

  def huber_loss(tiles, x, y, delta=1.):
    y_pred = predict(tiles, x)
    abs_diff = jnp.abs(y - y_pred)
    return jnp.where(abs_diff > delta,
                     delta * (abs_diff - .5 * delta),
                     0.5 * abs_diff ** 2).sum()

  grad_loss = objax.GradValues(huber_loss, nets.vars())
  opt = objax.optimizer.Adam(nets.vars())

  def train_op(tiles, x, target, lr):
    grads, l = grad_loss(tiles, x, target)
    opt(lr=lr, grads=grads)
    return l

  predict = objax.Jit(predict, nets.vars())
  train_op = objax.Jit(train_op, nets.vars() + opt.vars())

  return train_op, predict


def get_q_ops(q_nets, obs_tc, act_tc=None):
  def predict(obs, act, weights=None):
    obs_tiles = obs_tc[obs]
    act_tiles = act_tc[act]

    if weights is None:
      weights = np.arange(1, len(q_nets) + 1).astype(float)
    weights /= np.sum(weights)

    q_values = []
    for l, w in zip(range(obs_tc.n_layers), weights):
      o_a = jnp.concatenate([o_t for o_t in obs_tiles[:l + 1]] + [a_t for a_t in act_tiles[:l + 1]], axis=-1)
      o_a = o_a.reshape(*o_a.shape[:-2], -1)
      q_values.append(q_nets[l](o_a) * w)
    # q_values.append(q_nets[obs_tc.n_layers](jnp.hstack([obs, act])) * weights[-1])

    return jnp.mean(jnp.hstack(q_values), axis=-1)
    # return jnp.sum(jnp.hstack(q_values), axis=-1)

  def loss(obs, act, y):
    y_pred = predict(obs, act)
    return objax.functional.loss.mean_squared_error(y_pred, y).sum()

  def huber_loss(obs, act, y, weights, delta=1.):
    y_pred = predict(obs, act, weights)
    abs_diff = jnp.abs(y - y_pred)
    return jnp.where(abs_diff > delta,
                     delta * (abs_diff - .5 * delta),
                     0.5 * abs_diff ** 2).sum()

  grad_loss = objax.GradValues(huber_loss, q_nets.vars())
  opt = objax.optimizer.Adam(q_nets.vars())

  def train_op(obs, act, target, lr, weights):
    # Get the gradients and the losses at the current input
    grads, l = grad_loss(obs, act, target, weights)

    # Then, run the optimizer passing in the learning rate and the gradients
    opt(lr=lr, grads=grads)

    # Finally, return the loss on these examples
    return l

  predict = objax.Jit(predict, q_nets.vars())
  train_op = objax.Jit(train_op, q_nets.vars() + opt.vars())

  return train_op, predict


def get_p_ops(p_nets, obs_tc):
  def predict(obs):
    obs_tiles = obs_tc[obs]

    p_values = []
    for l in range(obs_tc.n_layers):
      o = jnp.concatenate([o_t for o_t in obs_tiles[:l + 1]], axis=-1)
      o = o.reshape(*o.shape[:-2], -1)
      p_values.append(p_nets[l](o))
    p_values.append(p_nets[obs_tc.n_layers](jnp.atleast_2d(obs)))

    return jnp.stack(p_values, axis=-2)

  def huber_loss(obs, act, delta=1.):
    act_pred = predict(obs)
    abs_diff = jnp.abs(jnp.expand_dims(act, -2) - act_pred)
    return jnp.where(abs_diff > delta,
                     delta * (abs_diff - .5 * delta),
                     0.5 * abs_diff ** 2).sum()

  grad_loss = objax.GradValues(huber_loss, p_nets.vars())
  opt = objax.optimizer.Adam(p_nets.vars())

  def train_op(obs, act, lr):
    # Get the gradients and the losses at the current input
    grads, l = grad_loss(obs, act)

    # Then, run the optimizer passing in the learning rate and the gradients
    opt(lr=lr, grads=grads)

    # Finally, return the loss on these examples
    return l

  predict = objax.Jit(predict, p_nets.vars())
  train_op = objax.Jit(train_op, p_nets.vars() + opt.vars())

  return train_op, predict


def get_policy_ops(gps):
  elbos = []
  opts = []
  all_vars = VarCollection()
  for i, gp in enumerate(gps):
    elbos.append(objax.GradValues(gp.elbo, gp.vars()))
    opts.append(objax.optimizer.Adam(gp.vars()))

    all_vars += gp.vars(str(i)) + opts[-1].vars(str(i))

  def train_op(obs, acts, lr):
    # obs_tiles = obs_tc[obs]
    obs = jnp.atleast_2d(obs)

    for l, (elbo, opt) in enumerate(zip(elbos, opts)):
      # o = jnp.concatenate([o_t for o_t in obs_tiles[:l + 1]], axis=-1)
      de, e = elbo(obs, acts)
      opt(lr, de)

  def inference(obs, acts):
    # obs_tiles = obs_tc[obs]
    obs = jnp.atleast_2d(obs)

    lams = []
    for l, gp in enumerate(gps):
      # o = jnp.concatenate([o_t for o_t in obs_tiles[:l + 1]], axis=-1)
      lams.append(gp.inference(obs, acts))
    return lams

  def predict(obs, p_Ls=None, q_Ls=None):
    # obs_tiles = obs_tc[obs]
    obs = jnp.atleast_2d(obs)

    act_dists = []
    for l, gp in enumerate(gps):
      # o = jnp.concatenate([o_t for o_t in obs_tiles[:l + 1]], axis=-1)
      p_L, q_L = None, None
      if p_Ls is not None and q_Ls is not None:
        p_L, q_L = p_Ls[l], q_Ls[l]
      act_dists.append(gp.predict(obs, p_L=p_L, q_L=q_L))
    return act_dists

  # train_op = objax.Jit(train_op, all_vars)
  # inference = objax.Jit(inference, all_vars)
  # predict = objax.Jit(predict, all_vars)

  def update_lams(obs, acts, lr):
    lams = inference(obs, acts)
    for gp, lam in zip(gps, lams):
      gp.update_lams(*lam, lr)

  return train_op, update_lams, predict


class NNTCContQAgent:
  def __init__(self, act_sp, act_low, act_high,
               obs_sp, obs_low, obs_high,
               eps: Schedule, gamma, alpha: Schedule,
               buffer, batch_sz, min_updates, update_freq,
               obs_tc: MyLayeredTileCoder, act_tc: MyLayeredTileCoder,
               q_dims=(64, 64), p_dims=(64, 32)):
    self.act_dim = act_low.shape[0]
    self.act_sp = act_sp
    self.act_low = act_low
    self.act_high = act_high
    self.obs_dim = obs_low.shape[0]

    low = np.hstack([obs_low, act_low])
    high = np.hstack([obs_high, act_high])

    self.scaler = lambda x: (x - low) / (high - low)

    self.eps = eps
    self.gamma = gamma
    self.alpha = alpha
    self.policy = Boltzmann(temp=0.1)
    self.n_rand = 60

    self.buffer = buffer
    self.batch_sz = batch_sz
    self.min_updates = min_updates
    self.update_freq = update_freq
    self.updates = 0
    self.grad_steps = update_freq // 2
    # self.grad_steps = 1

    self.obs_tc = obs_tc
    self.act_tc = act_tc
    total_dim = self.obs_dim + self.act_dim

    # q_nets = objax.ModuleList()
    # w_init = partial(xavier_normal, gain=0.001)
    # for l in range(obs_tc.n_layers):
    #   in_dim = np.sum(obs_tc.tilings[:l + 1]) * total_dim
    #   q_nets.append(get_net(in_dim, 1, w_init))
    # q_nets.append(get_net(total_dim, 1, w_init))
    #
    # self.q_train_op, self.q_predict = get_q_ops(q_nets, obs_tc, act_tc)
    #
    # p_nets = objax.ModuleList()
    # w_init = partial(xavier_normal, gain=0.001)
    # for l in range(obs_tc.n_layers):
    #   in_dim = np.sum(obs_tc.tilings[:l + 1]) * self.obs_dim
    #   p_nets.append(get_net(in_dim, self.act_dim, w_init))
    # p_nets.append(get_net(self.obs_dim, self.act_dim, w_init))
    #
    # self.p_train_op, self.p_predict = get_p_ops(p_nets, obs_tc)
    #
    # # p_gps = objax.ModuleList()
    # # for l in range(self.obs_tc.n_layers):
    # #   z = jnp.vstack([obs_sp.sample() for _ in range(100)])
    # #   gp = DualSVGP(noise=0.5, length=[1.0] * e_dim, scale=0.5, z=z)
    # #   p_gps.append(gp)
    # #
    # # self.p_train_op, self.p_update_lams, self.p_predict = get_policy_ops(p_gps)

    include_last = True
    q_in_dims = [np.sum(obs_tc.tilings[:l + 1]) * total_dim for l in range(obs_tc.n_layers)]
    if include_last:
      q_in_dims.append(total_dim)
    q_nets = get_nets(obs_tc.n_layers, q_in_dims, 1, hidden_dims=q_dims, include_last=include_last)
    self.q_train_op, self.q_predict = get_ops(q_nets, include_last=include_last)

    p_in_dims = [np.sum(obs_tc.tilings[:l + 1]) * self.obs_dim for l in range(act_tc.n_layers)]
    if include_last:
      p_in_dims.append(self.obs_dim)
    p_nets = get_nets(act_tc.n_layers, p_in_dims, self.act_dim, hidden_dims=p_dims, include_last=include_last)
    self.p_train_op, self.p_predict = get_ops(p_nets, include_last=include_last)

  def begin_episode(self):
    pass

  def end_episode(self):
    self.eps.update()
    self.alpha.update()

  def get_q_values(self, obs, act=None):
    pass

  def get_action(self, obs):
    obs_tiles = [t.reshape(t.shape[0], -1) for t in self.obs_tc[obs]]

    act = self.p_predict(obs_tiles, obs).squeeze(0)
    act = jnp.clip(act, self.act_low, self.act_high)

    if np.random.random() < self.eps:
      act = self.act_sp.sample()

    return act

  def do_update(self, t: Transition):
    obs, act, rew, obs_n, done, term, trunc = t.s, t.a, t.r_p, t.s_p, t.a_d, t.te, t.tr
    self.buffer.store(dict(obs=obs, act=act, rew=rew, obs_n=obs_n, done=done, term=term, trunc=trunc))

    if self.buffer.curr_size > self.min_updates and self.updates % self.update_freq == 0:
      q_losses, p_losses = [], []
      for i in range(self.grad_steps):
        batch = self.buffer.sample(self.batch_sz, gamma=self.gamma)

        obs_tiles = [t.reshape(t.shape[0], -1) for t in self.obs_tc[batch['obs']]]
        obs_n_tiles = [t.reshape(t.shape[0], -1) for t in self.obs_tc[batch['obs_n']]]
        act_tiles = [t.reshape(t.shape[0], -1) for t in self.act_tc[batch['act']]]

        act_n = self.p_predict(obs_n_tiles, batch['obs_n'])
        act_n = jnp.clip(act_n, self.act_low, self.act_high)
        act_n_tiles = [t.reshape(t.shape[0], -1) for t in self.act_tc[act_n]]

        # this seems to be SARSA rather than q-learning, because act_n is the action
        # the agent 'would' have taken. Perhaps I can put some random actions in here
        # to be more q-learning like, as this would be sampling from the off-policy actions

        q_next = self.q_predict([jnp.hstack([o_t, a_t]) for o_t, a_t in zip(obs_n_tiles, act_n_tiles)],
                                jnp.hstack([batch['obs_n'], act_n]))
        target = (batch['rew'] + (1. - batch['done']) * self.gamma * q_next.squeeze())[:, None]

        q_loss = self.q_train_op([jnp.hstack([o_t, a_t]) for o_t, a_t in zip(obs_tiles, act_tiles)],
                                 jnp.hstack([batch['obs'], batch['act']]),
                                 target, self.alpha.value)

        a_curr = self.p_predict(obs_tiles, batch['obs'])
        a_curr = jnp.clip(a_curr, self.act_low, self.act_high)
        a_curr = jnp.vstack([a_curr, batch['act']])

        n_acts = self.n_rand + 2  # + 2 for chosen action and current predicted action
        rand_acts = objax.random.uniform((self.batch_sz * self.n_rand, self.act_dim)) * (self.act_high - self.act_low) + self.act_low
        a_curr = jnp.vstack([a_curr, rand_acts])

        a_curr_tiles = [t.reshape(t.shape[0], -1) for t in self.act_tc[a_curr]]

        o_curr = jnp.tile(batch['obs'], (n_acts, 1))
        o_curr_tiles = [t.reshape(t.shape[0], -1) for t in self.obs_tc[o_curr]]  # TODO: wasteful

        q_curr = self.q_predict([jnp.hstack([o_t, a_t]) for o_t, a_t in zip(o_curr_tiles, a_curr_tiles)],
                                jnp.hstack([o_curr, a_curr]))
        q_curr = q_curr.reshape(n_acts, self.batch_sz)

        # boltzmann policy update
        probs = boltzmann_probabilities(q_curr.T, self.policy.temp)
        cum_probs = probs.cumsum(axis=-1)
        sample = np.random.rand(probs.shape[0])
        a_inds = (cum_probs < np.expand_dims(sample, axis=-1)).sum(-1)

        # greedy policy update
        # a_inds = jnp.argmax(q_curr, axis=0)

        a_curr = a_curr.reshape(n_acts, self.batch_sz, self.act_dim)
        max_acts = jnp.take_along_axis(a_curr, jnp.expand_dims(a_inds, (0, -1)), axis=0).squeeze(0)

        p_loss = self.p_train_op(obs_tiles, batch['obs'], max_acts, self.alpha.value)

        q_losses.append(q_loss)
        p_losses.append(p_loss)

      print(f'\tu: {self.updates}, mean q_loss: {np.mean(q_losses)}, mean p_loss: {np.mean(p_losses)}')

    self.updates += 1

  def get_action_multiple(self, obs):
    # q_values, obs_act = self.get_q_values(obs)
    # action_idx = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])
    # action = obs_act[action_idx, -self.act_dim:]

    # if np.random.random() < self.eps.value:
    #   action = self.act_sp.sample()

    # q_values = self.predict(obs, act)

    # act_dists = self.p_predict(obs)
    #
    # acts = []
    # for m_f, var_f, _, _ in act_dists:
    #   acts.append(objax.random.normal((20, 1), mean=m_f.squeeze(), stddev=jnp.sqrt(var_f.squeeze())))
    #   acts.append(jnp.array([self.act_sp.sample() for _ in range(20)]))
    #
    # acts = jnp.vstack(acts)
    # obss = jnp.repeat(jnp.array([obs]), acts.shape[0], axis=0)
    #
    # q_values = self.predict(obss, acts)
    # probs = self.policy.probabilities(q_values)
    #
    # inds = np.random.choice(range(acts.shape[0]), 20, p=probs)
    #
    # upd_obs = obss[inds]
    # upd_acts = acts[inds]
    # self.p_update_lams(upd_obs, upd_acts, 0.05)
    # self.p_train_op(upd_obs, upd_acts, 0.01)
    #
    # temp = self.p_predict(obs)

    samples = 60

    acts = self.p_predict(obs).squeeze(0)
    # acts = jnp.vstack([acts, [self.act_sp.sample() for _ in range(samples)]])

    obss = jnp.repeat(jnp.array([obs]), acts.shape[0], axis=0)
    q_values = self.q_predict(obss, acts)

    # probs = self.policy.probabilities(q_values)
    # inds = np.random.choice(range(acts.shape[0]), samples, p=probs)

    # upd_obs = obss[inds]
    # upd_acts = acts[inds]
    #
    # self.p_train_op(upd_obs, upd_acts, 0.001)

    act_idx = np.random.choice(np.nonzero(q_values == np.max(q_values))[0])
    act = acts[act_idx, :]

    if np.random.random() < self.eps.value:
      act = self.act_sp.sample()

    return act

  def do_update_multiple(self, t: Transition):
    obs, act, rew, obs_n, done, term, trunc = t.s, t.a, t.r_p, t.s_p, t.a_d, t.te, t.tr
    self.buffer.store(dict(obs=obs, act=act, rew=rew, obs_n=obs_n, done=done, term=term, trunc=trunc))

    if self.buffer.curr_size > self.min_updates and self.updates % self.update_freq == 0:
      q_losses, p_losses = [], []
      for i in range(self.grad_steps):
        batch = self.buffer.sample(self.batch_sz)

        acts_n = self.p_predict(batch['obs_n'])

        obs_ns = jnp.repeat(jnp.array(batch['obs_n']), acts_n.shape[-2], axis=0)
        acts_n_r = acts_n.reshape(np.prod(acts_n.shape[:-1]), acts_n.shape[-1])
        q_values_next = self.q_predict(obs_ns, acts_n_r).reshape(acts_n.shape[:-1])

        q_next = jnp.max(q_values_next, axis=-1)
        target = batch['rew'] + (1. - batch['done']) * self.gamma * q_next

        q_loss = self.q_train_op(batch['obs'], batch['act'], target, self.alpha)
        q_losses.append(q_loss)

        # now update the current action choice
        acts = self.p_predict(batch['obs'])
        taken_acts = jnp.expand_dims(batch['act'], axis=-2)
        acts = jnp.concatenate([acts, taken_acts], axis=-2)

        # add small random jitter?
        # random_acts = 0.5 * (self.act_high - self.act_low) * objax.random.normal(acts.shape, stddev=0.05)
        # acts = jnp.concatenate([acts, random_acts], axis=-2)

        obss = jnp.repeat(jnp.array(batch['obs']), acts.shape[-2], axis=0)
        acts_r = acts.reshape(np.prod(acts.shape[:-1]), acts.shape[-1])
        q_values = self.q_predict(obss, acts_r).reshape(acts.shape[:-1])

        q_max = q_values == jnp.max(q_values, axis=-1, keepdims=True)
        probs = jnp.float64(q_max) / jnp.sum(q_max, axis=-1, keepdims=True)

        cum_probs = probs.cumsum(axis=-1)
        sample = np.random.rand(1, probs.shape[0])
        inds = (np.expand_dims(cum_probs, axis=0) < np.expand_dims(sample, axis=-1)).sum(-1).T
        inds = jnp.expand_dims(inds, -1)

        obs_r = obss.reshape(*acts.shape[:-1], -1)
        upd_obs = jnp.take_along_axis(obs_r, indices=inds, axis=-2)
        upd_acts = jnp.take_along_axis(acts, indices=inds, axis=-2)

        p_loss = self.p_train_op(upd_obs.reshape(-1, upd_obs.shape[-1]),
                                 upd_acts.reshape(-1, upd_acts.shape[-1]), self.alpha * 0.1)
        p_losses.append(p_loss)

      print(f'\tu: {self.updates}, mean q_loss: {np.mean(q_losses)}, mean p_loss: {np.mean(p_losses)}')

    self.updates += 1

  def do_update_with_random(self, t: Transition):
    obs, act, rew, obs_n, done, term, trunc = t.s, t.a, t.r_p, t.s_p, t.a_d, t.te, t.tr
    self.buffer.store(dict(obs=obs, act=act, rew=rew, obs_n=obs_n, done=done, term=term, trunc=trunc))

    if self.buffer.curr_size > self.min_updates and self.updates % self.update_freq == 0:
      q_losses, p_losses = [], []
      for i in range(self.grad_steps):
        batch = self.buffer.sample(self.batch_sz)

        # update
        # batch['obs'], batch['act'], batch['rew'], batch['obs_n'], batch['done']

        # next_a = np.vstack([self.act_sp.sample() for _ in range(self.batch_sz)])
        # next_a = np.tile(next_a, (self.batch_sz, 1))
        # next_o = np.repeat(batch['obs'], self.batch_sz, axis=0)

        acts_n = self.p_predict(batch['obs_n'])

        # rand_acts_n = jnp.vstack([self.act_sp.sample() for _ in range(self.batch_sz)])
        # rand_acts_n = jnp.tile(rand_acts_n, (self.batch_sz, 1, 1))
        #
        # acts_n = jnp.concatenate([acts_n, rand_acts_n], axis=-2)

        obs_ns = jnp.repeat(jnp.array(batch['obs_n']), acts_n.shape[-2], axis=0)
        acts_n_r = acts_n.reshape(np.prod(acts_n.shape[:-1]), acts_n.shape[-1])
        q_values_next = self.q_predict(obs_ns, acts_n_r).reshape(acts_n.shape[:-1])

        q_next = jnp.max(q_values_next, axis=-1)

        target = batch['rew'] + (1. - batch['done']) * self.gamma * q_next

        q_loss = self.q_train_op(batch['obs'], batch['act'], target, self.alpha)
        q_losses.append(q_loss)

        # now update the current action choice
        acts = self.p_predict(batch['obs'])

        rand_acts = jnp.vstack([self.act_sp.sample() for _ in range(self.batch_sz)])
        rand_acts = jnp.tile(rand_acts, (self.batch_sz, 1, 1))

        acts = jnp.concatenate([acts, rand_acts], axis=-2)

        obss = jnp.repeat(jnp.array(batch['obs']), acts.shape[-2], axis=0)
        acts_r = acts.reshape(np.prod(acts.shape[:-1]), acts.shape[-1])
        q_values = self.q_predict(obss, acts_r).reshape(acts.shape[:-1])

        probs = self.policy.probabilities(q_values)
        cum_probs = probs.cumsum(axis=-1)
        sample = np.random.rand(64, probs.shape[0])
        inds = (np.expand_dims(cum_probs, axis=0) < np.expand_dims(sample, axis=-1)).sum(-1).T
        inds = jnp.expand_dims(inds, -1)

        obs_r = obss.reshape(*acts.shape[:-1], -1)
        upd_obs = jnp.take_along_axis(obs_r, indices=inds, axis=-2)
        upd_acts = jnp.take_along_axis(acts, indices=inds, axis=-2)

        p_loss = self.p_train_op(upd_obs.reshape(-1, upd_obs.shape[-1]),
                                 upd_acts.reshape(-1, upd_acts.shape[-1]), self.alpha * 0.1)
        p_losses.append(p_loss)

        # q_values = self.q_predict(next_o, next_a)
        # q_values = q_values.reshape(self.batch_sz, self.batch_sz)
        # q_next = jnp.max(q_values, axis=-1)
        #
        # target = batch['rew'] + (1. - batch['done']) * self.gamma * q_next
        #
        # loss = self.train_op(batch['obs'], batch['act'], target, self.alpha)
        # losses.append(loss)

      # print(f'u: {self.updates}, mean loss: {np.mean(losses)}')

      print(f'u: {self.updates}, mean q_loss: {np.mean(q_losses)}, mean p_loss: {np.mean(p_losses)}')

    self.updates += 1


def main():
  lims = np.array([[0, 1], [0, 1]])
  tiles = np.array([3, 3])
  tilings = np.array(7)

  o1 = np.array([0.6, 0.7])
  o2 = np.array([[0.4, 0.45], [0.78, 0.57]])

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

  my_ltc = MyLayeredTileCoder([tiles, 2 * tiles, 4 * tiles], [tilings, tilings, tilings], lims[:, 0], lims[:, 1])

  my_l_inds1 = my_ltc[o1]
  my_l_inds2 = my_ltc[o2]

  # fig, ax = plt.subplots()
  # ax.scatter(o1[0], o1[1], marker='x')
  # for i in range(len(my_l_inds1)):
  #   ax.scatter(my_l_inds1[i][0][:, 0], my_l_inds1[i][0][:, 1])
  # plt.show()


def get_data(x, noise=0.1):
  f = np.sin(3 * x[:, 0]) + np.cos(1.3 * x[:, 1]) + np.sin(0.6 * x[:, 0]) * np.cos(7 * x[:, 1]) + np.sum(x ** 2, axis=1)
  y = f + np.random.normal(scale=np.sqrt(noise), size=f.shape)
  return f, y


def get_net(in_dim, out_dim, w_init, hidden_dims=(64, 64)):
  dims = (in_dim,) + hidden_dims
  layers = []
  for i, o in zip(dims[:-1], dims[1:]):
    layers.append(objax.nn.Linear(i, o, w_init=w_init))
    layers.append(objax.functional.relu)
  layers.append(objax.nn.Linear(dims[-1], out_dim, w_init=w_init))
  return objax.nn.Sequential(layers)


def get_nets(n_layers, in_dims, out_dim, hidden_dims=None, w_init_gain=0.001, include_last=False):
  nets = objax.ModuleList()
  w_init = partial(xavier_normal, gain=w_init_gain)
  for l, in_dim in zip(range(n_layers), in_dims):
    nets.append(get_net(in_dim, out_dim, w_init, hidden_dims=hidden_dims))
  if include_last:
    nets.append(get_net(in_dims[-1], out_dim, w_init, hidden_dims=hidden_dims))
  return nets


def q_net_test():
  alpha = 0.001
  shape = (50, 50)
  o_a = np.meshgrid(np.linspace(-0.5, 0.5, shape[0]), np.linspace(-0.5, 0.5, shape[1]), indexing='ij')
  x_grid = np.vstack([_x.reshape(-1) for _x in o_a]).T

  y_true, _ = get_data(x_grid)

  fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(18, 6))
  axes[0].plot_surface(o_a[0], o_a[1], y_true.reshape(shape))
  plt.show(block=False)
  plt.pause(0.5)

  obs_sp = gym.spaces.Box(np.array([-1.]), np.array([1.01]), dtype=np.double)
  act_sp = gym.spaces.Box(np.array([-1.]), np.array([1.01]), dtype=np.double)

  e_low, e_high = obs_sp.low, obs_sp.high
  a_low, a_high = act_sp.low, act_sp.high
  e_dim, a_dim = e_low.shape[0], a_low.shape[0]
  total_dim = e_dim + a_dim

  start, depth = 0, 5
  # tiles = [[2 ** t + 1] * total_dim for t in range(start, depth + 1)]
  tiles = [[t] * (e_dim + a_dim) for t in [2, 3, 5, 7, 11]]
  tilings = [np.power(2, int(np.ceil(np.log2(4 * (e_dim + a_dim)))))] * len(tiles)

  e_tiles = [t[:e_dim] for t in tiles]
  a_tiles = [t[-a_dim:] for t in tiles]

  obs_tc = MyLayeredTileCoder(e_tiles, tilings, e_low, e_high)
  act_tc = MyLayeredTileCoder(a_tiles, tilings, a_low, a_high)

  with jax.disable_jit(False):
    q_nets = objax.ModuleList()
    w_init = partial(xavier_normal, gain=0.001)
    for l in range(obs_tc.n_layers):
      in_dim = np.sum(obs_tc.tilings[:l + 1]) * total_dim
      q_net = get_net(in_dim, 1, w_init)
      q_nets.append(q_net)
    # q_nets.append(get_net(total_dim, 1, w_init))

    train_op, predict = get_q_ops(q_nets, obs_tc, act_tc)

    # weights = np.sqrt(np.arange(1, len(q_nets) + 1).astype(float))
    weights = np.ones(len(q_nets)) / len(q_nets)

    art1 = None
    art2 = None
    art3 = None
    for i in range(1000000):
      x = np.random.uniform(-1., 1., (64, 2))
      f, y = get_data(x)

      train_op(x[:, 0:1], x[:, 1:2], y, alpha, weights)

      if i % 50 == 0:
        y_pred = predict(x_grid[:, 0:1], x_grid[:, 1:2], weights=weights)

        if art1 is not None:
          art1.remove()
        if art2 is not None:
          art2.remove()
        if art3 is not None:
          art3.remove()

        art3 = axes[0].scatter(x[:, 0], x[:, 1], y, color='tab:red', marker='x')
        art1 = axes[1].plot_surface(o_a[0], o_a[1], y_pred.reshape(shape), color='tab:blue')
        art2 = axes[2].plot_surface(o_a[0], o_a[1], jnp.abs((y_true - y_pred)).reshape(shape),
                                    cmap=matplotlib.colormaps['coolwarm'])
        axes[2].set_zlim([0, 1.])
        plt.show(block=False)
        plt.pause(0.5)

      if i % 1000 == 0 and i / 1000 <= 6.:
        weights *= weights

  plt.show()


def get_normal_ops(q_net):
  def loss(x, y):
    y_pred = q_net(x)
    return objax.functional.loss.mean_squared_error(y_pred, y).sum()

  def huber_loss(x, y, delta=1.):
    y_pred = q_net(x)
    abs_diff = jnp.abs(y - y_pred)
    return jnp.where(abs_diff > delta,
                     delta * (abs_diff - .5 * delta),
                     0.5 * abs_diff ** 2).sum()

  grad_loss = objax.GradValues(huber_loss, q_net.vars())
  opt = objax.optimizer.Adam(q_net.vars())

  def train_op(x, target, lr):
    # Get the gradients and the losses at the current input
    grads, l = grad_loss(x, target)

    # Then, run the optimizer passing in the learning rate and the gradients
    opt(lr=lr, grads=grads)

    # Finally, return the loss on these examples
    return l

  train_op = objax.Jit(train_op, q_net.vars() + opt.vars())

  return train_op


def normal_net_test():
  alpha = 0.001
  shape = (50, 50)
  o_a = np.meshgrid(np.linspace(-1., 1., shape[0]), np.linspace(-1., 1., shape[1]), indexing='ij')
  x_grid = np.vstack([_x.reshape(-1) for _x in o_a]).T

  y_true, _ = get_data(x_grid)

  fig, axes = plt.subplots(1, 3, subplot_kw={"projection": "3d"}, figsize=(18, 6))
  axes[0].plot_surface(o_a[0], o_a[1], y_true.reshape(shape))
  plt.show(block=False)
  plt.pause(0.5)

  with jax.disable_jit(False):
    w_init = partial(xavier_normal, gain=0.001)
    q_net = get_net(2, 1, w_init)
    train_op = get_normal_ops(q_net)

    art1 = None
    art2 = None
    art3 = None
    for i in range(1000000):
      x = np.random.uniform(-1., 1., (64, 2))
      f, y = get_data(x)

      train_op(x, y, alpha)

      if i % 50 == 0:
        y_pred = q_net(x_grid).squeeze()

        if art1 is not None:
          art1.remove()
        if art2 is not None:
          art2.remove()
        if art3 is not None:
          art3.remove()

        art3 = axes[0].scatter(x[:, 0], x[:, 1], y, color='tab:red', marker='x')
        art1 = axes[1].plot_surface(o_a[0], o_a[1], y_pred.reshape(shape), color='tab:blue')
        art2 = axes[2].plot_surface(o_a[0], o_a[1], jnp.abs((y_true - y_pred)).reshape(shape),
                                    cmap=matplotlib.colormaps['coolwarm'])
        axes[2].set_zlim([0, 1.])
        plt.show(block=False)
        plt.pause(0.5)

  plt.show()


def action_pull_test():
  acts = np.random.uniform(size=(40, 2))

  for i in range(1000):
    act_ind = np.random.choice(range(2))
    act = acts[act_ind, :]
    if np.random.random() < 0.01:
      act = np.random.uniform(size=2)

    target = act - acts

    acts += 0.05 * target

    debug = 1
  debug = 0


if __name__ == '__main__':
  # main()
  q_net_test()
  # normal_net_test()
  # action_pull_test()
