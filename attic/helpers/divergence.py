import numpy as np


def kl_divergence(p, q):
  p[p == 0.] = 1e-50
  q[q == 0.] = 1e-50
  return np.max([np.sum(np.where(p != 0, p * np.log(p / q), 0)), 0])


def js_divergence(p, q):
  m = .5 * (p + q)
  return np.sqrt(0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m))
