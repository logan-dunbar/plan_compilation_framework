import numpy as np


class EpsilonGreedy:
  def __init__(self, eps):
    self.eps = eps

  def probabilities(self, q_values):
    q_values = np.array(q_values)
    i_max = q_values == np.max(q_values)
    probs = np.full(q_values.shape, self.eps / len(q_values))
    probs[i_max] += (1 - self.eps) / np.count_nonzero(i_max)
    return probs


