import numpy as np


class Boltzmann:
  def __init__(self, temp):
    self.temp = temp

  def probabilities(self, q_values):
    q_values = np.array(q_values)
    int_values = q_values - np.max(q_values, axis=-1, keepdims=True)
    exp_values = np.exp(int_values / self.temp)
    probs = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
    return probs
