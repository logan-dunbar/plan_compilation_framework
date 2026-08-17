import numpy as np


class TileCoder:
  def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
    tiling_dims = np.array(np.ceil(tiles_per_dim), dtype=int) + 1
    self._offsets = offset(len(tiles_per_dim)) * \
                    np.repeat([np.arange(tilings)], len(tiles_per_dim), 0).T / float(tilings) % 1
    self._limits = np.array(value_limits)
    self._norm_dims = np.array(tiles_per_dim) / (self._limits[:, 1] - self._limits[:, 0])
    self._tile_base_ind = np.prod(tiling_dims) * np.arange(tilings)
    self._hash_vec = np.array([np.prod(tiling_dims[0:i]) for i in range(len(tiles_per_dim))])
    self._n_tiles = tilings * np.prod(tiling_dims)

  def __getitem__(self, x):
    off_coords = ((x - self._limits[:, 0]) * self._norm_dims + self._offsets).astype(int)
    return self._tile_base_ind + np.dot(off_coords, self._hash_vec)

  @property
  def n_tiles(self):
    return self._n_tiles


class LayeredTileCoder:
  def __init__(self, tiles_per_dim, value_limits, tilings, offset=lambda n: 2 * np.arange(n) + 1):
    assert len(tiles_per_dim) == len(tilings)
    assert len(tiles_per_dim[0]) == value_limits.shape[0]

    self._n_layers = len(tilings)

    self._tc = []
    self._tw = []
    self._lw = []
    self._n_t = []

    for t, n_t in zip(tiles_per_dim, tilings):
      tc = TileCoder(t, value_limits, n_t)
      self._tc.append(tc)
      self._tw.append(1. / n_t)
      self._lw.append(1. / len(tilings))
      self._n_t.append(tc.n_tiles)

  def __getitem__(self, x):
    res = []
    for tc, tw, lw in zip(self._tc, self._tw, self._lw):
      tiles = tc[x]
      res.append((tiles, tw, lw))

    return res

  @property
  def n_layers(self):
    return self._n_layers

  @property
  def n_tiles(self):
    return self._n_t

  @property
  def t_weights(self):
    return self._tw

  @property
  def l_weights(self):
    return self._lw


def main():
  tiles_per_dim = [1, 1]
  lims = [(0., 1.), (0., 1.)]
  tilings = 4

  tc = TileCoder(tiles_per_dim, lims, tilings)

  obs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

  inds = [tc[o] for o in obs]
  debug = 0


if __name__ == '__main__':
  main()
