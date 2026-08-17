import numpy as np


class ReplayBuffer:
  def __init__(self, max_size=1e5, dtype=np.float32):
    self.max_size = int(max_size)
    self.dtype = dtype
    self.curr_size = 0
    self.curr_idx = 0

    self.obs = None
    self.action = None
    self.reward = None
    self.obs_n = None
    self.term = None
    self.trunc = None

  def store(self, data):
    if self.obs is None:
      self.obs = np.empty([self.max_size] + list(data['obs'].shape), dtype=self.dtype)
      self.action = np.empty([self.max_size] + list(data['act'].shape), dtype=self.dtype)
      self.reward = np.empty([self.max_size], dtype=self.dtype)
      self.obs_n = np.empty([self.max_size] + list(data['obs_n'].shape), dtype=self.dtype)
      self.term = np.empty([self.max_size], dtype=bool)
      self.trunc = np.empty([self.max_size], dtype=bool)

    self.obs[self.curr_idx, :] = data['obs']
    self.action[self.curr_idx] = data['act']
    self.reward[self.curr_idx] = data['rew']
    self.obs_n[self.curr_idx] = data['obs_n']
    self.term[self.curr_idx] = data['term']
    self.trunc[self.curr_idx] = data['trunc']

    self.curr_idx = (self.curr_idx + 1) % self.max_size
    self.curr_size = min(self.curr_size + 1, self.max_size)

  def sample(self, batch_size, **kwargs):
    inds = np.random.choice(self.curr_size, batch_size)

    obs = self.obs[inds, :]
    act = self.action[inds, :]
    rew = self.reward[inds]
    obs_n = self.obs_n[inds, :]
    done = self.term[inds]
    term = self.term[inds]
    trunc = self.trunc[inds]

    return {'obs': obs, 'act': act, 'rew': rew, 'obs_n': obs_n,
            'done': done, 'term': term, 'trunc': trunc, 'inds': inds}

  def sample_constant_size(self, batch_size, n_actions):
    act_sz = batch_size // n_actions

    batch = {'obs': [], 'act': [], 'rew': [], 'obs_n': [], 'done': [], 'inds': []}
    for a in range(n_actions):
      # TODO: this where clause will break because I added multi-dim actions
      a_inds = np.where(self.action[:self.curr_size] == a)[0]
      a_inds_inds = np.random.choice(len(a_inds), act_sz)
      inds = a_inds[a_inds_inds]
      batch['obs'].append(self.obs[inds, :])
      batch['act'].append(self.action[inds, :])
      batch['rew'].append(self.reward[inds])
      batch['obs_n'].append(self.obs_n[inds, :])
      batch['done'].append(self.term[inds])
      batch['term'].append(self.term[inds])
      batch['trunc'].append(self.trunc[inds])
      batch['inds'].append(inds)

    batch = {k: (np.vstack(v) if v[0].ndim > 1 else np.hstack(v)) for k, v in batch.items()}
    return batch


class NStepReplayBuffer(ReplayBuffer):
  def __init__(self, max_size=1e5, dtype=np.float32, n=4):
    super().__init__(max_size, dtype)
    self.n = n

  def sample(self, batch_size, gamma=0.99):
    inds = np.random.choice(self.curr_size, batch_size)

    obs = self.obs[inds, :]
    act = self.action[inds, :]

    rew = self.reward[inds]
    for i in range(self.n):
      term = self.term[inds]
      trunc = self.trunc[inds]
      ep_end = np.logical_or.reduce((term, trunc, inds + 1 >= self.curr_size))

      not_ep_end = np.logical_not(ep_end)
      inds += not_ep_end.astype(int)
      rew += gamma * not_ep_end * self.reward[inds]

    obs_n = self.obs_n[inds, :]
    done = self.term[inds]
    term = self.term[inds]
    trunc = self.trunc[inds]

    return {'obs' : obs, 'act': act, 'rew': rew, 'obs_n': obs_n,
            'done': done, 'term': term, 'trunc': trunc, 'inds': inds}


class MonteCarloReplayBuffer(ReplayBuffer):
  def __init__(self, max_size=1e5, dtype=np.float32):
    super().__init__(max_size, dtype)

    self.mc_return = None
    self.mc_samples = None

  def store(self, data):
    if self.mc_return is None:
      self.mc_return = np.empty([self.max_size], dtype=self.dtype)
      self.mc_samples = np.empty([self.max_size], dtype=int)

    self.mc_return[self.curr_idx] = data.get('mc_ret', 0.)
    self.mc_samples[self.curr_idx] = 0

    super().store(data)

  def _add_mc_samples(self, sample):
    mc_ret = self.mc_return[sample['inds']]

    mc_samp = self.mc_samples[sample['inds']]
    self.mc_samples[sample['inds']] += 1

    return {**sample, 'mc_ret': mc_ret, 'mc_samp': mc_samp}

  def sample(self, batch_size, **kwargs):
    sample = super().sample(batch_size, **kwargs)
    return self._add_mc_samples(sample)

  def sample_constant_size(self, batch_size, n_actions):
    sample = super().sample_constant_size(batch_size, n_actions)
    return self._add_mc_samples(sample)

  def update_mc_samples(self, batch, mc_ret):
    self.mc_return[batch['inds']] = mc_ret
