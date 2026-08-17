import matplotlib
import matplotlib.pyplot as plt


class Plotter:
  def __init__(self, plot_freq_ep, pause=0.1, title=None):
    self.fig, self.ax = plt.subplots(figsize=(6, 3))
    self.plot_freq_ep = plot_freq_ep
    self.pause = pause
    self.art = None

    if title is not None:
      self.fig.suptitle(title)

  def plot_step(self, t, agent):
    pass

  def plot_episode(self, e, ep_rews, ep_lens, agent):
    if e % self.plot_freq_ep == 0:
      if self.art is not None:
        self.art.remove()

      self.art = self.ax.plot(ep_rews, color='tab:blue')[0]

      plt.show(block=False)
      plt.pause(self.pause)
