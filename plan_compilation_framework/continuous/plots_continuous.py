"""Figures for the Chapter 4 experiment.

  uv run python -m plan_compilation_framework.continuous.plots_continuous

Five panels over episodes (median over seeds, IQR band): fraction of probe
states labelled learnt, label churn per episode, false-learnt rate, reward and
states expanded; plus a summary table. Writes
results/figures/continuous_*.{png,pdf}.

The panels are laid out three over two rather than five across. Five across is
what fits the data and not what fits the page: at \textwidth each panel is
about 32 mm wide, which is too narrow to read a curve out of. Three over two
gives each panel roughly 2.5x the area for the same page width, and the
gridspec below is six columns wide so the bottom row of two can be centred
under the top row of three.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS = 'results'
FIGURES = os.path.join(RESULTS, 'figures')
LABELS = {'tabular': 'Tabular (cells)', 'tilecoding': 'Tile coding', 'mlp': 'Neural network', 'gp': 'Gaussian process',
          'rff_linear': 'RFF + linear', 'rff_mlp': 'RFF + neural network'}
ORDER = ['tabular', 'tilecoding', 'rff_linear', 'gp', 'rff_mlp', 'mlp']
sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
              rc={'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})


def load():
  runs = {}
  for d in sorted(glob.glob(os.path.join(RESULTS, 'continuous', '*/'))):
    name = os.path.basename(d.rstrip('/'))
    dfs = [pd.read_parquet(p) for p in sorted(glob.glob(d + 'seed_*.parquet'))]
    if dfs:
      runs[name] = dfs
  return runs


def main():
  runs = load()
  if not runs:
    print('no results'); return
  names = [n for n in ORDER if n in runs] + [n for n in runs if n not in ORDER]
  pal = dict(zip(names, ['black'] + list(sns.color_palette('colorblind', len(names) - 1))))
  n_ep = len(runs[names[0]][0])
  w = max(1, n_ep // 30)

  panels = [('learnt_frac', 'Fraction of probe states labelled learnt', False),
            ('churn', 'Label churn per episode', True),
            ('false_learnt', 'False-learnt rate', False),
            ('reward', 'Reward per episode', False),
            ('expansions', 'States expanded per episode', True)]
  # Six columns so the bottom row of two centres under the top row of three:
  # top spans [0:2] [2:4] [4:6], bottom spans [1:3] [3:5], leaving half a panel
  # of margin at each end of the second row.
  # 10.5in wide, not 15: the figure is placed at \textwidth (about 160 mm), so a
  # wider canvas is only downscaled harder and the text shrinks with it. At 15in
  # a 10pt label renders at about 4.4pt; at 10.5in it renders at 6.3pt for the
  # same footprint on the page. Aspect is what controls the footprint, not size.
  fig = plt.figure(figsize=(10.5, 5.9))
  # bottom=0.19 reserves a strip for the shared legend: at 0.13 it sat on top of
  # the second row's "Episode" labels.
  gs = fig.add_gridspec(2, 6, hspace=0.42, wspace=0.38,
                        left=0.055, right=0.99, top=0.95, bottom=0.19)
  slots = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[1, 1:3], gs[1, 3:5]]
  assert len(slots) == len(panels), (len(slots), len(panels))
  axes = [fig.add_subplot(sl) for sl in slots]
  for ax, (col, title, log) in zip(axes, panels):
    for n in names:
      arr = np.stack([d[col].rolling(w, min_periods=1).mean().to_numpy() for d in runs[n]])
      x = np.arange(arr.shape[1])
      med = np.nanmedian(arr, 0)
      if log:
        med = np.maximum(med, 1e-4)
      ax.plot(x, med, color=pal[n], lw=2 if n == 'tabular' else 1.6, label=LABELS.get(n, n))
      ax.fill_between(x, np.nanpercentile(arr, 25, 0), np.nanpercentile(arr, 75, 0), color=pal[n], alpha=0.12, lw=0)
    if log:
      ax.set_yscale('log')
    ax.set(title=title, xlabel='Episode')
  # One shared legend under the figure rather than one crammed into a panel:
  # with two rows there is no panel with room to spare, and the series are the
  # same in all five.
  handles, labels = axes[0].get_legend_handles_labels()
  fig.legend(handles, labels, loc='lower center', ncol=len(labels),
             fontsize=10.5, bbox_to_anchor=(0.5, 0.005))
  os.makedirs(FIGURES, exist_ok=True)
  fig.savefig(os.path.join(FIGURES, 'continuous_curves.png'), dpi=160, bbox_inches='tight')
  fig.savefig(os.path.join(FIGURES, 'continuous_curves.pdf'), bbox_inches='tight')
  print('  wrote', os.path.join(FIGURES, 'continuous_curves.png'))

  rows = []
  for n in names:
    dfs = runs[n]
    t = max(1, n_ep // 10)
    f = lambda c: np.array([d[c].tail(t).mean() for d in dfs])
    rows.append(dict(learner=LABELS.get(n, n), seeds=len(dfs),
                     tail_reward=f"{f('reward').mean():.0f} ± {f('reward').std():.0f}",
                     tail_expansions=f"{f('expansions').mean():.1f}",
                     learnt_frac=f"{f('learnt_frac').mean():.2f}",
                     churn=f"{np.nanmean(f('churn')):.4f}",
                     policy_churn=f"{np.nanmean(f('policy_churn')):.4f}",
                     false_learnt=f"{f('false_learnt').mean():.3f}",
                     mean_false_learnt_all=f"{np.mean([d.false_learnt.mean() for d in dfs]):.3f}"))
  df = pd.DataFrame(rows).set_index('learner')
  print(df.to_string())
  with open(os.path.join(FIGURES, 'continuous_summary.md'), 'w') as fh:
    fh.write(df.to_markdown())


if __name__ == '__main__':
  main()
