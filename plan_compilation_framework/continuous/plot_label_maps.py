"""The learnt-label maps behind the Chapter 4 curves: WHERE the labels sit and which are lies.

  uv run python -m plan_compilation_framework.continuous.plot_label_maps

One row per learner, one column per snapshot episode. Each probe (a free-cell centre) is

  blue    labelled learnt, greedy action agrees with the end-of-run policy
  red     labelled learnt, greedy action DISAGREES with the end-of-run policy (false-learnt)
  grey    not labelled learnt

Writes results/figures/continuous_label_maps.{png,pdf}.
"""
from __future__ import annotations

import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RESULTS = 'results'
LABELS = {'tabular': 'Tabular (cells)', 'tilecoding': 'Tile coding', 'rff_linear': 'RFF + linear',
          'rff_mlp': 'RFF + neural network', 'mlp': 'Neural network', 'gp': 'Gaussian process'}
ROWS = ['tabular', 'rff_linear', 'mlp', 'gp']
EPISODES = [75, 375, 1500]
SEED = 0

C_OK, C_FALSE, C_OFF = '#2b6cb0', '#c53030', '#d0d0d0'


def main():
  fig, axes = plt.subplots(len(ROWS), len(EPISODES),
                           figsize=(2.6 * len(EPISODES), 2.45 * len(ROWS)),
                           squeeze=False)
  for r, name in enumerate(ROWS):
    d = np.load(os.path.join(RESULTS, 'continuous', name, f'seed_{SEED}_audit.npz'))
    labels, greedy, probes = d['labels'], d['greedy'], d['probes']
    final = greedy[-1]
    for c, ep in enumerate(EPISODES):
      i = min(ep, labels.shape[0]) - 1
      lab = labels[i]
      false = lab & (greedy[i] != final)
      ax = axes[r][c]
      ax.scatter(*probes[~lab].T, c=C_OFF, s=13, marker='s', lw=0)
      ax.scatter(*probes[lab & ~false].T, c=C_OK, s=13, marker='s', lw=0)
      ax.scatter(*probes[false].T, c=C_FALSE, s=13, marker='s', lw=0)
      ax.set_aspect('equal')
      ax.set_xticks([]); ax.set_yticks([])
      if r == 0:
        ax.set_title(f'episode {ep}', fontsize=11)
      if c == 0:
        ax.set_ylabel(LABELS[name], fontsize=11)
  handles = [Line2D([], [], marker='s', ls='', color=c, label=l, markersize=7)
             for c, l in ((C_OK, 'learnt, agrees with final policy'),
                          (C_FALSE, 'learnt, greedy action still wrong'),
                          (C_OFF, 'not learnt'))]
  fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=10)
  fig.tight_layout(rect=(0, 0.045, 1, 1))
  out = os.path.join(RESULTS, 'figures')
  os.makedirs(out, exist_ok=True)
  for ext in ('png', 'pdf'):
    fig.savefig(os.path.join(out, f'continuous_label_maps.{ext}'),
                dpi=170, bbox_inches='tight')
  print('  wrote', os.path.join(out, 'continuous_label_maps.pdf'))


if __name__ == '__main__':
  main()
