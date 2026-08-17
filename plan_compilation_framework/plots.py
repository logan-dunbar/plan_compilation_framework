"""Thesis figures from results/ parquet files.

  uv run python -m plan_compilation_framework.plots            # everything present
  uv run python -m plan_compilation_framework.plots paper      # one experiment

Writes results/figures/{experiment}_{kind}.{png,pdf}. Curves are rolling
means over episodes with an inter-quartile band across seeds; summary tables
report the last 10% of episodes ('tail') and totals, mean +- s.d. over seeds.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS = 'results'
FIGURES = os.path.join(RESULTS, 'figures')

LABELS = {
  'qlearning': 'Q-learning', 'astar': 'A*', 'rtdp': 'RTDP', 'lrtdp': 'LRTDP',
  'pcf_astar': 'PC-A*', 'pcf_rtdp': 'PC-RTDP', 'base': 'PCF (base)',
}
COLORS = {
  'qlearning': '#7f7f7f', 'astar': '#1f77b4', 'rtdp': '#2ca02c', 'lrtdp': '#17becf',
  'pcf_astar': '#d62728', 'pcf_rtdp': '#ff7f0e', 'base': '#d62728',
}
ORDER = ['qlearning', 'astar', 'rtdp', 'lrtdp', 'pcf_astar', 'pcf_rtdp', 'base']

plt.rcParams.update({
  'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False,
  'legend.frameon': False, 'figure.dpi': 120,
})


def load(exp: str) -> dict[str, list[pd.DataFrame]]:
  runs = {}
  for agent_dir in sorted(glob.glob(os.path.join(RESULTS, exp, '*/'))):
    agent = os.path.basename(agent_dir.rstrip('/'))
    dfs = [pd.read_parquet(p) for p in sorted(glob.glob(agent_dir + 'seed_*.parquet'))]
    if dfs:
      runs[agent] = dfs
  return runs


def _exp_col(d):
  return d['expansions'] if 'expansions' in d else pd.Series(0., index=d.index)


def _order(agents):
  return sorted(agents, key=lambda a: (ORDER.index(a) if a in ORDER else 99, a))


def _style(agent, i):
  return dict(color=COLORS.get(agent, plt.cm.tab20(i % 20)), label=LABELS.get(agent, agent))


def curves(exp: str, runs, window: int | None = None, reward_ylim=None):
  n_ep = len(next(iter(runs.values()))[0])
  window = window or max(1, n_ep // 50)
  agents = _order(runs)

  fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
  for i, agent in enumerate(agents):
    dfs = runs[agent]
    st = _style(agent, i)
    rew = np.stack([d.reward.rolling(window, min_periods=1).mean().to_numpy() for d in dfs])
    x = np.arange(rew.shape[1])
    axes[1].plot(x, np.median(rew, 0), lw=1.5, **st)
    axes[1].fill_between(x, np.percentile(rew, 25, 0), np.percentile(rew, 75, 0),
                         color=st['color'], alpha=0.12, lw=0)
    ex = np.stack([_exp_col(d).rolling(window, min_periods=1).mean().to_numpy() for d in dfs])
    if ex.max() > 0:
      axes[0].plot(x, np.maximum(np.median(ex, 0), 0.05), lw=1.5, **st)

  axes[0].set_yscale('log')
  axes[0].set_xlabel('Episode'); axes[0].set_ylabel('States expanded per episode (log)')
  axes[0].set_title(f'{exp}: search work'); axes[0].legend(fontsize=8)
  axes[1].set_xlabel('Episode'); axes[1].set_ylabel(f'Reward per episode (rolling {window})')
  axes[1].set_title(f'{exp}: reward (median, IQR over seeds)')
  if reward_ylim:
    axes[1].set_ylim(*reward_ylim)
  else:
    lo = min(np.percentile(np.stack([d.reward.rolling(window, min_periods=1).mean().to_numpy()
                                     for d in runs[a]]), 25, 0)[n_ep // 10:].min()
             for a in agents)
    axes[1].set_ylim(lo * 1.15, 0)
  axes[1].legend(fontsize=8, loc='lower right')
  fig.tight_layout()
  _save(fig, f'{exp}_curves')


def summary_table(exp: str, runs) -> pd.DataFrame:
  rows = []
  for agent in _order(runs):
    dfs = runs[agent]
    n_tail = max(1, len(dfs[0]) // 10)
    tail_rew = np.array([d.reward.tail(n_tail).mean() for d in dfs])
    early_rew = np.array([d.reward.head(n_tail).mean() for d in dfs])
    tail_exp = np.array([_exp_col(d).tail(n_tail).mean() for d in dfs])
    total_exp = np.array([_exp_col(d).sum() for d in dfs])
    learnt = np.array([d['n_learnt'].iloc[-1] if 'n_learnt' in d else 0 for d in dfs])
    trunc = sum(int((~d.terminated).sum()) for d in dfs)
    rows.append(dict(agent=agent, seeds=len(dfs),
                     early_reward=f'{early_rew.mean():.0f} ± {early_rew.std():.0f}',
                     tail_reward=f'{tail_rew.mean():.0f} ± {tail_rew.std():.0f}',
                     tail_expansions=f'{tail_exp.mean():.1f}',
                     total_expansions=f'{total_exp.mean():,.0f}',
                     learnt=f'{learnt.mean():.0f}', truncated=trunc))
  df = pd.DataFrame(rows).set_index('agent')
  os.makedirs(FIGURES, exist_ok=True)
  df.to_csv(os.path.join(FIGURES, f'{exp}_summary.csv'))
  with open(os.path.join(FIGURES, f'{exp}_summary.md'), 'w') as f:
    f.write(df.to_markdown())
  return df


def bars(exp: str, runs):
  """Two-panel bar chart: tail reward and total expansions per agent (for ablations)."""
  agents = _order(runs)
  n_tail = max(1, len(runs[agents[0]][0]) // 10)
  tail = [np.array([d.reward.tail(n_tail).mean() for d in runs[a]]) for a in agents]
  tot = [np.array([_exp_col(d).sum() for d in runs[a]]) for a in agents]
  fig, axes = plt.subplots(1, 2, figsize=(12, 3.8 + 0.15 * len(agents)))
  y = np.arange(len(agents))
  labels = [LABELS.get(a, a) for a in agents]
  axes[0].barh(y, [t.mean() for t in tail], xerr=[t.std() for t in tail], color='#d62728', alpha=0.8)
  axes[0].set_yticks(y); axes[0].set_yticklabels(labels); axes[0].invert_yaxis()
  axes[0].set_xlabel('Tail reward (last 10% of episodes)'); axes[0].set_title(f'{exp}: reward')
  axes[1].barh(y, [t.mean() for t in tot], xerr=[t.std() for t in tot], color='#1f77b4', alpha=0.8)
  axes[1].set_yticks(y); axes[1].set_yticklabels([]); axes[1].invert_yaxis()
  axes[1].set_xscale('log'); axes[1].set_xlabel('Total states expanded (log)')
  axes[1].set_title(f'{exp}: search work')
  fig.tight_layout()
  _save(fig, f'{exp}_bars')


def paired(exp: str, runs, pairs=(('pcf_astar', 'astar'), ('pcf_rtdp', 'rtdp'), ('pcf_astar', 'lrtdp'))):
  """Per-seed tail reward of PCF vs its planner on the same grid."""
  pairs = [(a, b) for a, b in pairs if a in runs and b in runs]
  if not pairs:
    return
  fig, axes = plt.subplots(1, len(pairs), figsize=(4.2 * len(pairs), 4), squeeze=False)
  for ax, (a, b) in zip(axes[0], pairs):
    n_tail = max(1, len(runs[a][0]) // 10)
    xa = [d.reward.tail(n_tail).mean() for d in runs[a]]
    xb = [d.reward.tail(n_tail).mean() for d in runs[b]]
    lo, hi = min(xa + xb) * 1.05, max(xa + xb) * 0.95
    ax.plot([lo, hi], [lo, hi], color='#999', lw=1, ls='--')
    ax.scatter(xb, xa, color=COLORS.get(a, '#d62728'), zorder=3)
    for i, (x_, y_) in enumerate(zip(xb, xa)):
      ax.annotate(str(i), (x_, y_), fontsize=7, xytext=(3, 3), textcoords='offset points')
    wins = sum(1 for p, q in zip(xa, xb) if p > q)
    ax.set_xlabel(f'{LABELS.get(b, b)} tail reward'); ax.set_ylabel(f'{LABELS.get(a, a)} tail reward')
    ax.set_title(f'{LABELS.get(a, a)} beats {LABELS.get(b, b)} on {wins}/{len(xa)} grids')
  fig.tight_layout()
  _save(fig, f'{exp}_paired')


def _save(fig, name):
  os.makedirs(FIGURES, exist_ok=True)
  fig.savefig(os.path.join(FIGURES, f'{name}.png'), dpi=160)
  fig.savefig(os.path.join(FIGURES, f'{name}.pdf'))
  plt.close(fig)
  print(f'  wrote {FIGURES}/{name}.png')


def make_all(exp: str):
  runs = load(exp)
  if not runs:
    print(f'{exp}: no results'); return
  print(f'== {exp} ({len(runs)} agents)')
  df = summary_table(exp, runs)
  print(df.to_string(), '\n')
  curves(exp, runs)
  if len(runs) > 6 or 'base' in runs:
    bars(exp, runs)
  paired(exp, runs)


if __name__ == '__main__':
  exps = sys.argv[1:] or [os.path.basename(p.rstrip('/'))
                          for p in sorted(glob.glob(os.path.join(RESULTS, '*/')))
                          if os.path.basename(p.rstrip('/')) != 'figures']
  for e in exps:
    make_all(e)
