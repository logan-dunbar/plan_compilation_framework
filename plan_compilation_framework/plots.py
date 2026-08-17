"""Thesis figures from results/ parquet files (seaborn styling).

  uv run python -m plan_compilation_framework.plots            # everything present
  uv run python -m plan_compilation_framework.plots paper      # one experiment

Writes results/figures/{experiment}_{kind}.{png,pdf}, plus a summary table
(csv + markdown) per experiment. Curves are rolling means over episodes
(window = 5% of the run) with an inter-quartile band across seeds.

Colour policy: reference agents have fixed, neutral styles (PCF base black,
A* / Q-learning grey with distinct dashes); every other agent gets a colour
from a categorical palette chosen for the number of curves. Experiments with
many variants (the sensitivity sweep) are drawn as grouped panels, each
panel repeating the reference curves.
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
import seaborn as sns

RESULTS = 'results'
FIGURES = os.path.join(RESULTS, 'figures')

LABELS = {
  'qlearning': 'Q-learning', 'astar': 'A*', 'rtdp': 'RTDP', 'lrtdp': 'LRTDP',
  'pcf_astar': 'PC-A*', 'pcf_rtdp': 'PC-RTDP', 'base': 'PCF (base)',
  'lh_model': 'Learned heuristic (model)', 'lh_env': 'Learned heuristic (env)',
  'dynaq': 'Dyna-Q', 'shaped_q': 'Plan-shaped Q',
}
# reference agents: fixed neutral styles so they read the same in every figure
REFERENCE = {
  'base':      dict(color='black',   ls='-',  lw=2.2),
  'astar':     dict(color='#555555', ls='--', lw=1.6),
  'qlearning': dict(color='#999999', ls=':',  lw=1.8),
}
ORDER = ['qlearning', 'astar', 'rtdp', 'lrtdp', 'lh_model', 'lh_env', 'dynaq',
         'shaped_q', 'pcf_astar', 'pcf_rtdp', 'base']

# grouped panels for wide sweeps: panel title -> agent-name prefixes/names
GROUPS = {
  'sensitivity_50x50': {
    'Learning rate': ['alpha_const_0.1', 'alpha_per_episode', 'alpha_learnt_0.4'],
    'Exploration': ['const_eps_0.1', 'longer_explore', 'no_explorer'],
    'Labelling': ['alpha_l_0.1', 'tau_d_0.3', 'tau_l_0.95', 'no_latch_slow'],
    'Initialisation & updates': ['q0_-300', 'q0_-5000', 'td_everywhere'],
  },
  'ablations': {
    'Value function': ['A1_optimistic_q0', 'A2_td_everywhere', 'A4_no_latch'],
    'Planning': ['A3_no_horizon_reduction', 'A6_no_value_ordering', 'R1_plan_cache'],
    'Exploration': ['A5_no_explorer', 'A5_unbounded_quota'],
  },
}

sns.set_theme(style='whitegrid', context='paper', font_scale=1.15,
              rc={'axes.spines.top': False, 'axes.spines.right': False,
                  'legend.frameon': False, 'figure.dpi': 120})


# ------------------------------------------------------------------ loading

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


def _label(agent):
  return LABELS.get(agent, agent)


def _styles(agents):
  """Fixed styles for reference agents, a categorical palette for the rest."""
  others = [a for a in agents if a not in REFERENCE]
  n = len(others)
  pal = (sns.color_palette('colorblind', n) if n <= 8 else
         sns.color_palette('tab10', n) if n <= 10 else
         sns.color_palette('husl', n))
  styles = {a: dict(color=pal[i], ls='-', lw=1.6) for i, a in enumerate(others)}
  styles.update({a: REFERENCE[a] for a in agents if a in REFERENCE})
  return styles


def _rolled(dfs, col_fn, window):
  return np.stack([col_fn(d).rolling(window, min_periods=1).mean().to_numpy() for d in dfs])


# ------------------------------------------------------------------ figures

def _draw_curves(axes, runs, agents, styles, window):
  ax_exp, ax_rew = axes
  for agent in agents:
    dfs = runs[agent]
    st = styles[agent]
    rew = _rolled(dfs, lambda d: d.reward, window)
    x = np.arange(rew.shape[1])
    ax_rew.plot(x, np.median(rew, 0), label=_label(agent), **st)
    ax_rew.fill_between(x, np.percentile(rew, 25, 0), np.percentile(rew, 75, 0),
                        color=st['color'], alpha=0.10, lw=0)
    ex = _rolled(dfs, _exp_col, window)
    if ex.max() > 0:
      ax_exp.plot(x, np.maximum(np.median(ex, 0), 0.05), label=_label(agent), **st)
  ax_exp.set_yscale('log')


def _reward_ylim(runs, agents, window):
  n_ep = len(next(iter(runs.values()))[0])
  lo = min(np.percentile(_rolled(runs[a], lambda d: d.reward, window), 25, 0)[n_ep // 10:].min()
           for a in agents)
  return lo * 1.15, 0


def curves(exp: str, runs):
  n_ep = len(next(iter(runs.values()))[0])
  window = max(1, n_ep // 20)
  agents = _order(runs)
  styles = _styles(agents)

  fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
  _draw_curves(axes, runs, agents, styles, window)
  axes[0].set(xlabel='Episode', ylabel='States expanded per episode (log)',
              title='Search work')
  axes[1].set(xlabel='Episode', ylabel=f'Reward per episode (rolling {window})',
              title='Reward (median, IQR over seeds)', ylim=_reward_ylim(runs, agents, window))
  axes[1].legend(fontsize=9, loc='lower right', ncol=1 if len(agents) <= 7 else 2)
  fig.suptitle(exp, y=1.02, fontsize=12)
  fig.tight_layout()
  _save(fig, f'{exp}_curves')


def grouped_curves(exp: str, runs, groups: dict):
  """One row of (expansions, reward) per group; reference agents in every row."""
  n_ep = len(next(iter(runs.values()))[0])
  window = max(1, n_ep // 20)
  refs = [a for a in ('base', 'astar', 'qlearning') if a in runs]
  ylim = _reward_ylim(runs, list(runs), window)

  rows = [(title, [a for a in members if a in runs]) for title, members in groups.items()]
  rows = [(t, m) for t, m in rows if m]
  fig, axes = plt.subplots(len(rows), 2, figsize=(12.5, 3.9 * len(rows)), squeeze=False)
  for (title, members), (ax_exp, ax_rew) in zip(rows, axes):
    agents = refs + members
    styles = _styles(agents)
    _draw_curves((ax_exp, ax_rew), runs, agents, styles, window)
    ax_exp.set(ylabel='States expanded / episode (log)', title=f'{title}: search work')
    ax_rew.set(ylabel=f'Reward / episode (rolling {window})', title=f'{title}: reward',
               ylim=ylim)
    ax_rew.legend(fontsize=8.5, loc='lower right')
  for ax in axes[-1]:
    ax.set_xlabel('Episode')
  fig.suptitle(exp, y=1.005, fontsize=12)
  fig.tight_layout()
  _save(fig, f'{exp}_curves')


def bars(exp: str, runs):
  """Tail reward and total expansions per agent, mean +- s.d. over seeds."""
  agents = _order(runs)
  n_tail = max(1, len(runs[agents[0]][0]) // 10)
  rows = []
  for a in agents:
    for i, d in enumerate(runs[a]):
      rows.append(dict(agent=_label(a), seed=i, tail_reward=d.reward.tail(n_tail).mean(),
                       total_expansions=max(_exp_col(d).sum(), 1.)))
  df = pd.DataFrame(rows)
  order = [_label(a) for a in agents]
  styles = _styles(agents)
  palette = {_label(a): styles[a]['color'] for a in agents}

  fig, axes = plt.subplots(1, 2, figsize=(12.5, 3.6 + 0.28 * len(agents)))
  sns.barplot(df, y='agent', x='tail_reward', order=order, hue='agent', palette=palette,
              errorbar='sd', err_kws={'linewidth': 1.2}, capsize=0.2, legend=False, ax=axes[0])
  sns.barplot(df, y='agent', x='total_expansions', order=order, hue='agent', palette=palette,
              errorbar='sd', err_kws={'linewidth': 1.2}, capsize=0.2, legend=False, ax=axes[1])
  axes[0].set(xlabel='Tail reward (last 10% of episodes)', ylabel='', title='Reward')
  axes[1].set(xscale='log', xlabel='Total states expanded (log)', ylabel='',
              title='Search work', yticklabels=[])
  fig.suptitle(exp, y=1.01, fontsize=12)
  fig.tight_layout()
  _save(fig, f'{exp}_bars')


def paired(exp: str, runs,
           pairs=(('pcf_astar', 'astar'), ('pcf_rtdp', 'rtdp'), ('pcf_astar', 'lrtdp'),
                  ('pcf_astar', 'qlearning'))):
  """Per-seed tail reward of PCF vs a baseline on the same grid."""
  pairs = [(a, b) for a, b in pairs if a in runs and b in runs]
  if not pairs:
    return
  fig, axes = plt.subplots(1, len(pairs), figsize=(4.1 * len(pairs), 4), squeeze=False)
  for ax, (a, b) in zip(axes[0], pairs):
    n_tail = max(1, len(runs[a][0]) // 10)
    ya = [d.reward.tail(n_tail).mean() for d in runs[a]]
    xb = [d.reward.tail(n_tail).mean() for d in runs[b]]
    lo, hi = min(ya + xb) * 1.05, max(ya + xb) * 0.95
    ax.plot([lo, hi], [lo, hi], color='#999', lw=1, ls='--')
    sns.scatterplot(x=xb, y=ya, ax=ax, s=55, color=sns.color_palette('colorblind')[3],
                    edgecolor='white', zorder=3)
    for i, (x_, y_) in enumerate(zip(xb, ya)):
      ax.annotate(str(i), (x_, y_), fontsize=7, xytext=(4, 3), textcoords='offset points')
    wins = sum(1 for p, q in zip(ya, xb) if p > q)
    ax.set(xlabel=f'{_label(b)} tail reward', ylabel=f'{_label(a)} tail reward',
           title=f'{_label(a)} beats {_label(b)} on {wins}/{len(ya)} grids', xlim=(lo, hi),
           ylim=(lo, hi))
  fig.suptitle(exp, y=1.02, fontsize=12)
  fig.tight_layout()
  _save(fig, f'{exp}_paired')


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


def _save(fig, name):
  os.makedirs(FIGURES, exist_ok=True)
  fig.savefig(os.path.join(FIGURES, f'{name}.png'), dpi=160, bbox_inches='tight')
  fig.savefig(os.path.join(FIGURES, f'{name}.pdf'), bbox_inches='tight')
  plt.close(fig)
  print(f'  wrote {FIGURES}/{name}.png')


def make_all(exp: str):
  runs = load(exp)
  if not runs:
    print(f'{exp}: no results'); return
  print(f'== {exp} ({len(runs)} agents)')
  df = summary_table(exp, runs)
  print(df.to_string(), '\n')
  if exp in GROUPS:
    grouped_curves(exp, runs, GROUPS[exp])
  else:
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
