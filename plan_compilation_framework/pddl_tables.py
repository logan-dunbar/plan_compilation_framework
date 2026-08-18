"""Tables for the PDDL chapter: the optimal-vs-satisficing-vs-PCF frontier.

  uv run python -m plan_compilation_framework.pddl_tables            # all domains with probes
  uv run python -m plan_compilation_framework.pddl_tables puzzle15   # one domain

For each domain with a results/pddl_<domain>_probe/ directory: from the same
random starts, how often did the optimal planner (A* + LM-cut, time-capped)
solve, at what cost and time; what did the satisficing planners cost; and
where did PCF end up (tail cost and search) from results/pddl_<domain>/.
Writes results/figures/pddl_frontier.{md,csv,tex}.
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

RESULTS = 'results'
FIGURES = os.path.join(RESULTS, 'figures')

PROBES = {'probe_opt': 'A* LM-cut (optimal)', 'probe_gff': 'lazy greedy FF', 'probe_g': 'lazy greedy CEA',
          'probe_gff_unit': 'lazy greedy FF, unit-cost model'}
PCF_AGENTS = {'pcf_fd_g': 'PC-FD-G', 'pcf_fd_gff': 'PC-FD-GFF', 'pcf_fd_gff_unit': 'PC-FD-GFF (unit-cost model)'}
PLANNERS = {'fd_g': 'FD-G', 'fd_gff': 'FD-GFF', 'fd_gff_unit': 'FD-GFF (unit-cost model)'}


def _load_seed_files(path):
  return [pd.read_parquet(p) for p in sorted(glob.glob(os.path.join(path, 'seed_*.parquet')))]


def probe_summary(domain: str):
  rows = []
  base = os.path.join(RESULTS, f'pddl_{domain}_probe')
  for agent, label in PROBES.items():
    dfs = _load_seed_files(os.path.join(base, agent))
    if not dfs:
      continue
    d = pd.concat(dfs)
    solved = d.terminated.astype(bool)
    rows.append(dict(
      domain=domain, kind='probe', agent=label,
      starts=len(d), solved_pct=100. * solved.mean(),
      cost_solved=(-d.reward[solved]).mean() if solved.any() else np.nan,
      time_median=d.plan_time.median(), time_max=d.plan_time.max(),
      expansions_median=d.expansions.median()))
  return rows


def run_summary(domain: str):
  rows = []
  base = os.path.join(RESULTS, f'pddl_{domain}')
  for agent, label in {**PLANNERS, **PCF_AGENTS}.items():
    dfs = _load_seed_files(os.path.join(base, agent))
    if not dfs:
      continue
    n_tail = max(1, len(dfs[0]) // 10)
    tail_cost = np.array([-d.reward.tail(n_tail).mean() for d in dfs])
    tail_exp = np.array([d.expansions.tail(n_tail).mean() for d in dfs])
    tail_time = np.array([d.plan_time.tail(n_tail).mean() for d in dfs])
    rows.append(dict(
      domain=domain, kind='run', agent=label, starts=len(dfs[0]) * len(dfs),
      solved_pct=100. * np.mean([d.terminated.astype(bool).mean() for d in dfs]),
      cost_solved=tail_cost.mean(), time_median=tail_time.mean(),
      time_max=np.nan, expansions_median=tail_exp.mean(),
      cost_sd=tail_cost.std(), episodes=len(dfs[0]), seeds=len(dfs)))
  return rows


def main(domains):
  rows = []
  for d in domains:
    rows += probe_summary(d)
    rows += run_summary(d)
  if not rows:
    print('no probe/run results found'); return
  df = pd.DataFrame(rows)
  os.makedirs(FIGURES, exist_ok=True)
  df.to_csv(os.path.join(FIGURES, 'pddl_frontier.csv'), index=False)

  lines = []
  for d in domains:
    sub = df[df.domain == d]
    if sub.empty:
      continue
    lines.append(f'\n### {d}\n')
    lines.append('| agent | starts / episodes | solved % | cost | search time (s) | expansions |')
    lines.append('|---|---|---|---|---|---|')
    for _, r in sub.iterrows():
      if r.kind == 'probe':
        lines.append(f"| {r.agent} | {int(r.starts)} starts | {r.solved_pct:.0f}% | {r.cost_solved:.1f} | "
                     f"median {r.time_median:.3f}, max {r.time_max:.1f} | {r.expansions_median:,.0f} |")
      else:
        lines.append(f"| {r.agent} (tail) | {int(r.episodes)} eps x {int(r.seeds)} seeds | {r.solved_pct:.0f}% | "
                     f"{r.cost_solved:.1f} ± {r.cost_sd:.1f} | {r.time_median:.4f} per episode | {r.expansions_median:,.1f} per episode |")
  md = '\n'.join(lines)
  with open(os.path.join(FIGURES, 'pddl_frontier.md'), 'w') as f:
    f.write(md)
  print(md)


if __name__ == '__main__':
  if len(sys.argv) > 1:
    doms = sys.argv[1:]
  else:
    doms = sorted(os.path.basename(p.rstrip('/'))[5:-6]
                  for p in glob.glob(os.path.join(RESULTS, 'pddl_*_probe/')))
  main(doms)
