"""PDDL experiments through the native Fast Downward PCF driver.

  ./run.sh -m plan_compilation_framework.run_pddl --preset smoke
  ./run.sh -m plan_compilation_framework.run_pddl --preset paper --episodes 20000 --seeds 5

For each domain the task is translated once; then every (agent, seed) job
runs the `downward` binary on that SAS file with a `pcf(...)` configuration
and its per-episode CSV is converted to the same parquet layout the GridWorld
runner writes, so plots.py works unchanged:

  results/pddl_<domain>/<agent>/seed_<k>.parquet

Agents mirror the ECML'22 setup: FD-G (lazy greedy, context-enhanced additive
heuristic) and FD-GFF (lazy greedy, FF heuristic) as plan-and-enact
baselines, and PC-FD-G / PC-FD-GFF as the Plan Compilation Framework driving
the same engines. Both baseline and PCF run inside the same driver loop, so
initial states, plan following and expansion accounting are identical.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from multiprocessing import Pool

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FD_ROOT = os.path.abspath(os.path.join(ROOT, '..', 'downward'))
FD_BIN = os.path.join(FD_ROOT, 'builds', 'release', 'bin', 'downward')
FD_PY = os.path.join(FD_ROOT, 'fast-downward.py')
BENCH = os.path.abspath(os.path.join(ROOT, '..', 'downward-benchmarks'))

DOMAINS = {
  'gripper': (f'{BENCH}/gripper/domain.pddl', f'{BENCH}/gripper/prob01.pddl'),
  'depot': (f'{BENCH}/depot/domain.pddl', f'{BENCH}/depot/p01.pddl'),
  'depot3': (f'{BENCH}/depot/domain.pddl', f'{BENCH}/depot/p03.pddl'),
  'blocks15': (f'{BENCH}/blocks/domain.pddl', f'{BENCH}/blocks/probBLOCKS-15-0.pddl'),
  'puzzle15': (f'{ROOT}/domains/sliding_tile/domain.pddl',
               f'{ROOT}/domains/sliding_tile/p15-s80-0.pddl'),
}

ENGINES = {
  'g': ('hcea=cea()', 'lazy_greedy([hcea], preferred=[hcea], verbosity=silent)'),
  'gff': ('hff=ff()', 'lazy_greedy([hff], preferred=[hff], verbosity=silent)'),
  'blind': (None, 'astar(blind(), verbosity=silent)'),
}


@dataclass
class PddlAgent:
  mode: str      # 'planner' | 'pcf'
  engine: str    # key of ENGINES


AGENTS = {
  'fd_g': PddlAgent('planner', 'g'),
  'fd_gff': PddlAgent('planner', 'gff'),
  'pcf_fd_g': PddlAgent('pcf', 'g'),
  'pcf_fd_gff': PddlAgent('pcf', 'gff'),
}


@dataclass
class PddlConfig:
  episodes: int = 20_000
  walk_min: int = 5
  walk_max: int = 30
  max_episode_steps: int = 20_000
  q0: float = -1000.
  anneal_frac: float = 0.08     # explore_prob / quota anneal (800/10k in GridWorld)
  eps_anneal_frac: float = 0.5  # learner epsilon anneal

  def search_string(self, agent: PddlAgent, seed: int, log_file: str) -> str:
    _, engine = ENGINES[agent.engine]
    return (f'pcf({engine}, mode={agent.mode}, episodes={self.episodes}, '
            f'walk_min={self.walk_min}, walk_max={self.walk_max}, '
            f'max_episode_steps={self.max_episode_steps}, q0={self.q0}, '
            f'anneal={max(1, int(self.episodes * self.anneal_frac))}, '
            f'eps_anneal={max(1, int(self.episodes * self.eps_anneal_frac))}, '
            f'log_file={log_file}, log_every=0, random_seed={seed})')


def translate(domain: str, work: str) -> str:
  """Translate once; return the path of output.sas."""
  dom, prob = DOMAINS[domain]
  out = os.path.join(work, domain)
  os.makedirs(out, exist_ok=True)
  sas = os.path.join(out, 'output.sas')
  if not os.path.exists(sas):
    subprocess.run([sys.executable, FD_PY, '--translate', dom, prob],
                   cwd=out, check=True, capture_output=True)
  return sas


def run_job(job):
  domain, agent_name, seed, cfg, sas, out_dir = job
  agent = AGENTS[agent_name]
  evaluator, _ = ENGINES[agent.engine]
  job_dir = os.path.join(out_dir, '.work', domain, agent_name, f'seed_{seed}')
  os.makedirs(job_dir, exist_ok=True)
  log_file = os.path.join(job_dir, 'episodes.csv')

  cmd = [FD_BIN]
  if evaluator:
    cmd += ['--evaluator', evaluator]
  cmd += ['--search', cfg.search_string(agent, seed, log_file)]

  t0 = time.perf_counter()
  with open(sas) as stdin, open(os.path.join(job_dir, 'stdout.txt'), 'w') as stdout:
    proc = subprocess.run(cmd, stdin=stdin, stdout=stdout, stderr=subprocess.STDOUT, cwd=job_dir)
  wall = time.perf_counter() - t0
  if proc.returncode != 0 or not os.path.exists(log_file):
    return (domain, agent_name, seed, wall, None, f'downward exit {proc.returncode}')

  df = pd.read_csv(log_file)
  df['terminated'] = df['terminated'].astype(bool)
  df['cache_hits'] = 0
  df['wall_time'] = df['plan_time']
  res_dir = os.path.join(out_dir, f'pddl_{domain}', agent_name)
  os.makedirs(res_dir, exist_ok=True)
  df.to_parquet(os.path.join(res_dir, f'seed_{seed}.parquet'))
  with open(os.path.join(res_dir, 'config.json'), 'w') as f:
    json.dump({'domain': domain, 'files': DOMAINS[domain], 'agent': asdict(agent),
               'config': asdict(cfg), 'search': cfg.search_string(agent, seed, 'episodes.csv')},
              f, indent=2)
  tail = df.tail(max(1, len(df) // 10))
  return (domain, agent_name, seed, wall, (float(tail.reward.mean()), float(tail.expansions.mean()),
                                            int(df.n_learnt.iloc[-1])), None)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--preset', choices=['smoke', 'paper'], default='smoke')
  ap.add_argument('--domains', nargs='*', default=None)
  ap.add_argument('--agents', nargs='*', default=None)
  ap.add_argument('--episodes', type=int, default=None)
  ap.add_argument('--seeds', type=int, default=None)
  ap.add_argument('--workers', type=int, default=None)
  ap.add_argument('--out', default='results')
  args = ap.parse_args()
  args.out = os.path.abspath(args.out)

  if args.preset == 'smoke':
    domains = args.domains or ['gripper', 'depot', 'blocks15', 'puzzle15']
    agents = args.agents or ['fd_gff', 'pcf_fd_gff']
    cfg = PddlConfig(episodes=args.episodes or 300)
    seeds = args.seeds or 1
  else:
    domains = args.domains or ['depot', 'blocks15', 'puzzle15']
    agents = args.agents or list(AGENTS)
    cfg = PddlConfig(episodes=args.episodes or 20_000)
    seeds = args.seeds or 5

  assert os.path.exists(FD_BIN), f'build Fast Downward first: {FD_BIN}'
  work = os.path.join(args.out, '.work')
  jobs = []
  for d in domains:
    sas = translate(d, work)
    for a in agents:
      for s in range(seeds):
        jobs.append((d, a, s, cfg, sas, args.out))

  workers = min(args.workers or max(1, os.cpu_count() - 4), len(jobs))
  print(f'pddl/{args.preset}: {len(jobs)} jobs on {workers} workers '
        f'({len(domains)} domains x {len(agents)} agents x {seeds} seeds x {cfg.episodes} episodes)',
        flush=True)
  t0 = time.perf_counter()
  with Pool(processes=workers) as pool:
    for d, a, s, wall, summary, err in pool.imap_unordered(run_job, jobs):
      if err:
        print(f'  FAILED {d:>9} {a:>11} seed {s}: {err}', flush=True)
      else:
        rew, exp, learnt = summary
        print(f'  done {d:>9} {a:>11} seed {s}: {wall:7.1f}s  tail reward {rew:8.1f}  '
              f'tail expansions {exp:9.1f}  learnt {learnt}', flush=True)
  print(f'pddl/{args.preset}: finished in {time.perf_counter() - t0:.1f}s', flush=True)


if __name__ == '__main__':
  main()
