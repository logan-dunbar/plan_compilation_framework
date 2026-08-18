"""Chapter 4 experiment: the framework under four value representations.

  ./run.sh -m plan_compilation_framework.continuous.run_continuous --preset smoke
  ./run.sh -m plan_compilation_framework.continuous.run_continuous --preset full

Per episode the runner records the usual statistics plus an audit of the
learnt-state mechanism on a fixed lattice of probe points (the centres of
all free cells):

  learnt_frac    fraction of probes labelled learnt
  churn          fraction of probes whose label flipped since the last episode
  policy_churn   fraction of probes labelled learnt in both of two consecutive
                 episodes whose greedy action changed between them
  false_learnt   (post hoc) fraction of learnt-labelled probes whose greedy
                 action differs from the greedy action at the end of the run

Outputs results/continuous/<learner>/seed_k.parquet plus an .npz of the
per-episode label and greedy-action matrices for post-hoc analysis.
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import replace
import multiprocessing as mp

import numpy as np
import pandas as pd

os.environ.setdefault('JAX_PLATFORMS', 'cpu')   # 2-D problems: CPU JAX is plenty, and it
                                                 # keeps thirty worker processes off the GPUs

from plan_compilation_framework.continuous.env import ContinuousMaze
from plan_compilation_framework.continuous.agent import ContinuousPCFConfig, ContinuousPlanCompiler
from plan_compilation_framework.continuous.learners import LabelConfig


def audit(agent, probes):
  labels = np.asarray(agent.learner.learnt_batch(probes), dtype=bool)
  greedy = np.argmax(agent.learner.q_batch(probes), axis=1).astype(np.int8)
  return labels, greedy


def run_single(job):
  name, cfg, seed, episodes, out_dir, size = job
  env = ContinuousMaze.random(size=(size, size), seed=seed, max_steps=3000)
  agent = ContinuousPlanCompiler(env, cfg, seed=seed * 7919 + 1)
  probes = env.probe_points()
  n = len(probes)
  labels_hist = np.zeros((episodes, n), dtype=bool)
  greedy_hist = np.zeros((episodes, n), dtype=np.int8)
  prev_labels = prev_greedy = None
  records = []
  t0 = time.perf_counter()
  for ep in range(episodes):
    s = env.reset()
    agent.begin_episode(ep)
    total, steps, terminated, truncated = 0., 0, False, False
    while not (terminated or truncated):
      a = agent.act(s)
      s2, r, terminated, truncated = env.step(a)
      agent.observe(s, a, r, s2, terminated)
      s = s2
      total += r
      steps += 1
    stats = agent.end_episode()
    labels, greedy = audit(agent, probes)
    labels_hist[ep], greedy_hist[ep] = labels, greedy
    churn = policy_churn = np.nan
    if prev_labels is not None:
      churn = float(np.mean(labels != prev_labels))
      both = labels & prev_labels
      policy_churn = float(np.mean(greedy[both] != prev_greedy[both])) if both.any() else 0.
    prev_labels, prev_greedy = labels, greedy
    records.append(dict(episode=ep, reward=total, steps=steps, terminated=terminated,
                        learnt_frac=float(labels.mean()), churn=churn, policy_churn=policy_churn,
                        wall_time=0., cache_hits=0, n_learnt=int(labels.sum()), **stats))
  df = pd.DataFrame.from_records(records)
  # false-learnt: labelled learnt while the greedy action still differs from the final policy
  final = greedy_hist[-1]
  fl = [(float(np.mean(greedy_hist[i][labels_hist[i]] != final[labels_hist[i]])) if labels_hist[i].any() else 0.)
        for i in range(episodes)]
  df['false_learnt'] = fl
  d = os.path.join(out_dir, 'continuous', name)
  os.makedirs(d, exist_ok=True)
  df.to_parquet(os.path.join(d, f'seed_{seed}.parquet'))
  np.savez_compressed(os.path.join(d, f'seed_{seed}_audit.npz'), labels=labels_hist, greedy=greedy_hist,
                      probes=probes)
  tail = df.tail(max(1, episodes // 10))
  return (name, seed, time.perf_counter() - t0, float(tail.reward.mean()), float(tail.expansions.mean()),
          float(tail.learnt_frac.mean()), float(tail.churn.mean()), float(tail.false_learnt.mean()))


def variants(size):
  base = ContinuousPCFConfig(q0=-300., label=LabelConfig())
  return {
    'tabular': replace(base, learner='tabular'),
    'tilecoding': replace(base, learner='tilecoding', learner_kwargs=dict(n_tilings=8, tile_width=1.0)),
    'mlp': replace(base, learner='mlp', learner_kwargs=dict(hidden=64, lr=1e-3)),
    'gp': replace(base, learner='gp', learner_kwargs=dict(lengthscale=1.5, cap=1500, tau_std=10.)),
  }


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--preset', choices=['smoke', 'full'], default='smoke')
  ap.add_argument('--learners', nargs='*', default=None)
  ap.add_argument('--episodes', type=int, default=None)
  ap.add_argument('--seeds', type=int, default=None)
  ap.add_argument('--size', type=int, default=20)
  ap.add_argument('--workers', type=int, default=None)
  ap.add_argument('--out', default='results')
  args = ap.parse_args()
  episodes = args.episodes or (200 if args.preset == 'smoke' else 2000)
  seeds = args.seeds or (1 if args.preset == 'smoke' else 5)
  vs = variants(args.size)
  names = args.learners or list(vs)
  jobs = [(n, vs[n], s, episodes, os.path.abspath(args.out), args.size) for n in names for s in range(seeds)]
  workers = min(args.workers or max(1, os.cpu_count() - 4), len(jobs))
  print(f'continuous/{args.preset}: {len(jobs)} jobs on {workers} workers ({episodes} episodes)', flush=True)
  # spawn, not fork: JAX does not survive fork in worker processes
  with mp.get_context('spawn').Pool(workers) as pool:
    for name, seed, wall, rew, exp, lf, ch, fl in pool.imap_unordered(run_single, jobs):
      print(f'  done {name:>10} seed {seed}: {wall:7.1f}s  tail reward {rew:8.1f}  expansions {exp:7.1f}  '
            f'learnt {lf:5.2f}  churn {ch:6.4f}  false-learnt {fl:5.3f}', flush=True)


if __name__ == '__main__':
  main()
