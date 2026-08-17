"""Experiment runner: (agent x seed) grid, multiprocessing, parquet output.

Fairness contract: the environment (and therefore the generated grid) is
seeded by the seed index alone, so every agent at seed k faces the same maze.
Agent randomness is seeded independently per (agent, seed).

Output layout:
  {out_dir}/{experiment}/{agent}/seed_{k}.parquet   per-episode records
  {out_dir}/{experiment}/{agent}/config.json        resolved agent + env spec
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict, is_dataclass
from multiprocessing import Pool

import pandas as pd

from plan_compilation_framework.core import Transition
from plan_compilation_framework.environments import GridWorld
from plan_compilation_framework.agents import (
  PCFConfig, PlanCompiler, PlannerAgent, QLearning, QLearningConfig,
)
from plan_compilation_framework.planners import PLANNERS


@dataclass
class EnvSpec:
  kind: str = 'random'              # 'random' | 'file'
  size: tuple = (20, 20)
  file_name: str | None = None
  wall_frac: float = 0.2
  sand_frac: float = 0.25
  min_reachable_frac: float = 0.5
  stochastic: bool = True
  random_start: bool = True
  # hard per-episode step cap. Never None: an agent bug that cycles forever
  # must truncate an episode, not consume the machine (17 Aug 2026 OOM).
  max_steps: int = 1_000_000
  move_reward: float = -1.
  bump_reward: float = -5.
  sand_reward: float = -100.

  def build(self, seed: int) -> GridWorld:
    assert self.max_steps is not None and self.max_steps > 0, \
      'EnvSpec.max_steps must be a positive step cap'
    kwargs = dict(stochastic=self.stochastic, random_start=self.random_start,
                  max_steps=self.max_steps, move_reward=self.move_reward,
                  bump_reward=self.bump_reward, sand_reward=self.sand_reward)
    if self.kind == 'random':
      return GridWorld.random(size=self.size, wall_frac=self.wall_frac,
                              sand_frac=self.sand_frac,
                              min_reachable_frac=self.min_reachable_frac,
                              seed=seed, **kwargs)
    return GridWorld.from_file(self.file_name, seed=seed, **kwargs)


@dataclass
class AgentSpec:
  """kind: 'qlearning' | 'planner' | 'pcf'."""
  kind: str
  planner: str | None = None        # 'astar' | 'rtdp' | 'lrtdp'
  planner_kwargs: dict = field(default_factory=dict)
  config: object = None             # QLearningConfig | PCFConfig | None
  use_cache: bool = False           # planner-agent cache (R1)
  model_constant_cost: bool = True  # the inaccurate-model setting

  def build(self, env: GridWorld, seed: int):
    if self.kind == 'qlearning':
      return QLearning(env, self.config or QLearningConfig(), seed=seed)

    model = env.model(constant_cost=self.model_constant_cost)
    planner = PLANNERS[self.planner](model, seed=seed, **self.planner_kwargs)
    if self.kind == 'planner':
      return PlannerAgent(env, planner, use_cache=self.use_cache, seed=seed)
    if self.kind == 'pcf':
      return PlanCompiler(env, planner, self.config or PCFConfig(), seed=seed)
    raise ValueError(f'unknown agent kind {self.kind!r}')


@dataclass
class ExperimentSpec:
  name: str
  env: EnvSpec
  agents: dict                      # name -> AgentSpec
  episodes: int = 1000
  seeds: int = 3
  out_dir: str = 'results'


def _jsonable(obj):
  if is_dataclass(obj) and not isinstance(obj, type):
    return {k: _jsonable(v) for k, v in asdict(obj).items()}
  if isinstance(obj, dict):
    return {str(k): _jsonable(v) for k, v in obj.items()}
  if isinstance(obj, (list, tuple)):
    return [_jsonable(v) for v in obj]
  if isinstance(obj, (str, int, float, bool)) or obj is None:
    return obj
  return repr(obj)


def run_single(job):
  """One (agent, seed) run. Top-level so it pickles for multiprocessing."""
  spec, agent_name, seed = job
  agent_spec: AgentSpec = spec.agents[agent_name]

  env = spec.env.build(seed=seed)
  # stable per-(agent, seed) agent seed, independent of the env stream
  agent_index = list(spec.agents).index(agent_name)
  agent = agent_spec.build(env, seed=seed * 10_007 + agent_index)

  records = []
  t_run = time.perf_counter()
  for ep in range(spec.episodes):
    obs = env.reset()
    agent.begin_episode(ep)
    ep_reward, steps = 0., 0
    terminated = truncated = False
    t_ep = time.perf_counter()

    while not (terminated or truncated):
      a = agent.act(obs)
      obs_next, r, terminated, truncated = env.step(a)
      agent.observe(Transition(s=obs, a=a, r=r, s_next=obs_next,
                               terminated=terminated, truncated=truncated))
      obs = obs_next
      ep_reward += r
      steps += 1

    stats = agent.end_episode()
    records.append(dict(episode=ep, reward=ep_reward, steps=steps,
                        terminated=terminated,
                        wall_time=time.perf_counter() - t_ep, **stats))

  out = os.path.join(spec.out_dir, spec.name, agent_name)
  os.makedirs(out, exist_ok=True)
  pd.DataFrame.from_records(records).to_parquet(os.path.join(out, f'seed_{seed}.parquet'))
  with open(os.path.join(out, 'config.json'), 'w') as f:
    json.dump({'env': _jsonable(spec.env), 'agent': _jsonable(agent_spec),
               'episodes': spec.episodes}, f, indent=2)

  df = pd.DataFrame.from_records(records)
  tail = df.tail(max(1, len(df) // 10))
  return (agent_name, seed, time.perf_counter() - t_run,
          float(tail['reward'].mean()), float(tail.get('expansions', pd.Series([0.])).mean()))


def run_experiment(spec: ExperimentSpec, workers: int | None = None):
  jobs = [(spec, name, seed) for name in spec.agents for seed in range(spec.seeds)]
  workers = min(workers or os.cpu_count() - 2, len(jobs))
  print(f'{spec.name}: {len(jobs)} runs on {workers} workers '
        f'({len(spec.agents)} agents x {spec.seeds} seeds x {spec.episodes} episodes)')

  t0 = time.perf_counter()
  if workers <= 1:
    results = [run_single(j) for j in jobs]
  else:
    with Pool(processes=workers, maxtasksperchild=1) as pool:
      results = []
      for res in pool.imap_unordered(run_single, jobs):
        name, seed, dt, rew, exp = res
        print(f'  done {name:>28} seed {seed}: {dt:7.1f}s   '
              f'tail reward {rew:10.1f}   tail expansions {exp:8.1f}', flush=True)
        results.append(res)

  print(f'{spec.name}: finished in {time.perf_counter() - t0:.1f}s '
        f'-> {os.path.join(spec.out_dir, spec.name)}')
  return results
