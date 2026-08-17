"""The Plan Compilation Framework agent (thesis Algorithm 1), rebuilt.

One class, three modes (PLANNER / LEARNER / EXPLORER), every knob a config
field, every ablation a switch:

  A1  q0_learner / q0_explorer     initialisation study
  A2  mc_until_learnt              False -> TD bootstrapping everywhere
  A3  reduce_horizon               False -> planner always plans to the true goal
  A4  latch (via the label rule)   False -> learnt states may revert
  A5  use_explorer / bound_quota   exploration mechanism variants
  A6  value_ordered_goals          False -> any learnt state is a goal
  R1  use_plan_cache               cross-episode plan cache on/off

Expansion accounting is honest: `expansions` counts real search work done
this episode; `cache_hits` counts actions served from the cross-episode plan
cache; following the plan you just computed costs nothing and counts nothing.

Plan following. The planner holds a single current plan (a state->action
corridor to one goal state) and follows it by state lookup, so stochastic
drift back onto the corridor is served for free while drift off it triggers a
replan. The plan is discarded when its goal state stops being a goal — which
only happens with the latch off, when a learnt surrogate goal reverts — and
when the agent has been served from it far more times than its length
(FOLLOW_CAP), which bounds any pathological cycle. The 2022 code accumulated
every plan of the episode into one memo; with the latch off that memo can
hold two unlearnt states pointing at each other and, with no step limit,
loop forever (this is what took the workstation down on 17 Aug 2026).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from plan_compilation_framework.core import (
  Agent, Constant, Schedule, State, Transition, epsilon_greedy_probabilities,
)
from plan_compilation_framework.labelling import LabelRule, make_label_rule
from plan_compilation_framework.planners.base import Planner

PLANNER, LEARNER, EXPLORER = 'planner', 'learner', 'explorer'

# a plan may serve at most this many actions per state on its path before it
# is considered stale and replaced (tolerates stochastic bouncing, bounds cycles)
FOLLOW_CAP_PER_STATE = 8
FOLLOW_CAP_MIN = 64


@dataclass
class PCFConfig:
  gamma: float = 1.0
  # Learner Q learning rate. With alpha_per_visit=True the schedule is indexed
  # by the update count of the (state, action) pair rather than the episode,
  # which is what the 2022 code did (a per-(s,a) copy of the schedule): the
  # first update to a pessimistically initialised pair is a full replace, then
  # decays. Harmonic(1.0, 0.1) reproduces the paper.
  alpha: Schedule = Constant(0.1)
  alpha_per_visit: bool = True
  # Optional separate rate for bootstrap updates between learnt states (a
  # sensitivity knob, C4). None = same as alpha, which is the paper setting.
  alpha_learnt: Schedule | None = None
  alpha_explorer: Schedule = Constant(0.1)
  eps_learner: Schedule = Constant(0.1)      # epsilon-greedy over Q
  eps_explorer: Schedule = Constant(0.1)     # epsilon-greedy over Q_exp
  explore_prob: Schedule = Constant(0.03)    # epsilon_exp: chance to start exploring
  quota_frac: Schedule = Constant(0.1)       # xi: exploration budget fraction
  q0_learner: float = -1250.                 # pessimistic init (A1)
  q0_explorer: float = 0.                    # optimistic init (A1)
  label_rule: tuple = ('jsd',)               # L1-L5 (see labelling.py)
  mc_until_learnt: bool = True               # A2
  reduce_horizon: bool = True                # A3
  use_explorer: bool = True                  # A5
  bound_quota: bool = True                   # A5(ii)
  value_ordered_goals: bool = True           # A6
  use_plan_cache: bool = False               # R1 (paper's PDDL runs used True)


class PlanCompiler(Agent):
  def __init__(self, env, planner: Planner, config: PCFConfig | None = None,
               seed: int | None = None):
    self.env = env
    self.planner = planner
    self.config = config or PCFConfig()
    self.rng = np.random.default_rng(seed)

    self.n_actions = env.n_actions
    self.q: dict = {}      # state -> np.ndarray(n_actions), pessimistic
    self.q_exp: dict = {}  # state -> np.ndarray(n_actions), optimistic
    self.n_sa: dict = {}   # (state, action) -> number of Q updates so far
    self.label: LabelRule = make_label_rule(self.config.label_rule)

    self.mode = PLANNER
    self.quota = 0.
    self.traj: list[Transition] = []
    self.plan: dict = {}                          # current plan: state -> action
    self.plan_goal = None                         # the state the current plan ends at
    self.plan_follows = 0                         # actions served from the current plan
    self.cache_follows = 0                        # consecutive actions served from cache
    self.plan_cache: dict = {}                    # state -> {action: votes}, cross-episode
    self.episode = 0
    self._stats = {}

  # -- value tables ----------------------------------------------------------
  def _q(self, s):
    if s not in self.q:
      self.q[s] = np.full(self.n_actions, self.config.q0_learner)
    return self.q[s]

  def _q_e(self, s):
    if s not in self.q_exp:
      self.q_exp[s] = np.full(self.n_actions, self.config.q0_explorer)
    return self.q_exp[s]

  def _v(self, s):
    return float(self._q(s).max())

  def _alpha(self, s, a, learnt_phase: bool) -> float:
    """Learning rate for the next Q update to (s, a); advances the visit count."""
    cfg = self.config
    sched = cfg.alpha_learnt if (learnt_phase and cfg.alpha_learnt is not None) else cfg.alpha
    if cfg.alpha_per_visit:
      n = self.n_sa.get((s, a), 0)
      self.n_sa[(s, a)] = n + 1
      return sched(n)
    return sched(self.episode)

  def _greedy_eps(self, q_values, eps):
    probs = epsilon_greedy_probabilities(q_values, eps)
    return int(self.rng.choice(self.n_actions, p=probs))

  # -- lifecycle -------------------------------------------------------------
  def begin_episode(self, episode: int):
    self.episode = episode
    self.mode = PLANNER
    self.quota = 0.
    self.traj.clear()
    self._drop_plan()
    self.cache_follows = 0
    self.label.begin_episode(episode)
    self.planner.begin_episode()
    self._stats = dict(expansions=0, cache_hits=0, plans_computed=0,
                       planner_steps=0, learner_steps=0, explorer_steps=0,
                       plan_time=0.)

  # -- action selection (thesis Algorithm 1, act) ----------------------------
  def act(self, s: State) -> int:
    cfg = self.config

    if self.quota > 0.:
      self.mode = EXPLORER
      self._stats['explorer_steps'] += 1
      return self._greedy_eps(self._q_e(s), cfg.eps_explorer(self.episode))

    if not self.label.is_learnt(s):
      self.mode = PLANNER
      self._stats['planner_steps'] += 1
      return self._planner_action(s)

    if cfg.use_explorer and self.rng.random() < cfg.explore_prob(self.episode):
      self.quota = (cfg.quota_frac(self.episode) * abs(self._v(s))
                    if cfg.bound_quota else np.inf)
      self.mode = EXPLORER
      self._stats['explorer_steps'] += 1
      return self._greedy_eps(self._q_e(s), cfg.eps_explorer(self.episode))

    self.mode = LEARNER
    self._stats['learner_steps'] += 1
    return self._greedy_eps(self._q(s), cfg.eps_learner(self.episode))

  def _drop_plan(self):
    self.plan = {}
    self.plan_goal = None
    self.plan_follows = 0

  def _plan_is_valid(self):
    """The current plan is usable only while its goal is still a goal."""
    if self.plan_goal is None:
      return False
    if self.plan_follows >= max(FOLLOW_CAP_MIN, FOLLOW_CAP_PER_STATE * len(self.plan)):
      return False
    return (self.planner.model.is_goal(self.plan_goal)
            or (self.config.reduce_horizon and self.label.is_learnt(self.plan_goal)))

  def _planner_action(self, s) -> int:
    if s in self.plan and self._plan_is_valid():
      self.plan_follows += 1
      return self.plan[s]
    self._drop_plan()

    cache_cap = max(FOLLOW_CAP_MIN, FOLLOW_CAP_PER_STATE * len(self.plan_cache) ** 0.5)
    if (self.config.use_plan_cache and s in self.plan_cache
        and self.cache_follows < cache_cap):
      votes = self.plan_cache[s]
      self._stats['cache_hits'] += 1
      self.cache_follows += 1
      return max(votes, key=votes.get)
    self.cache_follows = 0

    t0 = time.perf_counter()
    result = self.planner.plan(s, extra_goal=self._extra_goal(s))
    self._stats['plan_time'] += time.perf_counter() - t0
    self._stats['expansions'] += result.expansions
    self._stats['plans_computed'] += 1

    if not result.found:
      return int(self.rng.integers(self.n_actions))

    self.plan = dict(result.plan.actions)
    self.plan_goal = result.plan.path[-1]
    self.plan_follows = 1
    if self.config.use_plan_cache:
      for state, action in result.plan.actions.items():
        votes = self.plan_cache.setdefault(state, {})
        votes[action] = votes.get(action, 0) + 1
    return self.plan[s]

  def _extra_goal(self, plan_start):
    """Goal augmentation: learnt states (of higher value than the start, A6)."""
    if not self.config.reduce_horizon:
      return None
    if not self.config.value_ordered_goals:
      return self.label.is_learnt
    start_value = self._v(plan_start)

    def is_goal(s):
      return self.label.is_learnt(s) and self._v(s) > start_value

    return is_goal

  # -- learning (thesis Algorithm 1, learn) ----------------------------------
  def observe(self, t: Transition):
    cfg = self.config
    eps = cfg.eps_learner(self.episode)
    learnt_s, learnt_next = self.label.is_learnt(t.s), self.label.is_learnt(t.s_next)

    if not cfg.mc_until_learnt:
      # A2: TD bootstrapping everywhere
      self._td_update(t, self._alpha(t.s, t.a, learnt_phase=False), eps, record=True)
    elif learnt_s and learnt_next:
      self._td_update(t, self._alpha(t.s, t.a, learnt_phase=True), eps, record=True)
    else:
      self.traj.append(t)

    if cfg.mc_until_learnt:
      if t.terminated:
        self._monte_carlo(0., eps)
      elif learnt_next:
        self._monte_carlo(self._v(t.s_next), eps)

    # explorer value function: always bootstrapped online
    q_e = self._q_e(t.s)
    target = t.r + (0. if t.terminated else cfg.gamma * float(self._q_e(t.s_next).max()))
    q_e[t.a] += cfg.alpha_explorer(self.episode) * (target - q_e[t.a])

    if self.quota > 0.:
      self.quota -= abs(t.r)

  def _td_update(self, t: Transition, alpha, eps, record: bool):
    q = self._q(t.s)
    q_before = q.copy()
    target = t.r + (0. if t.terminated else self.config.gamma * self._v(t.s_next))
    delta = alpha * (target - q[t.a])
    q[t.a] += delta
    if record:
      self.label.record_update(t.s, q_before, q, delta, eps)

  def _monte_carlo(self, g_init: float, eps):
    """Backward pass over the pending trajectory with the true return."""
    g = g_init
    for t in reversed(self.traj):
      g = t.r + self.config.gamma * g
      q = self._q(t.s)
      q_before = q.copy()
      delta = self._alpha(t.s, t.a, learnt_phase=False) * (g - q[t.a])
      q[t.a] += delta
      self.label.record_update(t.s, q_before, q, delta, eps)
    self.traj.clear()

  def end_episode(self) -> dict:
    return {**self._stats, 'n_learnt': self.label.count()}
