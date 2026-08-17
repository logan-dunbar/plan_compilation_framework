"""A* over the planning model.

With heuristic='zero' this is Dijkstra, matching the 2022 experiments (whose
heuristic slot was a literal TODO). 'manhattan' uses the model's heuristic to
the true goal; note it is only admissible for the unaugmented goal.
"""
from __future__ import annotations

import heapq
from itertools import count

from plan_compilation_framework.planners.base import Plan, Planner, SearchResult


class AStar(Planner):
  def __init__(self, model, heuristic: str = 'zero', seed: int | None = None):
    super().__init__(model, seed)
    assert heuristic in ('zero', 'manhattan')
    self.heuristic = heuristic

  def _h(self, s):
    return self.model.heuristic(s) if self.heuristic == 'manhattan' else 0.

  def plan(self, start, extra_goal=None) -> SearchResult:
    model = self.model
    is_goal = model.is_goal
    tiebreak = count()

    open_heap = [(self._h(start), next(tiebreak), start)]
    g = {start: 0.}
    came_from = {start: None}
    closed = set()
    expansions = 0

    while open_heap:
      _, _, s = heapq.heappop(open_heap)
      if s in closed:
        continue

      if is_goal(s) or (extra_goal is not None and extra_goal(s)):
        return SearchResult(self._reconstruct(s, came_from, g), expansions)

      closed.add(s)
      expansions += 1

      # randomised action order preserves the 2022 tie-breaking behaviour
      for a in self.rng.permutation(model.n_actions):
        a = int(a)
        s_next = model.next_state(s, a)
        if s_next == s:  # wall bump self-loop is never on a shortest path
          continue
        new_g = g[s] + model.cost(s, a, s_next)
        if s_next not in g or new_g < g[s_next]:
          g[s_next] = new_g
          came_from[s_next] = (s, a)
          heapq.heappush(open_heap, (new_g + self._h(s_next), next(tiebreak), s_next))

    return SearchResult(None, expansions)

  @staticmethod
  def _reconstruct(goal_state, came_from, g):
    actions, path = {}, [goal_state]
    node = came_from[goal_state]
    while node is not None:
      s, a = node
      actions[s] = a
      path.append(s)
      node = came_from[s]
    path.reverse()
    return Plan(actions=actions, path=path, cost=g[goal_state])
