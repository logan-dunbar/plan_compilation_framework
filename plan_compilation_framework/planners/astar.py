from queue import PriorityQueue

from plan_compilation_framework.planners.plan import Plan
from plan_compilation_framework.planners.planner import Planner


class AStar(Planner):
  def get_plan(self, start, is_goal=None):
    is_goal = is_goal if is_goal else lambda x: False

    frontier = PriorityQueue()
    came_from = {start: None}
    cost_so_far = {start: 0.}

    # (priority, state, action, done)
    frontier.put((0, start, None, False))

    _ = self.model.reset()

    while not frontier.empty():
      _, s, a, done = frontier.get()

      if done or is_goal(s):
        plan = {}
        state_and_action = came_from[s]
        while state_and_action is not None:
          state, action = state_and_action
          plan[state] = action
          state_and_action = came_from[state]
        plan = dict(reversed(plan.items()))
        return Plan(plan)

      for action in self.np_random.permutation(range(self.model.action_space.n)):
        self.model.set_state(s)

        s_p, r, done, info = self.model.step(action)
        r = 1 if self.constant_reward else (-r)
        new_cost = cost_so_far[s] + r

        if s_p not in cost_so_far or new_cost < cost_so_far[s_p]:
          cost_so_far[s_p] = new_cost
          priority = new_cost + 0.  # TODO: heuristic
          frontier.put((priority, s_p, action, done))
          came_from[s_p] = (s, action)

    raise Exception('No plan found')
