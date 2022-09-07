class Plan:
  def __init__(self, plan: dict):
    self._plan = plan
    self._actions_left = list(plan.values())

  def __getitem__(self, item):
    return self._plan[item]

  def next_action(self):
    return self._actions_left.pop(0)
