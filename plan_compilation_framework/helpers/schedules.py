import abc


class Schedule(abc.ABC):
  def __init__(self):
    self._value = None

  @property
  def value(self):
    return self._value

  @abc.abstractmethod
  def update(self):
    raise NotImplementedError()


class ConstantSchedule(Schedule):
  def __init__(self, value):
    super().__init__()
    self._value = value

  def update(self):
    pass


class LinearSchedule(Schedule):
  def __init__(self, start, stop=0., steps=1000):
    super().__init__()
    assert steps > 0 and start - stop != 0.
    self._value = self._start = start
    self._stop = stop
    self._steps = steps
    self._delta = (stop - start) / steps

  def update(self):
    if self._delta != 0:
      self._value += self._delta
      if not (self._start > self._value > self._stop or self._start < self._value < self._stop):
        self._value = self._stop
        self._delta = 0
