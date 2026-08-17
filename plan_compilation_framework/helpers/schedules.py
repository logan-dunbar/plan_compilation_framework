import abc
import math
from functools import reduce
from itertools import accumulate


class Schedule(abc.ABC):
  def __init__(self):
    self._value = None

  def __repr__(self):
    return str(self._value)

  def __add__(self, other):
    return self._value + other

  def __radd__(self, other):
    return self._value + other

  def __mul__(self, other):
    return self._value * other

  def __rmul__(self, other):
    return self._value * other

  def __lt__(self, other):
    return self._value < other

  def __le__(self, other):
    return self._value <= other

  def __gt__(self, other):
    return self._value > other

  def __ge__(self, other):
    return self._value >= other

  def __eq__(self, other):
    return self._value == other

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


class CosineAnnealingSchedule(Schedule):
  def __init__(self, lr_min, lr_max, freq=1000, periods=10, end_min=True):
    super().__init__()
    self._lr_min = lr_min
    self._lr_spread = lr_max - lr_min
    self._freq = freq
    self._max_steps = freq * periods + (0 if not end_min else 0.5 * freq)
    self._steps = 0
    self._value = lr_max

  def update(self):
    self._steps += 1
    step = min(self._steps, self._max_steps)
    self._value = self._lr_min + 0.5 * self._lr_spread * (1. + math.cos(step * math.pi / (0.5 * self._freq)))


class StepSchedule(Schedule):
  def __init__(self, first, second, freq=100, periods=10, end_second=True):
    super().__init__()
    self._first, self._second = first, second
    self._freq = freq
    self._max_steps = freq * periods + (0 if not end_second else freq)
    self._steps = 0
    self._value = first

  def update(self):
    self._steps += 1
    step = min(self._steps, self._max_steps)
    self._value = self._first if (step // self._freq) % 2 == 0 else self._second


class CompositeSchedule(Schedule):
  def __init__(self, schedules):
    super().__init__()

    self._schedules = schedules
    self._value = reduce(lambda a, b: a * b, schedules)

  def update(self):
    [s.update() for s in self._schedules]
    self._value = reduce(lambda a, b: a * b, self._schedules)


def operator_test():
  sched1 = LinearSchedule(0.1)
  sched2 = LinearSchedule(0.2)

  out1 = sched1 + 0.1
  out2 = 0.1 + sched1
  out3 = sched1 * 0.1
  out4 = 0.1 * sched1
  out5 = sched1 < 1.0
  out6 = 1.0 < sched1
  out7 = sched1 <= 1.0
  out8 = 1.0 <= sched1
  out9 = sched1 > 1.0
  out10 = 1.0 > sched1
  out11 = sched1 >= 1.0
  out12 = 1.0 >= sched1
  out13 = sched1 == 1.0
  out14 = 1.0 == sched1
  out15 = sched1 != 1.0
  out16 = 1.0 != sched1

  out17 = sched1 + sched2
  out18 = sched2 + sched1
  out19 = sched1 * sched2
  out20 = sched2 * sched1
  out21 = sched1 < sched2
  out22 = sched2 < sched1
  out23 = sched1 <= sched2
  out24 = sched2 <= sched1

  debug = 0


def sched_test():
  import matplotlib
  import matplotlib.pyplot as plt
  matplotlib.use('tkagg')

  lin = LinearSchedule(1., 0.1, 10000)
  # sched = CosineAnnealingSchedule(0.001, 0.1, 1000, 10, end_min=False)
  sched = StepSchedule(0.01, 0.1, 500, 20, end_second=False)

  comp = CompositeSchedule([lin, sched])

  # eps1 = CosineAnnealingSchedule(0.1, 1.0, 100, 15, end_min=True)
  # eps2 = LinearSchedule(1., 0.1, 1500)
  # eps = CompositeSchedule([eps1, eps2])
  eps1 = CosineAnnealingSchedule(0.1, 0.8, 100, 100, end_min=True)
  eps2 = LinearSchedule(1., 0.5, 10000)
  eps = CompositeSchedule([eps1, eps2])

  lr = []
  for i in range(15000):
    # lr.append(lin * sched)
    # sched.update()
    # lin.update()

    # lr.append(comp.value)
    # comp.update()d

    lr.append(eps.value)
    eps.update()

  plt.plot(lr)
  plt.show()
  debug = 0


if __name__ == '__main__':
  # operator_test()
  sched_test()
