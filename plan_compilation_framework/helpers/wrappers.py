from typing import SupportsFloat, Any

from gymnasium import Wrapper, Env
from gymnasium.core import ObsType, ActType


class ReacherRewardWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
  def __init__(self, env: Env[ObsType, ActType], ctrl_gain=0.1):
    """Constructor for the Reward wrapper."""
    Wrapper.__init__(self, env)
    self.ctrl_gain = ctrl_gain

  def step(
      self, action: ActType
  ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
    observation, reward, terminated, truncated, info = self.env.step(action)
    return observation, self.reward(reward, info), terminated, truncated, info

  def reward(self, reward: SupportsFloat, info: dict[str, Any]) -> SupportsFloat:
    """Returns a modified environment ``reward``.

    Args:
        reward: The :attr:`env` :meth:`step` reward

    Returns:
        The modified `reward`
        :param reward:
        :param info:
    """
    return info['reward_dist'] + self.ctrl_gain * info['reward_ctrl']


class PusherRewardWrapper(Wrapper[ObsType, ActType, ObsType, ActType]):
  def __init__(self, env: Env[ObsType, ActType], ctrl_gain=0.1):
    """Constructor for the Reward wrapper."""
    Wrapper.__init__(self, env)
    self.ctrl_gain = ctrl_gain

  def step(
      self, action: ActType
  ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
    observation, reward, terminated, truncated, info = self.env.step(action)
    return observation, self.reward(reward, info), terminated, truncated, info

  def reward(self, reward: SupportsFloat, info: dict[str, Any]) -> SupportsFloat:
    """Returns a modified environment ``reward``.

    Args:
        reward: The :attr:`env` :meth:`step` reward

    Returns:
        The modified `reward`
        :param reward:
        :param info:
    """
    reward_near = 2 * (float(reward) - info['reward_dist'] - 0.1 * info['reward_ctrl'])
    return info['reward_dist'] + self.ctrl_gain * info['reward_ctrl']
