from gym.envs.registration import register, registry
from plan_compilation_framework.environments.gridworld import SimpleGridWorld

if 'gridworld-default-v0' not in registry.env_specs:
  register(
    id='gridworld-default-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=150,
    kwargs={'file_name': 'maps/gridworld/default.txt'}
  )
  register(
    id='gridworld-bombs-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=200,
    kwargs={'file_name': 'maps/gridworld/bombs.txt'}
  )
  register(
    id='gridworld-bombs-lots-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=3000,
    kwargs={'file_name': 'maps/gridworld/bombs-lots.txt'}
  )
  register(
    id='gridworld-random-20x20-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=1000,
    kwargs={'random_graph': True,
            'random_start': True,
            'size'        : (20, 20)}
  )
  register(
    id='gridworld-random-20x20-stoch-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=1000,
    kwargs={'random_graph': True,
            'random_start': True,
            'stochastic'  : True,
            'size'        : (20, 20)}
  )
  register(
    id='gridworld-random-50x50-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=10000,
    kwargs={'random_graph': True,
            'random_start': True,
            'size'        : (50, 50)}
  )
  register(
    id='gridworld-random-50x50-stoch-v0',
    entry_point='plan_compilation_framework.environments.gridworld:SimpleGridWorld',
    max_episode_steps=10000,
    kwargs={'random_graph': True,
            'random_start': True,
            'stochastic'  : True,
            'size'        : (50, 50)}
  )
