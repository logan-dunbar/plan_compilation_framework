import gym
import plan_compilation_framework.environments
from plan_compilation_framework.agents.plan_compiler import PlanCompiler
from plan_compilation_framework.agents.qlearning import QLearning
from plan_compilation_framework.helpers import Transition
from plan_compilation_framework.planners.astar import AStar


def main():
  # env = gym.make('gridworld-default-v0', seed=0)
  # env = gym.make('gridworld-bombs-v0', seed=0)
  # env = gym.make('gridworld-bombs-lots-v0', seed=0)
  env = gym.make('gridworld-random-20x20-v0', seed=0)
  # env = gym.make('gridworld-random-20x20-stoch-v0', seed=0)
  # env = gym.make('gridworld-random-50x50-v0', seed=0)
  # env = gym.make('gridworld-random-50x50-stoch-v0', seed=0)

  # agent = QLearning(env.action_space.n)
  planner = AStar(env, constant_reward=True)
  agent = PlanCompiler(planner, env.action_space.n, -100., 0.)

  for e in range(10000):
    obs = env.reset()

    done = False
    agent.begin_episode()
    ep_rew = 0.

    # plan = planner.get_plan(obs)

    while not done:
      env.render()

      action = agent.get_action(obs)
      # action = plan.next_action()

      n_obs, rew, done, info = env.step(action)
      actually_done = done and not info.get('TimeLimit.truncated', False)

      t = Transition(s=obs, a=action, r_p=rew, s_p=n_obs, d=done, a_d=actually_done)
      agent.do_update(t)

      obs = n_obs
      ep_rew += rew

    print(f'e: {e: 5}, rew: {ep_rew}')
    agent.end_episode()
  debug = 0


if __name__ == '__main__':
  main()
