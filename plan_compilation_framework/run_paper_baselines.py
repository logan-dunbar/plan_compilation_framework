"""Add the value-in-search / model-based / shaping baselines to results/paper."""
from plan_compilation_framework.experiment import EnvSpec, ExperimentSpec, run_experiment
from plan_compilation_framework.run_gridworld import baseline_agents


def main():
  agents = {k: v for k, v in baseline_agents(10_000).items()
            if k in ('lh_model', 'lh_env', 'dynaq', 'shaped_q')}
  spec = ExperimentSpec(name='paper', env=EnvSpec(size=(50, 50), max_steps=1_000_000),
                        agents=agents, episodes=10_000, seeds=10)
  run_experiment(spec, workers=40)


if __name__ == '__main__':
  main()
