import sys
import argparse
from os import getcwd

from dm2gym.wrapper import convert_observation

sys.path.append(getcwd())

from stable_baselines3 import PPO
from dm_control import viewer, suite


def main(args):
    env = suite.load(domain_name='humanoid_CMU', task_name='humanoid_tracking')
    model = PPO.load(args.checkpoint)

    # Define a uniform random policy.
    def trained_policy(time_step):
        obs = convert_observation(time_step.observation)
        action, _states = model.predict(obs)
        return action

    viewer.launch(env, policy=trained_policy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)

    args = parser.parse_args()
    main(args)
