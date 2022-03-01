import sys
from collections import OrderedDict

import numpy as np
from dm_control import suite
from dm_env.specs import BoundedArray
from gym.core import Env as GymEnv
from gym.spaces import Box
from gym.utils import seeding


def spec2space(spec: BoundedArray) -> Box:
    _min = np.clip(spec.minimum, -sys.float_info.max, sys.float_info.max)
    _max = np.clip(spec.maximum, -sys.float_info.max, sys.float_info.max)

    if np.isscalar(_min) and np.isscalar(_max):
        # same min and max for every element
        return Box(_min, _max, shape=spec.shape)
    else:
        # different min and max for every element
        return Box(_min + np.zeros(spec.shape), _max + np.zeros(spec.shape))


def dict2space(obj: dict) -> Box:
    ndim = np.sum([int(np.prod(v.shape)) for v in obj.values()])
    return Box(-np.inf, np.inf, shape=(ndim,))


def convert_observation(obs: OrderedDict) -> np.ndarray:
    # time_step observation is OrderedDict
    obs = [val.ravel() if isinstance(val, np.ndarray) else [val] for val in obs.values()]
    return np.concatenate(obs)


class DmControlEnvWrapper(GymEnv):

    def __init__(self, domain_name: str, task_name: str, task_kwargs: dict = None, visualize_reward: bool = False):
        self.dm_env = suite.load(domain_name=domain_name, task_name=task_name, task_kwargs=task_kwargs,
                                 visualize_reward=visualize_reward)

        # need to convert dm_control spec into gym space
        self.action_space: Box = spec2space(self.dm_env.action_spec())
        self.observation_space: Box = dict2space(self.dm_env.observation_spec())

        self.np_random, _ = seeding.np_random(None)

    @property
    def observation(self) -> np.ndarray:
        return convert_observation(self.time_step.observation)

    def reset(self):
        self.time_step = self.dm_env.reset()
        return self.observation

    def step(self, action: np.ndarray):
        self.time_step = self.dm_env.step(action)
        return self.observation, self.time_step.reward, self.time_step.last(), {}
