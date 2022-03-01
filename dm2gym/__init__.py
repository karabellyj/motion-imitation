import hashlib

import gym
from gym.envs.registration import register

import dm2gym
from dm2gym import wrapper
from .tracking import *


def make(domain_name, task_name, task_kwargs=None, visualize_reward=False):
    # register environment
    prehash_id = domain_name + task_name + str(task_kwargs) + str(visualize_reward)
    h = hashlib.md5(prehash_id.encode())
    gym_id = h.hexdigest() + '-v0'

    # avoid re-registering
    if gym_id not in gym_id_list:
        register(
            id=gym_id,
            entry_point='dm2gym.wrapper:DmControlEnvWrapper',
            kwargs={'domain_name': domain_name, 'task_name': task_name, 'task_kwargs': task_kwargs,
                    'visualize_reward': visualize_reward}
        )
    # add to gym id list
    gym_id_list.append(gym_id)

    # make the Open AI env
    return gym.make(gym_id)


gym_id_list = []
