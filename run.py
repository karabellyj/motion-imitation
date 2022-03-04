# add current working directory to the system path
import sys
from os import getcwd
import argparse
from pathlib import Path

from stable_baselines3.common.callbacks import CheckpointCallback
import torch.nn as nn

from config import Config
from schedules import LinearDecay, ExponentialSchedule

sys.path.append(getcwd())

import dm2gym
from stable_baselines3 import PPO


def train(cfg: Config) -> None:
    env = dm2gym.make(domain_name='humanoid_CMU', task_name='humanoid_tracking')

    save_path = Path().cwd() / cfg.save_path
    training_timesteps = int(cfg.n_samples_m * 1e6)

    lr_schedule = LinearDecay(cfg.lr_start, cfg.lr_final, cfg.lr_scale).value
    clip_schedule = ExponentialSchedule(cfg.clip_start, cfg.clip_end, cfg.clip_exp_slope).value

    policy_kwargs = {'log_std_init': cfg.init_logstd,
                     'net_arch': [{'vf': [512, 512], 'pi': [512, 512]}],
                     'activation_fn': nn.Tanh}
    if cfg.checkpoint:
        model = PPO.load(
            cfg.checkpoint,
            env, verbose=1,
            n_steps=cfg.n_steps // cfg.n_envs,
            batch_size=cfg.minibatch_size,
            learning_rate=lr_schedule,
            ent_coef=cfg.ent_coef,
            gamma=cfg.gamma,
            n_epochs=cfg.n_epochs,
            clip_range_vf=clip_schedule, clip_range=clip_schedule,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(save_path / 'tb_logs')
        )
    else:
        model = PPO(
            "MlpPolicy",
            env, verbose=1,
            n_steps=cfg.n_steps // cfg.n_envs,
            batch_size=cfg.minibatch_size,
            learning_rate=lr_schedule,
            ent_coef=cfg.ent_coef,
            gamma=cfg.gamma,
            n_epochs=cfg.n_epochs,
            clip_range_vf=clip_schedule, clip_range=clip_schedule,
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(save_path / 'tb_logs')
        )

    # Save a checkpoint
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_freq,
        save_path=str(save_path / 'models/'),
        name_prefix='rl_model'
    )

    model.learn(total_timesteps=training_timesteps, callback=checkpoint_callback)

    model.save(str(save_path / 'model_final'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=False)
    parser.add_argument("--batch_size", type=int, required=False)
    parser.add_argument("--minibatch_size", type=int, required=False)
    parser.add_argument("--checkpoint", type=str, required=False)

    args = parser.parse_args()
    config = Config(**{k: v for k, v in vars(args).items() if v is not None})
    train(cfg=config)
