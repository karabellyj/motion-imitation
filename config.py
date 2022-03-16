from dataclasses import dataclass


@dataclass
class Config:
    save_path: str = './'
    n_steps: int = 4096 * 12
    minibatch_size: int = 256
    n_envs: int = 12
    n_samples_m: int = 100
    save_freq: int = 120_000

    lr_start: float = 2e-6
    lr_final: float = 1 * 1e-6
    lr_scale: float = 1

    clip_start: float = 0.2
    clip_end: float = 0.1
    clip_exp_slope: float = 5

    init_logstd: float = -0.75

    ent_coef: float = 0
    gamma: float = 0.95
    gae_lambda: float = 0.95
    n_epochs: int = 4

    checkpoint: str = ''
