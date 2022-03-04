from dataclasses import dataclass


@dataclass
class Config:
    save_path: str = './'
    n_steps: int = 2048 * 4
    minibatch_size: int = 512 * 4
    n_envs: int = 1
    n_samples_m: int = 4
    save_freq: int = 50_000

    lr_start: float = 500 * 1e-6
    lr_final: float = 1 * 1e-6
    lr_scale: float = 1

    clip_start: float = 0.55
    clip_end: float = 0.1
    clip_exp_slope: float = 5

    init_logstd: float = -0.75

    ent_coef: float = -0.0075
    gamma: float = 0.99
    n_epochs: int = 4

    checkpoint: str = ''
