from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize


def vectorize_env(env, n_envs=4, norm_rew=True, seed=42):
    def make_env_fn(seed, rank):
        def make_env():
            env.seed(seed + rank * 100)
            return env

        return make_env

    if n_envs == 1:
        vec_env = DummyVecEnv([make_env_fn(seed, 0)])
    else:
        env_fns = [make_env_fn(seed, rank) for rank in range(n_envs)]
        vec_env = SubprocVecEnv(env_fns)

    vec_normed = VecNormalize(vec_env, norm_obs=True, norm_reward=norm_rew)
    return vec_normed
