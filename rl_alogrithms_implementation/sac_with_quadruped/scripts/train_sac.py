import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
import register_env


def make_env():
    return gym.make('CustomQuadEnv-v0', render_mode=None)


if __name__ == "__main__":

    # vectorized and normalized
    venv = SubprocVecEnv([make_env for _ in range(8)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=5.0)

    model = SAC(
        "MlpPolicy",
        venv,
        verbose=1,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        ent_coef="auto_0.01",
        learning_rate=0.001,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    model.learn(total_timesteps=int(1e6), log_interval=10)
    model.save("sac_custom_quad")
