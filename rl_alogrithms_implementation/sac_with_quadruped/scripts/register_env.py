import gymnasium as gym
from quad_env import CustomQuadEnv

gym.register(id='CustomQuadEnv-v0',
             entry_point='quad_env:CustomQuadEnv',
             max_episode_steps=1000)
