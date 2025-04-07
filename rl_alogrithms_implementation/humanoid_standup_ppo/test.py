import gymnasium as gym
import stable_baselines3

# Create the Humanoid environment
env = gym.make('Humanoid-v4')  # or HumanoidStandup-v4

model = stable_baselines3.PPO.load("humanoid_ppo_model")  # Load the model

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs = env.reset()
