import gymnasium as gym

    
env = gym.make("LunarLander-v2", render_mode="human")

seed_value=0

# Define options if the environment supports them
options = {
    'start_position': 0,  # Example option, depends on the environment
}

observation, info = env.reset(seed=seed_value)

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()

env.close()