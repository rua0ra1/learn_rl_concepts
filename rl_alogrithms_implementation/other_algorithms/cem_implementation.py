import gym
import numpy as np
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')
state, _ = env.reset()
env.render()

# Vector of means (mu) and standard deviations (sigma) for each parameter
mu = np.random.uniform(size=env.observation_space.shape)
sigma = np.random.uniform(low=0.001, size=env.observation_space.shape)


def noisy_evaluation(env, W, render=False):
    """Uses parameter vector W to choose policy for 1 episode,
    returns reward from that episode"""
    reward_sum = 0
    state, _ = env.reset()
    t = 0
    while True:
        t += 1
        action = int(np.dot(W, state) > 0)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        reward_sum += reward
        if render and t % 3 == 0:
            env.render()
        if done or t > 205:
            break
    return reward_sum


def init_params(mu, sigma, n):
    """Sample matrix of weights from Gaussian defined by mu and sigma"""
    l = mu.shape[0]
    w_matrix = np.zeros((n, l))
    for p in range(l):
        w_matrix[:, p] = np.random.normal(
            loc=mu[p], scale=sigma[p] + 1e-17, size=(n,))
    return w_matrix


def get_constant_noise(step):
    return np.max(5 - step / 10., 0)


running_reward = 0
n = 40
p = 8
n_iter = 20
render = False

i = 0
while i < n_iter:
    wvector_array = init_params(mu, sigma, n)
    reward_sums = np.zeros((n))
    for k in range(n):
        reward_sums[k] = noisy_evaluation(env, wvector_array[k, :], render)

    rankings = np.argsort(reward_sums)
    top_vectors = wvector_array[rankings, -p:, :]

    print(f"top vectors shape: {top_vectors.shape}")

    # Update mu and sigma
    for q in range(top_vectors.shape[1]):
        mu[q] = top_vectors[:, q].mean()
        sigma[q] = top_vectors[:, q].std() + get_constant_noise(i)

    running_reward = 0.99 * running_reward + 0.01 * reward_sums.mean()
    print("#############################################################################")
    print(
        f"iteration: {i}, mean reward: {reward_sums.mean():.2f}, running reward mean: {running_reward:.2f}")
    print(f"reward range: {reward_sums.min()} to {reward_sums.max()}")

    i += 1

env.close()
