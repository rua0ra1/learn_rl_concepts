import numpy as np
from collections import defaultdict

# Define the grid-world environment
class GridWorld:
    def __init__(self):
        self.grid = [
            ["S1", "S2", "S3"],
            ["S4", "S5", "S6"],
            ["S7", "S8", "G"]
        ]
        self.rewards = {
            "G": 1.0  # Reward at the goal
        }
        self.rows = len(self.grid)
        self.cols = len(self.grid[0])
        self.terminal_state = "G"

    def step(self, state, action):
        """Takes a step in the environment."""
        row, col = self.state_to_coords(state)
        if state == self.terminal_state:
            return state, 0, True  # Stay in terminal state

        # Define movement
        if action == "up":
            row = max(0, row - 1)
        elif action == "down":
            row = min(self.rows - 1, row + 1)
        elif action == "left":
            col = max(0, col - 1)
        elif action == "right":
            col = min(self.cols - 1, col + 1)

        new_state = self.grid[row][col]
        reward = self.rewards.get(new_state, 0)
        is_terminal = new_state == self.terminal_state
        return new_state, reward, is_terminal

    def state_to_coords(self, state):
        """Convert state to grid coordinates."""
        for r, row in enumerate(self.grid):
            for c, cell in enumerate(row):
                if cell == state:
                    return r, c
        raise ValueError(f"State {state} not found.")

# Define a fixed policy
def fixed_policy(state):
    """
    A fixed policy mapping states to actions.
    For simplicity, assume the policy is to always move 'right' if possible,
    otherwise 'down'.
    """
    if state in ["S1", "S2", "S4", "S5", "S7", "S8"]:
        return "right"  # Always move right unless at the edge
    elif state in ["S3", "S6"]:
        return "down"  # Move down when at the right edge
    else:
        return None  # Terminal state (no action)

# Generate an episode
def generate_episode(env, policy):
    """Simulates an episode following the given policy."""
    state = "S1"  # Starting state
    episode = []
    while True:
        action = policy(state)
        if action is None:  # Terminal state, no action
            break
        next_state, reward, done = env.step(state, action)
        episode.append((state, reward))
        if done:
            break
        state = next_state
    return episode

# First-Visit Monte Carlo Prediction
def first_visit_mc_prediction(env, policy, num_episodes=1000):
    V = defaultdict(float)  # State-value function
    returns = defaultdict(list)  # Track returns for each state

    for _ in range(num_episodes):
        # Generate an episode
        episode = generate_episode(env, policy)

        # Track first visits
        first_visit = {}
        for t, (state, reward) in enumerate(episode):
            if state not in first_visit:
                first_visit[state] = t

        # Compute returns and update value function
        G = 0
        for t in reversed(range(len(episode))):
            state, reward = episode[t]
            G = reward + G  # Compute cumulative reward
            if t == first_visit[state]:
                returns[state].append(G)  # Add return only for first visit
                V[state] = np.mean(returns[state])  # Update value estimate

    return V

# Solve the grid-world problem
env = GridWorld()
num_episodes = 1000
value_function = first_visit_mc_prediction(env, fixed_policy, num_episodes)

# Display the estimated value function
print("Estimated State-Value Function:")
for state, value in sorted(value_function.items()):
    print(f"State {state}: {value:.2f}")
