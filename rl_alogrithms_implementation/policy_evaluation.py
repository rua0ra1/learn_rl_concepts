import numpy as np


#setup the grid world parameters
grid_size=5 # grid size 5 by 5
gamma=0.9 # discount factor
theta=1e-4 #converngence threshold

#define the rewards
rewards=np.full((grid_size,grid_size),-1)
rewards[4,4]= 0 #terminal state
rewards[2,2]=-10
rewards[2,3]=-10


#initialze value function with zeros
value_function= np.zeros((grid_size,grid_size))

#define 
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Move up, down, left, right

#chech the terminal state
def is_terminal(state):
    """Check if the state is terminal."""
    return state == (4, 4) 

def get_next_state(state,action):
    x,y=state
    dx,dy=action
    next_x, next_y = max(0, min(x + dx, grid_size - 1)), max(0, min(y + dy, grid_size - 1))
    return next_x, next_y


def policy_evaluation():
    """Evaluate the policy using the Bellman equation."""
    while True:
        delta = 0
        for i in range(grid_size):
            for j in range(grid_size):
                state = (i, j)
                if is_terminal(state):
                    continue  # Skip terminal states

                v = value_function[i, j]
                new_value = 0

                # For each action, calculate expected value
                for action in actions:
                    next_state = get_next_state(state, action)
                    reward = rewards[next_state]
                    new_value += 0.25 * (reward + gamma * value_function[next_state])

                # Update the value function
                value_function[i, j] = new_value
                delta = max(delta, abs(v - new_value))

        # Check for convergence
        if delta < theta:
            break

# Run policy evaluation
policy_evaluation()

# Display the final value function
print("Final value function:")
print(value_function)