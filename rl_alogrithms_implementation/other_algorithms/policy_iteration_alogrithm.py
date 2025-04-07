import numpy as np
from enum import Enum
from tabulate import tabulate


#setup the grid world parameters
grid_size=10 # grid size 5 by 5
gamma=0.9 # discount factor
theta=1e-4 #converngence threshold

#define the rewards
rewards=np.full((grid_size,grid_size),-1)
rewards[grid_size-1,grid_size-1]= 10 #terminal state

#penality states
rewards[3,3]=-10
rewards[6,6]=-10
rewards[7,8]=-10



#define 
actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Move up, down, left, right
class Action(Enum):
    UP=0
    DOWN=1
    LEFT=2
    RIGHT=3

#check the terminal state
def is_terminal(state):
    return state==(4,4)

# state transition function 
def get_next_state(state, action):
    """Get the next state given an action."""
    x, y = state
    if action == Action.UP and x > 0:
        return (x - 1, y)
    elif action == Action.DOWN and x < grid_size - 1:
        return (x + 1, y)
    elif action == Action.LEFT and y > 0:
        return (x, y - 1)
    elif action == Action.RIGHT and y < grid_size - 1:
        return (x, y + 1)
    return state  # If moving out of bounds, stay in the same state



def policy_evaluation(policy,value_function):
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

                #get the action from current policy for current state
                action_from_policy= policy[i,j]
    
                next_state = get_next_state(state, action_from_policy)
                reward = rewards[next_state]
                new_value += 0.25 * (reward + gamma * value_function[next_state])

                # Update the value function
                value_function[i, j] = new_value
                delta = max(delta, abs(v - new_value))

        # Check for convergence
        if delta < theta:
            break
    return value_function


def policy_improvement(policy, value_function):
    """Improve the policy based on the current value function."""
    policy_stable = True
    for x in range(grid_size):
        for y in range(grid_size):
            state = (x, y)
            if is_terminal(state):
                continue
            # Evaluate each action and choose the one with the highest value
            action_values = []
            for action in Action:
                next_state = get_next_state(state, action)
                if next_state is None:
                    print(f"Error: next_state is None for state {state} and action {action}")
                reward = rewards[next_state]
                action_values.append(reward + gamma * value_function[next_state])
            best_action = list(Action)[np.argmax(action_values)]
            # If the policy has changed, mark as unstable
            if best_action != policy[x, y]:
                policy_stable = False
            policy[x, y] = best_action
    return policy, policy_stable


# Main Policy Iteration loop

def policy_iteration():
    #initialze value function with zeros
    value_function= np.zeros((grid_size,grid_size))
    # Random policy initialization
    policy = np.random.choice(list(Action), size=(grid_size, grid_size))

    iteration = 0
    while True:
        iteration += 1
        print(f"Iteration: {iteration}")
        
        # Policy Evaluation
        value_function = policy_evaluation(policy, value_function)
        
        # Policy Improvement
        policy, policy_stable = policy_improvement(policy, value_function)
        
        # Check if policy is stable
        if policy_stable:
            print("Policy is stable and optimal.")
            break
    print("optimal value funciton ")
    print(tabulate(value_function, tablefmt="grid"))

    print("optimal policy")
    print(tabulate(policy, tablefmt="grid"))


    



if __name__=="__main__":
    policy_iteration()