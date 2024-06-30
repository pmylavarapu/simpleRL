# This is a simple Reinforcement Learning project
def mainfunc(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    import numpy as np

    # Define the environment
    n_states = 90  # Number of states in the grid world
    n_actions = 3  # Number of possible actions (up, down, left, right)
    goal_state = 8  # Goal state

    # Initialize Q-table with zeros
    Q_table = np.zeros((n_states, n_actions))

    # Define parameters
    learning_rate = 0.8
    discount_factor = 0.9
    exploration_prob = 0.8
    epochs = 1000

    # Q-learning algorithm
    for epoch in range(epochs):
        current_state = np.random.randint(0, n_states)  # Start from a random state

        while current_state != goal_state:
            # Choose action with epsilon-greedy strategy
            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, n_actions)  # Explore
            else:
                action = np.argmax(Q_table[current_state])  # Exploit

            # Simulate the environment (move to the next state)
            # For simplicity, move to the next state
            next_state = (current_state + 1) % n_states

            # Define a simple reward function (1 if the goal state is reached, 0 otherwise)
            reward = 100 if next_state == goal_state else 0

            # Update Q-value using the Q-learning update rule
            Q_table[current_state, action] += learning_rate * \
                                              (reward + discount_factor *
                                               np.max(Q_table[next_state]) - Q_table[current_state, action])

            current_state = next_state  # Move to the next state

    # After training, the Q-table represents the learned Q-values
    print("Learned Q-table:")
    print(Q_table)
