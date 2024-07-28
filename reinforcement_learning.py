import gym
import numpy as np
import random

# Initialize the FrozenLake environment with render mode
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")

# Q-learning settings
q_table = np.zeros((env.observation_space.n, env.action_space.n))
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_decay = 0.99
min_epsilon = 0.1
episodes = 10000
max_steps = 100

def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Training the agent
for episode in range(episodes):
    state, _ = env.reset()
    done = False
    step = 0

    while not done and step < max_steps:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action])

        state = next_state
        step += 1

    epsilon = max(min_epsilon, epsilon * epsilon_decay)  # Decay epsilon

# Display the learned Q-values
print("Learned Q-values:")
print(q_table)

# Display the optimal policy
actions = ['←', '↓', '→', '↑']
optimal_policy = [actions[np.argmax(q_table[state])] for state in range(env.observation_space.n)]
optimal_policy = np.array(optimal_policy).reshape((4, 4))
print("\nOptimal Policy:")
print(optimal_policy)

# Test the trained agent
state, _ = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(q_table[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    env.render()  # Display the environment state
    if done:
        if reward == 1:
            print("Agent reached the goal!")
        else:
            print("Agent fell into a hole.")

# Close the environment when done
env.close()
