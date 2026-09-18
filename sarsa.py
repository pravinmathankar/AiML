# import gymnasium as gym
# import numpy as np
# import random

# # Create environment
# env = gym.make("CliffWalking-v1")

# # print(env)

# # print(env.observation_space.n)

# # print(env.action_space.n)

# # starting_state = env.reset()

# # print(starting_state)




# gamma = 0.99
# alpha = 0.5
# epsilon = 0.1
# episodes = 500
# Q = np.zeros((48, 4))

# # Policy - epsilon-greedy : state -> action

# def epsilon_greedy(state):
#     if random.random() < epsilon:
#         return env.action_space.sample() # random action => EXPLORE
#     else:
#         return np.argmax(Q[state])  # exploit



#     for episode in range(episodes):
#      render = (episode % 50 == 0)

#     if render:
#         env = gym.make("CliffWalking-v1", render_mode="human")
#         print(f"rendering env for episode={episode+1}/500")
#     else:
#         env = gym.make("CliffWalking-v1")

#     done = False # when the episode has ended
#     state, _ = env.reset()
#     action = epsilon_greedy(state)

#     total_reward = 0
#     episode_len = 0

#     while not done:
#         next_state, reward, terminated, truncated, _ = env.step(action)
#         done = terminated or truncated
#         next_action = epsilon_greedy(next_state)

#         # SARSA update
#         Q[state, action] += alpha * (reward + gamma*(Q[next_state, next_action] - Q[state, action]))

#         state = next_state
#         action = next_action

#         total_reward += reward
#         episode_len += 1

#     print(f"episode={episode+1}/500: total reward = {total_reward} & ep length = {episode_len}")
#     env.close()






import gymnasium as gym
import numpy as np
import random

# Create environment
env = gym.make("CliffWalking-v1")

# Hyperparameters
gamma = 0.99
alpha = 0.5
epsilon = 0.1
episodes = 500

# Q-table
# 48 states × 4 actions
Q = np.zeros((env.observation_space.n, env.action_space.n))


# Epsilon-greedy policy
def epsilon_greedy(state):
    if random.random() < epsilon:
        return env.action_space.sample()   # EXPLORE
    else:
        return np.argmax(Q[state])          # EXPLOIT


# Training
for episode in range(episodes):

    state, _ = env.reset()

    # Choose first action
    action = epsilon_greedy(state)

    done = False
    total_reward = 0
    episode_len = 0

    while not done:

        # Take action
        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        # SARSA update
        if done:

            Q[state, action] += alpha * (
                reward - Q[state, action]
            )

        else:

            # Choose next action using same policy
            next_action = epsilon_greedy(next_state)

            Q[state, action] += alpha * (
                reward
                + gamma * Q[next_state, next_action]
                - Q[state, action]
            )

            # Move to next state/action
            state = next_state
            action = next_action

        total_reward += reward
        episode_len += 1

    print(
        f"Episode = {episode + 1}/{episodes} "
        f"| Reward = {total_reward} "
        f"| Length = {episode_len}"
    )

env.close()

# what did our agent learn?

env = gym.make("CliffWalking-v1", render_mode="human")
state, _ = env.reset()
done = False
total_reward = 0
episode_len = 0

while not done:
    action = np.argmax(Q[state])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    
    total_reward += reward
    episode_len += 1

print(f"total reward = {total_reward} & episode len = {episode_len}")
env.close()