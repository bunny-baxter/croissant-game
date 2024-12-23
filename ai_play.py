import gymnasium as gym

import game_env

env = gym.make("bunny-baxter/CroissantGame-v0")
observation, info = env.reset()

episode_over = False
while not episode_over:
    action = env.action_space.sample()
    print(f"action = {action}")
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"money = {observation[0]}, reward = {reward} ({observation[1]} - {info['error_count']}), turns left = {observation[3]}")

    episode_over = terminated or truncated

env.close()

