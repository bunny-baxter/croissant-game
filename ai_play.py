import argparse

import gymnasium as gym
import torch
from torch import nn

import game_env

parser = argparse.ArgumentParser(prog = "Pre-trained AI plays the croissant game")
parser.add_argument("--model", required = True, help = "model file to load")
parser.add_argument("--deterministic", action = "store_true", help = "Use argmax instead of sampling when deciding on action")
parser.add_argument("--enable_stash", action = "store_true", help = "Enable the stash, where money can carry over between games.")
parser.add_argument("--print_probs", action = "store_true", help = "Print out model probabilities at each step.")
parser.add_argument("--iterations", type = int, default = 1, help = "Number of games to play.")
args = parser.parse_args()

if args.enable_stash:
    env = gym.make("bunny-baxter/CroissantGameExploitable-v0")
else:
    env = gym.make("bunny-baxter/CroissantGame-v0")

# De-pickled on torch.load.
class ConvertToFloat(nn.Module):
    def forward(self, x):
        return torch.from_numpy(x).float()

policy_net = torch.load(args.model, weights_only = False)

def get_observation_string(observation):
    s = f"money {observation[0]}, croissants {observation[1]}, turns left {observation[3]}"
    if args.enable_stash:
        s += f", stash {observation[5]}"
    return s

scores = []

for i in range(args.iterations):
    if args.iterations > 1:
        print(f"Game {i + 1}/{args.iterations}")
        print("----------")

    observation, _info = env.reset()
    print(f"begin Croissant Game! -> {get_observation_string(observation)}")

    with torch.no_grad():
        while True:
            logits = policy_net(observation)
            probabilities = torch.softmax(logits, dim = -1)
            if args.print_probs:
                print(f"probabilities: {probabilities.tolist()}")
            if args.deterministic:
                action = logits.argmax(dim = -1).item()
            else:
                action = torch.distributions.Categorical(probabilities).sample().item()

            observation, _reward, terminated, truncated, _info = env.step(action)
            print(f"action {action} -> {get_observation_string(observation)}")

            if terminated or truncated:
                print(f"The AI ate {observation[1]} croissants.")
                scores.append(observation[1])
                break

    if args.iterations > 1:
        print()

if args.iterations > 1:
    print(f"Best score: {max(scores)}")
    average_score = sum(scores) / float(len(scores))
    print(f"Average score: {average_score}")

env.close()
