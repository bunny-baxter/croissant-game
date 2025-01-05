import argparse
import gymnasium as gym
import torch
from torch import nn

import game_env

parser = argparse.ArgumentParser(prog = "Pre-trained AI plays the croissant game")
parser.add_argument("--model", required = True, help = "model file to load")
parser.add_argument("--deterministic", action = "store_true", help = "Use argmax instead of sampling when deciding on action")
args = parser.parse_args()

env = gym.make("bunny-baxter/CroissantGame-v0")

# De-pickled on torch.load.
class ConvertToFloat(nn.Module):
    def forward(self, x):
        return torch.from_numpy(x).float()

policy_net = torch.load(args.model, weights_only = False)

observation, _info = env.reset()
print(f"begin Croissant Game! -> money {observation[0]}, croissants {observation[1]}, turns left {observation[3]}")
print()

with torch.no_grad():
    while True:
        logits = policy_net(observation)
        probabilities = torch.softmax(logits, dim = -1)
        print(f"probabilities: {probabilities.tolist()}")
        if args.deterministic:
            action = logits.argmax(dim = -1).item()
        else:
            action = torch.distributions.Categorical(probabilities).sample().item()

        observation, _reward, terminated, truncated, _info = env.step(action)
        print(f"action {action} -> money {observation[0]}, croissants {observation[1]}, turns left {observation[3]}")
        print()

        if terminated or truncated:
            print(f"The AI ate {observation[1]} croissants.")
            break

env.close()
