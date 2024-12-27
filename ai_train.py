from collections import defaultdict

import tensordict
import torch
from torch import nn
import torchrl

import game_env

DEVICE = "cpu"
HIDDEN_NEURON_COUNT = 16

env = torchrl.envs.libs.gym.GymEnv("bunny-baxter/CroissantGame-v0", device = DEVICE)

policy_net = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, env.action_spec.shape[0], device = DEVICE),
)

def policy_forward(observation):
    observation = observation.float()
    action_logits = policy_net(observation)
    action_probabilities = torch.softmax(action_logits, dim = -1)
    action = torch.distributions.Categorical(action_probabilities).sample()
    return {
        "action": action,
    }

policy_module = tensordict.nn.TensorDictModule(
    policy_forward,
    in_keys = ["observation"],
    out_keys = ["action"]
)

value_net = nn.Sequential(
    nn.Linear(env.observation_space.shape[0], HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, 1, device = DEVICE),
)

def value_forward(observation):
    observation = observation.float()
    return value_net(observation)

value_module = torchrl.modules.ValueOperator(
    module = value_forward,
    in_keys = ["observation"],
)

# TODO: Temporary code to test policy and value modules are set up correctly.
policy_out = policy_module(env.reset())
print("Action from policy module:", policy_out["action"])
value_out = value_module(env.reset())
print("Value from value module:", value_out["state_value"])
