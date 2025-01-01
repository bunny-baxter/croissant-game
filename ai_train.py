import math

import tensordict
import torch
from torch import nn
import torchrl

import game_env

DEVICE = "cpu"
HIDDEN_NEURON_COUNT = 16

env = torchrl.envs.libs.gym.GymEnv("bunny-baxter/CroissantGame-v0", device = DEVICE)

class ConvertToFloat(nn.Module):
    def forward(self, x):
        return x.float()

policy_net = nn.Sequential(
    ConvertToFloat(),
    nn.Linear(env.observation_space.shape[0], HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, env.action_spec.shape[0], device = DEVICE),
)

policy_module = torchrl.modules.ProbabilisticActor(
    module = tensordict.nn.TensorDictModule(
        policy_net,
        in_keys = ["observation"],
        out_keys = ["logits"]
    ),
    spec = env.action_spec,
    in_keys = ["logits"],
    distribution_class = torchrl.modules.distributions.OneHotCategorical,
    return_log_prob = True,
)

value_net = nn.Sequential(
    ConvertToFloat(),
    nn.Linear(env.observation_space.shape[0], HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
    nn.ReLU(),
    nn.Linear(HIDDEN_NEURON_COUNT, 1, device = DEVICE),
)

class ValueModule(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, observation):
        observation = observation.float()
        return self.network(observation)

value_module = torchrl.modules.ValueOperator(
    module = value_net,
    in_keys = ["observation"],
)

STEPS_PER_BATCH = 500
TOTAL_STEPS = 10_000

collector = torchrl.collectors.SyncDataCollector(
    env,
    policy_module,
    frames_per_batch = STEPS_PER_BATCH,
    total_frames = TOTAL_STEPS,
    split_trajs = False,
    device = DEVICE,
    trust_policy = True,
)

replay_buffer = torchrl.data.replay_buffers.ReplayBuffer(
    storage = torchrl.data.replay_buffers.storages.LazyTensorStorage(max_size = STEPS_PER_BATCH),
    sampler = torchrl.data.replay_buffers.samplers.SamplerWithoutReplacement(),
)

# These are two different discounting meta-params.
# I don't really understand the difference between them.
ADVANTAGE_GAMMA = 0.99
ADVANTAGE_LAMBDA = 0.95

advantage_module = torchrl.objectives.value.GAE(
    gamma = ADVANTAGE_GAMMA,
    lmbda = ADVANTAGE_LAMBDA,
    value_network = value_module,
    average_gae = True,
)

CLIP_EPSILON = 0.2
ENTROPY_MULTIPLIER = 1e-4

loss_module = torchrl.objectives.ClipPPOLoss(
    actor_network = policy_module,
    critic_network = value_module,
    clip_epsilon = CLIP_EPSILON,
    entropy_bonus = ENTROPY_MULTIPLIER > 0,
    entropy_coef = ENTROPY_MULTIPLIER,
)

LEARNING_RATE = 0.0003

optimizer = torch.optim.Adam(loss_module.parameters(), LEARNING_RATE)

EVAL_MAX_STEPS = 24

def evaluate(print_all_steps):
    with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.DETERMINISTIC), torch.no_grad():
        tensordict = env.reset()

        reward = 0
        for _ in range(EVAL_MAX_STEPS):
            policy_module(tensordict)
            env.step(tensordict)
            action = tensordict["action"].argmax(dim = -1)

            tensordict = tensordict["next"]

            reward = int(tensordict["reward"].item())
            if print_all_steps:
                observation = tensordict["observation"]
                print(f"action {action} -> money {observation[0]}, reward {reward}, turns left {observation[3]}")

            if tensordict["terminated"].item() or tensordict["truncated"].item():
                break
        print(f"[eval] reward: {reward}")

EPOCHS = 10
SUB_BATCH_SIZE = 64
GRAD_NORM_CLIP = 1.0

total_iterations = TOTAL_STEPS // STEPS_PER_BATCH
iteration_zfill = int(math.ceil(math.log(total_iterations, 10)))

for i, tensordict_data in enumerate(collector):
    # Run PPO
    for _ in range(EPOCHS):
        advantage_module(tensordict_data)
        data_view = tensordict_data.reshape(-1)
        replay_buffer.extend(data_view.cpu())
        for _ in range(STEPS_PER_BATCH // SUB_BATCH_SIZE):
            subdata = replay_buffer.sample(SUB_BATCH_SIZE)
            losses = loss_module(subdata.to(DEVICE))
            loss_value = losses["loss_objective"] + losses["loss_critic"] + losses["loss_entropy"]
            loss_value.backward()
            torch.nn.utils.clip_grad_norm_(loss_module.parameters(), GRAD_NORM_CLIP)
            optimizer.step()
            optimizer.zero_grad()

    if i == 0 or i % 5 == 4:
        average_reward = tensordict_data["next", "reward"].mean().item()
        rouned_average_reward = round(average_reward * 10000) / 10000
        print(f"[iteration {str(i+1).zfill(iteration_zfill)}/{total_iterations}] average reward: {rouned_average_reward}")

    # Skips last iteration to do the final evaluation below
    if i % 20 == 0:
        evaluate(False)

print("[final evaluation]")
evaluate(True)
