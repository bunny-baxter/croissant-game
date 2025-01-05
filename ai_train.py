import datetime
import math
import time
import toml

import tensordict
import torch
from torch import nn
import torchrl

import game_env

DEVICE = "cpu"

hyperparams = toml.load("hyperparams.toml")

env = torchrl.envs.libs.gym.GymEnv("bunny-baxter/CroissantGame-v0", device = DEVICE)

class ConvertToFloat(nn.Module):
    def forward(self, x):
        return x.float()

HIDDEN_NEURON_COUNT = 64

def make_net(out_size):
    return nn.Sequential(
        ConvertToFloat(),
        nn.Linear(env.observation_space.shape[0], HIDDEN_NEURON_COUNT, device = DEVICE),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURON_COUNT, HIDDEN_NEURON_COUNT, device = DEVICE),
        nn.ReLU(),
        nn.Linear(HIDDEN_NEURON_COUNT, out_size, device = DEVICE),
    )

policy_net = make_net(env.action_spec.shape[0])

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

value_module = torchrl.modules.ValueOperator(
    module = make_net(1),
    in_keys = ["observation"],
)

STEPS_PER_BATCH = 500

total_steps = hyperparams["total_steps"]

collector = torchrl.collectors.SyncDataCollector(
    env,
    policy_module,
    frames_per_batch = STEPS_PER_BATCH,
    total_frames = total_steps,
    split_trajs = False,
    device = DEVICE,
    trust_policy = True,
)

replay_buffer = torchrl.data.replay_buffers.ReplayBuffer(
    storage = torchrl.data.replay_buffers.storages.LazyTensorStorage(max_size = STEPS_PER_BATCH),
    sampler = torchrl.data.replay_buffers.samplers.SamplerWithoutReplacement(),
)

# GAE paper: https://arxiv.org/pdf/1506.02438
# Gamma "downweigh[s] rewards corresponding to delayed effects."
ADVANTAGE_GAMMA = 0.999
# Lambda is "the exponentially-weighted average" over estimates at multiple steps (I think).
ADVANTAGE_LAMBDA = 0.95

advantage_module = torchrl.objectives.value.GAE(
    gamma = ADVANTAGE_GAMMA,
    lmbda = ADVANTAGE_LAMBDA,
    value_network = value_module,
    average_gae = True,
)

CLIP_EPSILON = 0.2

entropy_multiplier = hyperparams["entropy_multiplier"]
loss_module = torchrl.objectives.ClipPPOLoss(
    actor_network = policy_module,
    critic_network = value_module,
    clip_epsilon = CLIP_EPSILON,
    entropy_bonus = entropy_multiplier > 0,
    entropy_coef = entropy_multiplier,
)

optimizer = torch.optim.Adam(loss_module.parameters(), lr = hyperparams["initial_learning_rate"])

scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = hyperparams["learning_rate_decay"])

EVAL_MAX_STEPS = 24

def evaluate(print_all_steps):
    with torchrl.envs.utils.set_exploration_type(torchrl.envs.utils.ExplorationType.DETERMINISTIC), torch.no_grad():
        tensordict = env.reset()

        score = 0
        reward = 0
        for _ in range(EVAL_MAX_STEPS):
            policy_module(tensordict)
            env.step(tensordict)
            action = tensordict["action"].argmax(dim = -1)

            tensordict = tensordict["next"]

            observation = tensordict["observation"]
            score = observation[1]
            if print_all_steps:
                print(f"action {action} -> money {observation[0]}, croissants {score}, turns left {observation[3]}")

            if tensordict["terminated"].item() or tensordict["truncated"].item():
                reward = round(tensordict["reward"].item() * 1000) / 1000
                break
        print(f"[eval] croissants: {score} (reward: {reward})")

EPOCHS = 5
SUB_BATCH_SIZE = 256
GRAD_NORM_CLIP = 1.0

total_iterations = total_steps // STEPS_PER_BATCH
iteration_zfill = int(math.ceil(math.log(total_iterations, 10)))

training_start_time = time.time()

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
        learning_rate = scheduler.get_last_lr()[0]
        print(f"[iteration {str(i+1).zfill(iteration_zfill)}/{total_iterations}] average reward: {rouned_average_reward}, learning rate: {learning_rate}")

    scheduler.step()

    # Skips last iteration to do the final evaluation below
    if i % 20 == 0:
        evaluate(False)

training_time = time.time() - training_start_time
print(f"training time was {round(training_time * 1000) / 1000}s")

print("[final evaluation]")
evaluate(True)

date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_filename = f"checkpoints/croissantgame_rewardmoney_{date_str}.pt"
torch.save(policy_net, model_filename)
print(f"saved model to {model_filename}")
