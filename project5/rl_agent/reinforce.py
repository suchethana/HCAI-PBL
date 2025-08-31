import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .mouse import (
    initialize_grid_with_cheese_types, move, GRID_SIZE, MOUSE, CHEESE,
    TRAP, WALL, ORGANIC_CHEESE, EMPTY, ACTIONS, print_grid_with_cheese_types
)


class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * GRID_SIZE * GRID_SIZE, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.conv_net(x)


class REINFORCE:
    def __init__(self, policy_net, learning_rate=0.01, gamma=0.99):
        self.policy_net = policy_net
        self.optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
        self.gamma = gamma


def get_reward_from_project_spec(cell_content):
    # Task 1 reward spec
    if cell_content in [CHEESE, ORGANIC_CHEESE]:
        return 10.0
    elif cell_content == TRAP:
        return -50.0
    elif cell_content in [EMPTY, WALL]:
        return -0.2
    return -0.2  # safe default


def state_to_tensor(grid, mouse_pos):
    state_tensor = torch.zeros(6, GRID_SIZE, GRID_SIZE)
    state_tensor[0, mouse_pos[0], mouse_pos[1]] = 1
    state_tensor[1, :, :] = torch.from_numpy((grid == CHEESE)).float()
    state_tensor[2, :, :] = torch.from_numpy((grid == TRAP)).float()
    state_tensor[3, :, :] = torch.from_numpy((grid == WALL)).float()
    state_tensor[4, :, :] = torch.from_numpy((grid == ORGANIC_CHEESE)).float()
    state_tensor[5, :, :] = torch.from_numpy((grid == EMPTY)).float()
    return state_tensor.unsqueeze(0)


def compute_loss(states, actions_taken, rewards, policy_net, gamma):
    discounted_rewards = []
    R = 0
    for r in rewards[::-1]:
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    policy_loss = []
    for state, action_index, reward in zip(states, actions_taken, discounted_rewards):
        action_probs = policy_net(state)
        log_prob = torch.log(action_probs.squeeze(0)[action_index])
        policy_loss.append(-log_prob * reward)

    return torch.stack(policy_loss).sum()


def compute_loss_with_penalty(states, actions_taken, policy_net, original_policy_net, gamma, reward_model):
    discounted_rewards = []
    R = 0
    for state in reversed(states):
        with torch.no_grad():
            r = reward_model(state)
        R = r.item() + gamma * R
        discounted_rewards.insert(0, R)
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)

    policy_loss = []
    for state, action_index, reward in zip(states, actions_taken, discounted_rewards):
        action_probs = policy_net(state)
        log_prob = torch.log(action_probs.squeeze(0)[action_index])
        policy_loss.append(-log_prob * reward)

    kl_penalty = 0.1
    kl_divergence_loss = 0
    for state in states:
        with torch.no_grad():
            probs_original = original_policy_net(state).squeeze(0)
        probs_new = policy_net(state).squeeze(0)
        kl_divergence_loss += torch.sum(probs_original * torch.log(probs_original / (probs_new + 1e-10)))

    return torch.stack(policy_loss).sum() + kl_penalty * kl_divergence_loss
