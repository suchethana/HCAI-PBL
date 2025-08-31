import torch
import torch.nn as nn
import torch.optim as optim
from .mouse import GRID_SIZE


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

    def train(self, num_episodes, num_trajectories, max_steps):
        pass