import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .reinforce import GRID_SIZE


class RewardModel(nn.Module):
    def __init__(self, input_channels=6):
        super(RewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * GRID_SIZE * GRID_SIZE, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, state):
        return self.net(state)


def train_reward_model(reward_model, feedbacks, num_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(reward_model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for feedback in feedbacks:
            traj1_states = feedback['trajectory1_states']
            traj2_states = feedback['trajectory2_states']
            preference = feedback['preference']

            score1 = sum(reward_model(state) for state in traj1_states)
            score2 = sum(reward_model(state) for state in traj2_states)

            if preference == 1:
                loss = -torch.log(torch.sigmoid(score1 - score2))
            else:
                loss = -torch.log(torch.sigmoid(score2 - score1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()