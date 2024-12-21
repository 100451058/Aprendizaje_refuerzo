import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gymnasium as gym
from collections import deque
import random
from src.maze import MazeEnv


# -----------------------------
# Perception Layer
# -----------------------------
class PerceptionLayer(nn.Module):
    def __init__(self, input_shape, hidden_dim):
        super(PerceptionLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(input_shape[0] * input_shape[1] * 32, hidden_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# -----------------------------
# Manager Network
# -----------------------------
class Manager(nn.Module):
    def __init__(self, hidden_dim, goal_dim):
        super(Manager, self).__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.goal_fc = nn.Linear(hidden_dim, goal_dim)

    def forward(self, x, hx):
        lstm_out, hx = self.lstm(x, hx)
        goal = self.goal_fc(lstm_out)
        goal = F.normalize(goal, p=2, dim=-1)  # Normalize goal vector
        return goal, hx


# -----------------------------
# Worker Network
# -----------------------------
class Worker(nn.Module):
    def __init__(self, hidden_dim, goal_dim, num_actions):
        super(Worker, self).__init__()
        self.lstm = nn.LSTM(hidden_dim + goal_dim, hidden_dim, batch_first=True)
        self.action_fc = nn.Linear(hidden_dim, num_actions)

    def forward(self, x, goal, hx):
        input_combined = torch.cat([x, goal], dim=-1)
        lstm_out, hx = self.lstm(input_combined, hx)
        action_logits = self.action_fc(lstm_out)
        return action_logits, hx


# -----------------------------
# FuN Model (Combining Manager & Worker)
# -----------------------------
class FuN(nn.Module):
    def __init__(self, input_shape, hidden_dim, goal_dim, num_actions):
        super(FuN, self).__init__()
        self.perception = PerceptionLayer(input_shape, hidden_dim)
        self.manager = Manager(hidden_dim, goal_dim)
        self.worker = Worker(hidden_dim, goal_dim, num_actions)

    def forward(self, state, manager_hx, worker_hx):
        features = self.perception(state)
        manager_goal, manager_hx = self.manager(features.unsqueeze(1), manager_hx)
        action_logits, worker_hx = self.worker(
            features.unsqueeze(1), manager_goal, worker_hx
        )
        return action_logits, manager_goal, manager_hx, worker_hx


# -----------------------------
# Training Loop
# -----------------------------
# Environment setup
env = MazeEnv(maze_size=(10, 10), coin_task=5, fixed_maze=True, render_mode="human")

# Hyperparameters
input_shape = env.maze_shape
hidden_dim = 128
goal_dim = 16
num_actions = env.action_space.n

# Model, optimizer, and loss
model = FuN(input_shape, hidden_dim, goal_dim, num_actions)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

episodes = 1000
gamma = 0.99


def preprocess_state(state):
    return torch.tensor(state, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


for episode in range(episodes):
    state = env.reset()
    manager_hx = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
    worker_hx = (torch.zeros(1, 1, hidden_dim), torch.zeros(1, 1, hidden_dim))
    total_reward = 0

    for t in range(500):  # Limit steps per episode
        state_tensor = preprocess_state(state)
        state_tensor.requires_grad = True  # Ensure state tensor tracks gradients
        action_logits, goal, manager_hx, worker_hx = model(
            state_tensor, manager_hx, worker_hx
        )
        action_prob = F.softmax(action_logits.squeeze(1), dim=-1)
        action = torch.multinomial(action_prob, 1).item()

        next_state, reward, done = env.step(action)
        total_reward += reward

        optimizer.zero_grad()
        loss = -reward * torch.log(action_prob[0, action])  # Policy gradient loss
        loss.backward(retain_graph=True)
        optimizer.step()

        if done:
            break
        state = next_state

    print(f"Episode {episode+1}: Total Reward = {total_reward}")

env.close()
