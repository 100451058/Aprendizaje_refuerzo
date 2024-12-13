import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.maze import MazeEnv


# Define the Feudal Networks using RNN based on the repository
class Manager(nn.Module):
    def __init__(self, input_dim, goal_dim, hidden_dim=128):
        super(Manager, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.goal_layer = nn.Linear(hidden_dim, goal_dim)

    def forward(self, state, hidden):
        rnn_out, hidden = self.rnn(state.unsqueeze(1), hidden)
        goal = self.goal_layer(rnn_out.squeeze(1))
        return goal, hidden


class Worker(nn.Module):
    def __init__(self, input_dim, goal_dim, action_dim, hidden_dim=128):
        super(Worker, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim + goal_dim, hidden_dim, batch_first=True)
        self.action_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal, hidden):
        combined_input = torch.cat((state, goal), dim=-1).unsqueeze(1)
        rnn_out, hidden = self.rnn(combined_input, hidden)
        action_logits = self.action_layer(rnn_out.squeeze(1))
        return action_logits, hidden


# Environment Setup
env = MazeEnv(maze_size=(5, 5), coin_task=3)
state_shape = env.observation_space.shape
action_dim = env.action_space.n

# Define hyperparameters
learning_rate = 1e-3
hidden_dim = 128
goal_dim = 8
num_episodes = 1000
gamma = 0.99
goal_update_interval = 10

# Initialize models
manager = Manager(np.prod(state_shape), goal_dim, hidden_dim)
worker = Worker(np.prod(state_shape), goal_dim, action_dim, hidden_dim)

# Optimizers
manager_optimizer = optim.Adam(manager.parameters(), lr=learning_rate)
worker_optimizer = optim.Adam(worker.parameters(), lr=learning_rate)

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

    # Initialize hidden states
    manager_hidden = torch.zeros(1, 1, hidden_dim)
    worker_hidden = torch.zeros(1, 1, hidden_dim)
    worker_loss_func = nn.CrossEntropyLoss()

    manager_goals = []
    worker_rewards = []
    done = False
    t = 0

    while not done:
        if t % goal_update_interval == 0:
            goal, manager_hidden = manager(state.detach(), manager_hidden.detach())
            goal = goal.detach()
            manager_goals.append(goal)

        action_logits, worker_hidden = worker(
            state.detach(), goal.detach(), worker_hidden.detach()
        )
        action = torch.argmax(action_logits).item()

        assert 0 <= action < action_dim, f"Invalid action index: {action}"

        result = env.step(action)
        next_state, reward, done = result[:3]
        next_state = (
            torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
        )

        # Update Worker
        worker_loss = worker_loss_func(
            action_logits, torch.tensor([action], dtype=torch.long)
        )
        worker_optimizer.zero_grad()
        worker_loss.backward()
        worker_optimizer.step()

        worker_rewards.append(reward)
        state = next_state
        t += 1

    # Detach hidden states after each episode
    manager_hidden = manager_hidden.detach()
    worker_hidden = worker_hidden.detach()

    # Update Manager
    manager_reward = torch.tensor(worker_rewards, dtype=torch.float32).sum().item() * (
        gamma**t
    )
    manager_loss = -manager_reward
    manager_optimizer.zero_grad()
    manager_loss.backward()
    manager_optimizer.step()

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {sum(worker_rewards)}")


print("Training complete.")
