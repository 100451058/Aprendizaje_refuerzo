"""Esta es la primera prueba funcional"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.maze import MazeEnv


class Manager(nn.Module):
    def __init__(self, input_dim, goal_dim, hidden_dim=128):
        super(Manager, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.goal_layer = nn.Linear(hidden_dim, goal_dim)
        self.optimizer = None

    def forward(self, state, hidden):
        rnn_out, hidden = self.rnn(state.unsqueeze(1), hidden)
        goal = self.goal_layer(rnn_out.squeeze(1))
        return goal, hidden

    def train_manager(self, state, hidden, worker_rewards, gamma, t):
        goal, hidden = self.forward(state, hidden)
        manager_reward = sum(worker_rewards) * (gamma**t)
        manager_loss = -torch.tensor(manager_reward, dtype=torch.float32).detach()

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        self.optimizer.zero_grad()
        manager_loss.backward()
        self.optimizer.step()
        return manager_loss.item()


class Worker(nn.Module):
    def __init__(self, input_dim, goal_dim, action_dim, hidden_dim=128):
        super(Worker, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(input_dim + goal_dim, hidden_dim, batch_first=True)
        self.action_layer = nn.Linear(hidden_dim, action_dim)
        self.optimizer = None

    def forward(self, state, goal, hidden):
        combined_input = torch.cat((state, goal), dim=-1).unsqueeze(1)
        rnn_out, hidden = self.rnn(combined_input, hidden)
        action_logits = self.action_layer(rnn_out.squeeze(1))
        return action_logits, hidden

    def train_worker(self, state, goal, hidden, action):
        action_logits, hidden = self.forward(state, goal, hidden)
        worker_loss = nn.CrossEntropyLoss()(
            action_logits, torch.tensor([action], dtype=torch.long)
        )

        if self.optimizer is None:
            self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        self.optimizer.zero_grad()
        worker_loss.backward()
        self.optimizer.step()
        return worker_loss.item()


# Environment Setup
env = MazeEnv(maze_size=(5, 5), coin_task=3)
state_shape = env.observation_space.shape
action_dim = env.action_space.n

# Define hyperparameters
hidden_dim = 128
goal_dim = 8
num_episodes = 1000
gamma = 0.99
goal_update_interval = 10

# Initialize models
manager = Manager(np.prod(state_shape), goal_dim, hidden_dim)
worker = Worker(np.prod(state_shape), goal_dim, action_dim, hidden_dim)

# Training Loop
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)

    # Initialize hidden states
    manager_hidden = torch.zeros(1, 1, hidden_dim)
    worker_hidden = torch.zeros(1, 1, hidden_dim).clone().detach()

    manager_goals = []
    worker_rewards = []
    done = False
    t = 0

    while not done:
        if t % goal_update_interval == 0:
            goal, manager_hidden = manager.forward(
                state.detach(), manager_hidden.detach()
            )
            goal = goal.detach()
            manager_goals.append(goal)

        action_logits, worker_hidden = worker.forward(
            state.detach(), goal.detach(), worker_hidden.detach()
        )
        action = torch.argmax(action_logits).item()

        assert (
            0 <= action < action_dim
        ), f"Invalid action index: {action}, expected in range 0 to {action_dim-1}"

        if action is not None:
            result = env.step(action)
            if len(result) == 3:
                next_state, reward, done = result
                info = {}
            else:
                next_state, reward, done, _, info = result

            next_state = (
                torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)
            )

            # Train Worker
            worker_loss = worker.train_worker(
                state.detach(), goal.detach(), worker_hidden.detach(), action
            )
            print(f"Worker loss: {worker_loss}")

            # Track rewards
            worker_rewards.append(float(reward))
            state = next_state
            t += 1

    # Train Manager
    manager_loss = manager.train_manager(worker_rewards, gamma, t)
    print(f"Manager loss: {manager_loss}")

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {sum(worker_rewards)}")

print("Training complete.")
