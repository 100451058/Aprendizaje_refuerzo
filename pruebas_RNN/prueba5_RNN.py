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
env = MazeEnv((5, 5), False, 3, 'human')
state_shape = env.maze_shape
action_dim = 4

# Define hyperparameters
hidden_dim = 128
goal_dim = 8
manager = Manager(np.prod(state_shape), goal_dim, hidden_dim)
worker = Worker(np.prod(state_shape), goal_dim, action_dim, hidden_dim)

# Training Loop
state = env.reset()
state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
manager_hidden = torch.zeros(1, 1, hidden_dim)
worker_hidden = torch.zeros(1, 1, hidden_dim)
done = False

# Coins to collect
coins = [tuple(pos) for pos in env.get_coin_position()]
current_coin_index = 0

epsilon = 1.0  # Exploration rate
min_epsilon = 0.1
epsilon_decay = 0.995

while not done:
    if done:
        break

    # Manager assigns the next coin as the goal
    goal, manager_hidden = manager(state, manager_hidden)
    goal_position = coins[current_coin_index]

    # Worker plans and executes actions towards the goal
    action_logits, worker_hidden = worker(state, goal, worker_hidden)
    if np.random.rand() < epsilon:
        action = np.random.choice(action_dim)  # Random action (exploration with higher randomness)
    else:
        action = torch.argmax(action_logits).item()  # Best action (exploitation)
    env.step(action)
    env.render()
    current_position = env.get_current_position()

    # Check if the agent is stuck
    # Decay epsilon after each step
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Update the state
    state = torch.tensor(env._maze.flatten(), dtype=torch.float32).unsqueeze(0)

    # Check if the worker reached the current coin
    if env.get_current_position() == goal_position:
        current_coin_index += 1
        if current_coin_index >= len(coins):
            done = True
