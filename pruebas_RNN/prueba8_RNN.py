import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.maze import MazeEnv
import keyboard  # Library for key input


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
maze_shape = (5, 5)
env = MazeEnv(maze_shape, False, 5, "human")
state_shape = env.maze_shape
action_dim = 4

# Define hyperparameters
hidden_dim = 128
goal_dim = 8
manager = Manager(np.prod(state_shape), goal_dim, hidden_dim)
worker = Worker(np.prod(state_shape), goal_dim, action_dim, hidden_dim)

# Optimizers
manager_optimizer = optim.Adam(manager.parameters(), lr=1e-3)
worker_optimizer = optim.Adam(worker.parameters(), lr=1e-3)

# Loss functions
worker_loss_fn = nn.CrossEntropyLoss()  # Worker loss based on actions
manager_loss_fn = nn.MSELoss()  # Manager loss based on goal performance

# Training Loop
state = env.reset()
state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
manager_hidden = torch.zeros(1, 1, hidden_dim)
worker_hidden = torch.zeros(1, 1, hidden_dim)
done = False

# Coins to collect
coins = set(tuple(map(int, pos)) for pos in env.get_coin_position())
current_coin_index = 0

# Exploration parameters
epsilon = 1.0
min_epsilon = 0.1
epsilon_decay = 0.995

max_iterations = 5000
iteration_count = 0

# Render toggle
render_enabled = False

# Exit position for the maze
exit_position = (maze_shape[0] * 2 - 1, maze_shape[1] * 2 - 1)

while not done:
    # Check for key inputs to toggle render
    if keyboard.is_pressed("r"):
        render_enabled = True
    if keyboard.is_pressed("n"):
        render_enabled = False

    # Manager assigns the next coin as the goal
    goal, manager_hidden = manager(state.detach(), manager_hidden.detach())
    goal_position = next(iter(coins))  # Select the next coin as goal

    # Worker plans and executes actions towards the goal
    action_logits, worker_hidden = worker(
        state.detach(), goal.detach(), worker_hidden.detach()
    )
    if np.random.rand() < epsilon:
        action = np.random.choice(action_dim)  # Random action (exploration)
    else:
        action = torch.argmax(action_logits).item()  # Best action (exploitation)

    next_state, _, _ = env.step(action)
    iteration_count += 1
    current_position = tuple(map(int, env.get_current_position()))

    # Calculate rewards
    manhattan_distance = abs(current_position[0] - goal_position[0]) + abs(
        current_position[1] - goal_position[1]
    )
    reward = -0.1 * manhattan_distance  # Small penalty for distance

    if current_position == goal_position:
        reward += 10  # Bonus for reaching the goal
        print(f"Manager assigned goal: {goal_position}, Worker reached it!")
        coins.remove(goal_position)  # Remove the coin from the set
        if not coins:
            print("All coins collected. Resetting environment.")
            env = MazeEnv(maze_shape, False, 5, "human")
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            manager_hidden = torch.zeros(1, 1, hidden_dim)
            worker_hidden = torch.zeros(1, 1, hidden_dim)
            coins = set(tuple(map(int, pos)) for pos in env.get_coin_position())
            iteration_count = 0
        else:
            goal_position = next(
                iter(coins)
            )  # Select a new goal  # Remove the coin from the list
        if current_coin_index >= len(coins):
            print("All coins collected. Resetting environment.")
            env = MazeEnv(maze_shape, False, 5, "human")
            state = env.reset()
            state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
            manager_hidden = torch.zeros(1, 1, hidden_dim)
            worker_hidden = torch.zeros(1, 1, hidden_dim)
            coins = [tuple(map(int, pos)) for pos in env.get_coin_position()]
            current_coin_index = 0
            iteration_count = 0
        else:
            goal_position = tuple(map(int, coins[current_coin_index]))
    elif current_position in coins and current_position != goal_position:
        reward -= 5  # Penalty for wrong coin
        print(f"Worker picked the wrong coin at {current_position}.")
    elif current_position == tuple(map(int, exit_position)):
        if current_position == goal_position:
            reward += 20  # Bonus for reaching the exit as a goal
            print("Worker reached the exit as assigned by the Manager!")
        else:
            reward -= 15  # Penalty for reaching the exit without it being a goal
            print("Worker reached the exit, but it was not the assigned goal!")
        print("Scenario failed. Resetting environment.")
        env = MazeEnv(maze_shape, False, 5, "human")
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        manager_hidden = torch.zeros(1, 1, hidden_dim)
        worker_hidden = torch.zeros(1, 1, hidden_dim)
        coins = [tuple(map(int, pos)) for pos in env.get_coin_position()]
        current_coin_index = 0
        iteration_count = 0

    # Prepare target for training
    target_action = torch.tensor([action], dtype=torch.long)
    goal_reward = torch.tensor([reward], dtype=torch.float32).repeat(
        goal_dim
    )  # Match goal dimensions

    # Compute losses
    worker_loss = worker_loss_fn(action_logits, target_action)
    manager_loss = manager_loss_fn(goal, goal_reward.unsqueeze(0))

    # Backpropagation
    worker_optimizer.zero_grad()
    worker_loss.backward()
    worker_optimizer.step()

    manager_optimizer.zero_grad()
    manager_loss.backward()
    manager_optimizer.step()

    # Update epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

    # Render environment if enabled
    if render_enabled:
        env.render()

    # Update state
    state = torch.tensor(next_state, dtype=torch.float32).flatten().unsqueeze(0)

    # Check if scenario is complete
    if current_coin_index >= len(coins) or iteration_count >= max_iterations:
        print("Scenario complete or maximum iterations reached. Resetting environment.")
        env = MazeEnv(maze_shape, False, 5, "human")
        state = env.reset()
        state = torch.tensor(state, dtype=torch.float32).flatten().unsqueeze(0)
        manager_hidden = torch.zeros(1, 1, hidden_dim)
        worker_hidden = torch.zeros(1, 1, hidden_dim)
        coins = [tuple(pos) for pos in env.get_coin_position()]
        current_coin_index = 0
        iteration_count = 0

print("Training completed.")
