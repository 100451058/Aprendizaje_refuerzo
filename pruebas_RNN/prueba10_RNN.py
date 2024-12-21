from math import e
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from src.maze import MazeEnv


# Definir el modelo RNN con PyTorch
class RNNAgent(nn.Module):
    def __init__(self, input_shape, action_space):
        super(RNNAgent, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.lstm = nn.LSTM(
            16 * (input_shape[1] // 2) * (input_shape[2] // 2), 64, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, action_space)
        )

    def forward(self, x, hidden_state=None):
        batch_size, seq_len, c, h, w = x.size()
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.conv(x)
        x = x.view(batch_size, seq_len, -1)  # Flatten para LSTM
        x, hidden_state = self.lstm(x, hidden_state)
        x = self.fc(x[:, -1, :])  # Usar solo la última salida del LSTM
        return x, hidden_state


# Clase para gestionar el agente y su entrenamiento
class Agent:
    def __init__(
        self,
        input_shape,
        action_space,
        lr=0.001,
        gamma=0.99,
        buffer_size=10000,
        batch_size=64,
    ):
        self.action_space = action_space
        self.gamma = gamma
        self.batch_size = batch_size

        self.model = RNNAgent(input_shape, action_space)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        self.memory = deque(maxlen=buffer_size)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, epsilon=0.1):
        if np.random.rand() < epsilon:
            return random.randint(0, self.action_space - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(
            0
        )  # Agregar batch dimension
        q_values, _ = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        q_values, _ = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        next_q_values, _ = self.model(next_states)
        max_next_q_values = torch.max(next_q_values, dim=1)[0]
        expected_q_values = rewards + (1 - dones) * self.gamma * max_next_q_values

        loss = self.criterion(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)


# Función de entrenamiento
def train_agent(
    env, agent, episodes=500, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1
):
    for episode in range(episodes):
        state = env.reset()
        state = np.expand_dims(state, axis=0)  # Expandir dimensión para secuencia
        done = False
        total_reward = 0
        hidden_state = None

        while not done:
            action = agent.act(state, epsilon)
            next_state, reward, done = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)  # Expandir dimensión
            total_reward += reward

            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
            env.render()

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        print(
            f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}, Epsilon: {epsilon:.3f}"
        )


# Configuración del entorno
maze_size = (10, 10)
env = MazeEnv(maze_size, False, 3, "human")

# Inicializar el agente
input_shape = (1, env.maze_shape[0], env.maze_shape[1])
action_space = env.action_space.n

agent = Agent(input_shape, action_space)

# Entrenar el agente
train_agent(env, agent, episodes=1000)

# Guardar el modelo
agent.save("rnn_agent.pth")
