"""En esta prueba, se implementa el algoritmo feudal empleando RNN en lugar de q-learning"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.maze import MazeEnv

# Parámetros
episodios_agente = 2000
hidden_size = 64  # Tamaño de las capas ocultas
action_size = 4  # Número de acciones posibles (arriba, abajo, izquierda, derecha)


# Manager con RNN y CNN
class ManagerRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, maze_shape):
        super(ManagerRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 32 * maze_shape[0] * maze_shape[1]
        self.rnn = nn.LSTM(conv_output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)  # Predecir coordenadas (x, y) del objetivo

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        conv_out = self.conv(x)  # Procesar estado del laberinto
        conv_out = conv_out.view(batch_size, 1, -1)  # Añadir dimensión temporal
        rnn_out, hidden = self.rnn(conv_out, hidden)
        goal = self.fc(rnn_out[:, -1, :])  # Predicción final
        return goal, hidden


# Worker con RNN y CNN
class WorkerRNN(nn.Module):
    def __init__(self, input_channels, hidden_size, maze_shape, action_size):
        super(WorkerRNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        conv_output_size = 32 * maze_shape[0] * maze_shape[1]
        self.rnn = nn.LSTM(conv_output_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)  # Predecir acción

    def forward(self, x, hidden):
        batch_size = x.shape[0]
        conv_out = self.conv(x)  # Procesar estado del laberinto
        conv_out = conv_out.view(batch_size, 1, -1)  # Añadir dimensión temporal
        rnn_out, hidden = self.rnn(conv_out, hidden)
        action_probs = self.fc(rnn_out[:, -1, :])  # Predicción final
        return action_probs, hidden


# Inicializar entorno y redes
torch.autograd.set_detect_anomaly(True)
env = MazeEnv((5, 5), False, 3, "human")
maze_shape = env.maze_shape

manager_net = ManagerRNN(
    input_channels=1, hidden_size=hidden_size, maze_shape=maze_shape
)
worker_net = WorkerRNN(
    input_channels=2,
    hidden_size=hidden_size,
    maze_shape=maze_shape,
    action_size=action_size,
)

manager_optimizer = optim.Adam(manager_net.parameters(), lr=0.01)
worker_optimizer = optim.Adam(worker_net.parameters(), lr=0.01)

loss_fn = nn.MSELoss()


# Preprocesar estado del laberinto para redes neuronales
def preprocess_maze(maze, goal=None):
    maze_tensor = (
        torch.tensor(maze, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    )  # (1, 1, H, W)
    if goal is not None:
        goal_map = np.zeros_like(maze, dtype=np.float32)
        goal_map[goal[0], goal[1]] = 1.0
        goal_tensor = (
            torch.tensor(goal_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        )
        return torch.cat((maze_tensor, goal_tensor), dim=1)  # (1, 2, H, W)
    return maze_tensor


# Entrenamiento del agente
for episode in range(episodios_agente):
    maze = env.reset()
    state = env.get_current_position()
    done = False
    manager_hidden = None
    worker_hidden = None

    # Selección del primer objetivo
    manager_input = preprocess_maze(maze)
    goal_pred, manager_hidden = manager_net(manager_input, manager_hidden)
    goal = goal_pred.squeeze(0).detach().numpy().round().astype(int)

    while not done:
        # Worker selecciona acción
        worker_input = preprocess_maze(maze, goal)
        action_pred, worker_hidden = worker_net(worker_input, worker_hidden)
        action = torch.argmax(action_pred).item()

        # Realizar acción
        _, reward, done = env.step(action)
        next_state = env.get_current_position()
        next_maze = env._maze  # Obtener el nuevo estado del laberinto

        # Actualizar Worker
        next_worker_input = preprocess_maze(next_maze, goal)
        with torch.no_grad():
            next_action_pred, _ = worker_net(next_worker_input, worker_hidden)
            target = reward + 0.9 * torch.max(next_action_pred)  # Q-target

        worker_optimizer.zero_grad()  # Reiniciar gradientes
        loss = loss_fn(
            action_pred[0, action], target.detach()
        )  # Separar el target del grafo
        loss.backward()  # Retropropagación
        worker_optimizer.step()

        # Si se alcanza el objetivo, seleccionar un nuevo objetivo
        if np.array_equal(next_state, goal):
            manager_input = preprocess_maze(next_maze)
            goal_pred, manager_hidden = manager_net(manager_input, manager_hidden)
            goal = goal_pred.squeeze(0).detach().numpy().round().astype(int)

        maze = next_maze

env.close()
