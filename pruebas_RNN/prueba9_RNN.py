import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.maze import MazeEnv


class Agent(nn.Module):
    def __init__(self, input_shape, goal_dim, action_dim, hidden_dim=128):
        super(Agent, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.GRU(np.prod(input_shape) + goal_dim, hidden_dim, batch_first=True)
        self.action_layer = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, goal, hidden):
        state_flat = state.view(state.size(0), -1)  # Aplanar el estado
        combined_input = torch.cat((state_flat, goal), dim=-1)
        rnn_out, hidden = self.rnn(combined_input.unsqueeze(1), hidden)
        action = self.action_layer(rnn_out.squeeze(1))
        return action, hidden


def manhattan_distance(a, b):
    return np.abs(a[0] - b[0]) + np.abs(a[1] - b[1])


def agent_reward(
    current_position,
    next_position,
    current_state,
    next_state,
    goal_position,
    visited_positions,
):
    reward = 0

    if next_state in visited_positions:
        reward -= visited_positions[next_state]  # Penalizar la repetición de estados
        visited_positions[next_state] += 1
    else:
        reward += 1  # Recompensa por alcanzar un estado nuevo
        visited_positions[next_state] = 1

    # Recompensa basada en la distancia de Manhattan al objetivo
    distance_to_goal = manhattan_distance(next_position, goal_position)
    reward += 1 / (distance_to_goal + 1)

    # Recompensa adicional por alcanzar el objetivo
    if next_position == goal_position:
        reward += 10

    return reward


# Definir una función de pérdida personalizada basada en la recompensa
def custom_loss(reward):
    return reward  # Minimizar la recompensa negativa (equivalente a maximizar la recompensa positiva)


# Inicialización
env = MazeEnv((5, 5), False, 3, "human")
state_shape = env.maze_shape  # Obtener la forma del estado del laberinto
goal_dim = 2  # Dimensión del objetivo (coordenadas x, y)
action_dim = 4  # Número de posibles acciones (ajustar según sea necesario)
agent = Agent(input_shape=state_shape, goal_dim=goal_dim, action_dim=action_dim)

optimizer = optim.Adam(agent.parameters(), lr=0.001)

# Entrenamiento
num_episodes = 100  # Número de episodios de entrenamiento (ajustar según sea necesario)
for episode in range(num_episodes):
    state = env.reset()
    state = torch.tensor(state, dtype=torch.float32).unsqueeze(
        0
    )  # Convertir el estado a tensor
    hidden = None
    done = False
    total_reward = 0
    visited_positions = {}

    # Seleccionar una moneda o la salida como objetivo
    goal_position = env.get_coin_position()[0] if env.get_coin_position() else env._end
    goal = torch.tensor(goal_position, dtype=torch.float32).unsqueeze(
        0
    )  # Convertir el objetivo a tensor

    while not done:
        current_position = env.get_current_position()

        # Agente selecciona la acción
        action, hidden = agent(state, goal, hidden)
        action = torch.argmax(action).item()

        # Ejecutar la acción en el entorno
        next_state, _, done = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(
            0
        )  # Convertir el siguiente estado a tensor
        next_position = env.get_current_position()

        # Calcular la recompensa del agente
        reward = agent_reward(
            current_position,
            next_position,
            state,
            next_state,
            goal_position,
            visited_positions,
        )

        # Actualizar el agente
        optimizer.zero_grad()
        loss = custom_loss(
            reward
        )  # Usar la función de pérdida personalizada basada en la recompensa
        loss_tensor = (
            torch.tensor(loss, dtype=torch.float32).unsqueeze(0).requires_grad_(True)
        )  # Convertir loss a tensor y requerir gradientes
        loss_tensor.backward()
        optimizer.step()

        state = next_state
        total_reward += reward

        # Renderizar el entorno
        env.render()

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

env.close()
