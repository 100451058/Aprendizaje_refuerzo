import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple
from src.maze import MazeEnv
from tensorboardX import SummaryWriter
import keyboard  # Biblioteca para capturar las teclas
import time


class Perception(nn.Module):
    def __init__(self, shape: tuple[int, int], action_space: int = 4) -> None:
        super().__init__()
        w, h = shape
        k, p, s = 2, 1, 2

        self.cn1 = nn.Conv2d(1, 8, k, s, p)
        w, h = self._conv_output_size(w, h, k, s, p)
        self.cn2 = nn.Conv2d(8, 32, k, s, p)
        w, h = self._conv_output_size(w, h, k, s, p)

        self.fc1 = nn.Linear(32 * w * h, action_space * 16)

    def _conv_output_size(self, w, h, k, s, p):
        w = (w - k + 2 * p) // s + 1
        h = (h - k + 2 * p) // s + 1
        return w, h

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.cn1(img))
        x = F.relu(self.cn2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return x


class Manager(nn.Module):
    def __init__(self, dilation, num_actions, device):
        super(Manager, self).__init__()

        hidden_size = num_actions * 16
        self.device = device  # Guardamos el dispositivo

        # Inicializar las memorias en el dispositivo correcto
        self.hx_memory = [
            torch.zeros(1, hidden_size, device=device) for _ in range(dilation)
        ]
        self.cx_memory = [
            torch.zeros(1, hidden_size, device=device) for _ in range(dilation)
        ]

        self.hidden_size = hidden_size
        self.horizon = dilation
        self.index = 0

        self.fc = nn.Linear(hidden_size, hidden_size)
        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc_critic1 = nn.Linear(hidden_size, 50)
        self.fc_critic2 = nn.Linear(50, 1)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.fc(x))

        hx_t_1 = self.hx_memory[self.index]
        cx_t_1 = self.cx_memory[self.index]

        hx, cx = self.lstm(x, (hx_t_1, cx_t_1))

        # Actualizar las memorias
        self.hx_memory[self.index] = hx
        self.cx_memory[self.index] = cx
        self.index = (self.index + 1) % self.horizon

        goal = cx
        value = F.relu(self.fc_critic1(goal))
        value = self.fc_critic2(value)

        goal_norm = torch.norm(goal, p=2, dim=1).unsqueeze(1)
        goal = goal / goal_norm.detach()
        return goal, (hx, cx), value, x


class Worker(nn.Module):
    def __init__(self, num_actions, device):
        super(Worker, self).__init__()
        self.num_actions = num_actions
        self.device = device  # Guardamos el dispositivo

        self.lstm = nn.LSTMCell(num_actions * 16, num_actions * 16)
        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)

        self.fc = nn.Linear(num_actions * 16, 16, bias=False)

        self.fc_critic1 = nn.Linear(num_actions * 16, 50)
        self.fc_critic1_out = nn.Linear(50, 1)

        self.fc_critic2 = nn.Linear(num_actions * 16, 50)
        self.fc_critic2_out = nn.Linear(50, 1)

        # Inicializar parámetros en el dispositivo correcto
        self.hx_memory = torch.zeros(1, num_actions * 16, device=device)
        self.cx_memory = torch.zeros(1, num_actions * 16, device=device)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, inputs):
        x, (hx, cx), goals = inputs
        hx, cx = self.lstm(x, (hx, cx))

        value_ext = F.relu(self.fc_critic1(hx))
        value_ext = self.fc_critic1_out(value_ext)

        value_int = F.relu(self.fc_critic2(hx))
        value_int = self.fc_critic2_out(value_int)

        worker_embed = hx.view(hx.size(0), self.num_actions, 16)

        goals = goals.sum(dim=1)
        goal_embed = self.fc(goals.detach())
        goal_embed = goal_embed.unsqueeze(-1)

        policy = torch.bmm(worker_embed, goal_embed).squeeze(-1)
        policy = F.softmax(policy, dim=-1)
        return policy, (hx, cx), value_ext, value_int


class FuN(nn.Module):
    def __init__(self, shape, num_actions, horizon, device):
        super(FuN, self).__init__()
        self.horizon = horizon
        self.device = device

        self.percept = Perception(shape, num_actions).to(device)
        self.manager = Manager(self.horizon, num_actions, device).to(device)
        self.worker = Worker(num_actions, device).to(device)

    def forward(self, x, m_lstm, w_lstm, goals_horizon):
        percept_z = self.percept(x)

        goal, m_lstm, m_value, m_state = self.manager((percept_z, m_lstm))
        goals_horizon = torch.cat([goals_horizon[:, 1:], goal.unsqueeze(1)], dim=1)

        policy, (w_hx, w_cx), w_value_ext, w_value_int = self.worker(
            (percept_z, w_lstm, goals_horizon)
        )
        w_lstm = (w_hx, w_cx)
        return policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext


# Definir el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Inicialización del entorno
env = MazeEnv((6, 6), False, 3, True, render_mode="human")
initial = (
    torch.tensor(env.reset(random_start=True), dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)

H, W = initial.shape[2], initial.shape[3]
num_actions = env.action_space.n
horizon = 9

# Inicializar la red FuN y llevarla al dispositivo
net = FuN((H, W), num_actions, horizon, device).to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

# Inicialización de memorias LSTM
m_hx = torch.zeros(1, num_actions * 16, device=device)
m_cx = torch.zeros(1, num_actions * 16, device=device)
w_hx = torch.zeros(1, num_actions * 16, device=device)
w_cx = torch.zeros(1, num_actions * 16, device=device)
goals_horizon = torch.zeros(1, horizon + 1, num_actions * 16, device=device)

score = 0


# Función para calcular los retornos acumulados
def compute_returns(rewards, gamma=0.99):
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns, dtype=torch.float32)


# Bucle de entrenamiento
for global_steps in range(10000):
    state = (
        torch.tensor(env.reset(random_start=True), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Reiniciar memorias LSTM al inicio del episodio
    m_hx = torch.zeros(1, num_actions * 16, device=device)
    m_cx = torch.zeros(1, num_actions * 16, device=device)
    w_hx = torch.zeros(1, num_actions * 16, device=device)
    w_cx = torch.zeros(1, num_actions * 16, device=device)
    goals_horizon = torch.zeros(1, horizon + 1, num_actions * 16, device=device)

    done = False
    score = 0
    rewards = []
    policies = []
    values = []

    while not done:
        # Obtener la política y el valor
        policy, goal, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext = net(
            state, (m_hx, m_cx), (w_hx, w_cx), goals_horizon
        )

        # Actualizar memorias LSTM
        m_hx, m_cx = m_lstm
        w_hx, w_cx = w_lstm

        # Seleccionar acción
        action = torch.multinomial(policy.detach(), 1).item()

        # Almacenar política y valor para calcular la pérdida más adelante
        policies.append(policy[0, action])
        values.append(m_value.detach())

        # Tomar acción en el entorno
        next_state, reward, done = env.step(action)
        rewards.append(reward)

        # Actualizar el estado
        state = (
            torch.tensor(next_state, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )

        score += reward
        env.render()

    # Calcular retornos acumulados
    returns = compute_returns(rewards).to(device)
    values = torch.cat(values).squeeze()

    # Calcular pérdidas
    policy_loss = 0
    value_loss = F.mse_loss(values.detach(), returns.detach())
    for log_prob, R, value in zip(policies, returns.detach(), values.detach()):
        advantage = R - value
        policy_loss += -log_prob * advantage.detach()  # Pérdida de política

    loss = policy_loss + 0.5 * value_loss

    # Retropropagación
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"Episode: {global_steps + 1} | Score: {score:.2f} | Loss: {loss.item():.4f}")
