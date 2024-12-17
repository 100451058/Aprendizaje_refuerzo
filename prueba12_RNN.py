from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import random
from collections import namedtuple
from src.maze import MazeEnv
from tensorboardX import SummaryWriter
import keyboard
import time
import torch
import torch.nn.functional as F


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


# Función para calcular retornos acumulados
def get_returns(rewards, masks, gamma, values):
    returns = torch.zeros_like(rewards)
    running_returns = values[-1].squeeze()

    for t in reversed(range(0, len(rewards) - 1)):
        running_returns = rewards[t] + gamma * running_returns * masks[t]
        returns[t] = running_returns

    if returns.std() != 0:
        returns = (returns - returns.mean()) / returns.std()

    return returns


# Función de entrenamiento adaptada
def train_model(net, optimizer, transition, args):

    # Convertir transiciones a tensores
    actions = torch.tensor(transition.action).long().to(args.device)
    rewards = torch.tensor(transition.reward).to(args.device)
    masks = torch.tensor(transition.mask).to(args.device)
    goals = torch.stack(transition.goal).to(args.device)
    policies = torch.stack(transition.policy).squeeze(1).to(device)
    m_states = torch.stack(transition.m_state).to(args.device)
    m_values = torch.stack(transition.m_value).to(args.device)
    w_values_ext = torch.stack(transition.w_value_ext).to(args.device)
    w_values_int = torch.stack(transition.w_value_int).to(args.device)

    # Calcular retornos extrínsecos e intrínsecos
    m_returns = get_returns(rewards, masks, args.m_gamma, m_values)
    w_returns = get_returns(rewards, masks, args.w_gamma, w_values_ext)

    # Calcular recompensas intrínsecas
    intrinsic_rewards = torch.zeros_like(rewards).to(args.device)
    for i in range(args.horizon, len(rewards)):
        cos_sum = 0
        for j in range(1, args.horizon + 1):
            alpha = m_states[i] - m_states[i - j]
            beta = goals[i - j]
            cosine_sim = F.cosine_similarity(alpha, beta, dim=-1)
            cos_sum += cosine_sim
        intrinsic_rewards[i] = (cos_sum / args.horizon).detach()

    returns_int = get_returns(intrinsic_rewards, masks, args.w_gamma, w_values_int)

    # Cálculo de pérdidas
    m_loss = torch.zeros_like(w_returns).to(args.device)
    w_loss = torch.zeros_like(m_returns).to(args.device)

    for i in range(0, len(rewards) - args.horizon):
        m_advantage = m_returns[i] - m_values[i].squeeze(-1)
        alpha = m_states[i + args.horizon] - m_states[i]
        beta = goals[i]
        cosine_sim = F.cosine_similarity(alpha.detach(), beta, dim=-1)
        m_loss[i] = -m_advantage * cosine_sim
        log_policy = torch.log(policies[i] + 1e-5)
        w_advantage = (
            w_returns[i]
            + returns_int[i]
            - w_values_ext[i].squeeze(-1)
            - w_values_int[i].squeeze(-1)
        )
        log_policy = log_policy.gather(-1, actions[i])
        w_loss[i] = -w_advantage * log_policy.squeeze(-1)

    # Promediar las pérdidas
    m_loss = m_loss.mean()
    w_loss = w_loss.mean()
    m_loss_value = F.mse_loss(m_values.squeeze(-1), m_returns.detach())
    w_loss_value_ext = F.mse_loss(w_values_ext.squeeze(-1), w_returns.detach())
    w_loss_value_int = F.mse_loss(w_values_int.squeeze(-1), returns_int.detach())

    # Combinar las pérdidas
    loss = w_loss + w_loss_value_ext + w_loss_value_int + m_loss + m_loss_value

    # Optimización
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), args.clip_grad_norm)
    optimizer.step()

    return loss.item()


# Parámetros de configuración
class Args:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_gamma = 0.99
        self.w_gamma = 0.99
        self.horizon = 5
        self.clip_grad_norm = 0.5


args = Args()


# Definir el dispositivo
device = args.device
maze_size = (4, 4)
# Inicialización del entorno
env = MazeEnv(maze_size, False, 3, True, render_mode="rgb-array")
num_actions = env.action_space.n
horizon = 9
initial = (
    torch.tensor(env.reset(random_start=True), dtype=torch.float32)
    .unsqueeze(0)
    .unsqueeze(0)
    .to(device)
)

H, W = initial.shape[2], initial.shape[3]

# Transición
Transition = namedtuple(
    "Transition",
    [
        "action",
        "reward",
        "mask",
        "goal",
        "policy",
        "m_state",
        "m_value",
        "w_value_ext",
        "w_value_int",
    ],
)

# Inicializar red
net = FuN((H, W), num_actions, horizon, device).to(device)
optimizer = torch.optim.RMSprop(net.parameters(), lr=0.00025, eps=0.01)

# Entrenamiento
# Flujo de entrenamiento adaptado a FeUdal Networks
for global_steps in range(10000):
    state = (
        torch.tensor(env.reset(random_start=True), dtype=torch.float32)
        .unsqueeze(0)
        .unsqueeze(0)
        .to(device)
    )

    # Reiniciar memorias LSTM
    m_hx, m_cx = torch.zeros(1, num_actions * 16, device=device), torch.zeros(
        1, num_actions * 16, device=device
    )
    w_hx, w_cx = torch.zeros(1, num_actions * 16, device=device), torch.zeros(
        1, num_actions * 16, device=device
    )
    goals_horizon = torch.zeros(1, horizon + 1, num_actions * 16, device=device)

    transitions = []
    done = False
    step_count = 0

    # El Manager define su primer goal
    with torch.no_grad():
        _, goal, goals_horizon, m_lstm, _, m_value, _ = net(
            state, (m_hx, m_cx), (w_hx, w_cx), goals_horizon
        )

    while not done:
        step_count += 1

        # Obtener la política y valores actuales
        with torch.no_grad():
            policy, _, goals_horizon, m_lstm, w_lstm, m_value, w_value_ext = net(
                state, m_lstm, (w_hx, w_cx), goals_horizon
            )

        # Seleccionar acción
        action = torch.multinomial(policy, 1).item()
        next_state, reward_ext, done = env.step(action)

        # Calcular recompensa intrínseca basada en la similitud coseno
        with torch.no_grad():
            state_diff = m_lstm[0] - m_hx  # Progreso observado
            cosine_sim = F.cosine_similarity(state_diff, goal, dim=-1)
            reward_int = cosine_sim.item()

        # Recompensa total
        total_reward = reward_ext + reward_int

        # Registrar transición
        mask = 0 if done else 1
        transitions.append(
            Transition(
                action,
                total_reward,
                mask,
                goal,
                policy,
                m_lstm[0],
                m_value,
                w_value_ext,
                w_value_ext,
            )
        )

        # Actualizar el estado
        state = (
            torch.tensor(next_state, dtype=torch.float32)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(device)
        )
        m_hx, m_cx = m_lstm  # Actualizar memoria Manager
        w_hx, w_cx = w_lstm  # Actualizar memoria Worker

        # Actualizar el goal del Manager cada horizonte
        if step_count % args.horizon == 0:
            _, goal, goals_horizon, m_lstm, _, m_value, _ = net(
                state, m_lstm, (w_hx, w_cx), goals_horizon
            )

    # Entrenamiento del modelo después del episodio
    loss = train_model(net, optimizer, Transition(*zip(*transitions)), args)
    print(f"Episode: {global_steps + 1} | Loss: {loss:.4f}")
