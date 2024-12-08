import numpy as np
import random
import gymnasium as gym
from src.maze import MazeEnv
import pygame

# Parámetros
episodios_usuario = 5
episodios_agente = 5

class Manager:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], env.maze_shape[0], env.maze_shape[1]))  # Q-table para el manager
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.end = env._end

    def select_goal(self, state):
        if np.random.rand() < self.epsilon:
            coins = np.argwhere(self.env._maze == 5)
            if len(coins) > 0:
                return coins[np.random.randint(len(coins))]
            else:
                return self.env._end
        else:
            end = [int(item) for item in self.end]
            goals = np.argwhere(self.env._maze == 5).tolist() + [end]
            q_values = [self.q_table[state[0], state[1], goal[0], goal[1]] for goal in goals]
            return goals[np.argmax(q_values)]

    def update_q_table(self, state, goal, reward, next_state):
        end = [int(item) for item in self.end]
        best_next_q = np.max([self.q_table[next_state[0], next_state[1], g[0], g[1]] for g in np.argwhere(self.env._maze == 5).tolist() + [end]])
        self.q_table[state[0], state[1], goal[0], goal[1]] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state[0], state[1], goal[0], goal[1]])

class Worker:
    def __init__(self, env):
        self.env = env
        self.q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], 4))  # Q-table para el worker
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1

    def select_action(self, state, goal):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state[0], next_state[1]])
        self.q_table[state[0], state[1], action] += self.alpha * (reward + self.gamma * best_next_q - self.q_table[state[0], state[1], action])


# Función para mapear las teclas a acciones
def get_action_from_key(key):
    if key == pygame.K_RIGHT:
        return 0
    elif key == pygame.K_DOWN:
        return 1
    elif key == pygame.K_LEFT:
        return 2
    elif key == pygame.K_UP:
        return 3
    return None

env = MazeEnv((5, 5), False, 3, 'human')
manager = Manager(env)
worker = Worker(env)

# ITERACION DEL USUARIO
state = env.reset()
done = False
goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
# Bucle principal para controlar el agente con el teclado
for i in range(episodios_usuario):
    state = env.reset()
    done = False
    goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True
            elif event.type == pygame.KEYDOWN:
                action = get_action_from_key(event.key)
                if action is not None:
                    next_state, reward, done = env.step(action)
                    worker.update_q_table(state, action, reward, next_state)
                    manager.update_q_table(state, goal, reward, next_state)
                    state = next_state
                    env.render()
                    # print(f"Estado: {state}, Recompensa: {reward}, Objetivo: {goal}")

                    # Si se alcanza el objetivo, el episodio termina
                    if np.array_equal(state, goal):
                        goal = manager.select_goal(state)
                        manager.update_q_table(state, goal, reward, next_state)
                    





# ITERACION DEL AGENTE
state = env.reset()
done = False
goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio

# Iteración por episodios
for i in range(episodios_agente):
    state = env.reset()
    done = False
    goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio

    # Iteración por pasos
    while not done:
        action = worker.select_action(state, goal)
        next_state, reward, done = env.step(action)
        worker.update_q_table(state, action, reward, next_state)
        manager.update_q_table(state, goal, reward, next_state)
        state = next_state
        env.render()
        # print(f"Estado: {state}, Recompensa: {reward}, Objetivo: {goal}")

        # Si se alcanza el objetivo, selecciona un nuevo objetivo
        if np.array_equal(state, goal):
            goal = manager.select_goal(state)
            manager.update_q_table(state, goal, reward, next_state)

env.close()

# PRINT THE Q-TABLES
print("Q-table del manager:")
print(manager.q_table)

print("Q-table del worker:")
print(worker.q_table)