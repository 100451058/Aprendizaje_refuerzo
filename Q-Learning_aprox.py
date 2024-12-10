import re
from time import sleep
import numpy as np
import random
import gymnasium as gym
from src.maze import MazeEnv
import pygame

# Parámetros
episodios_usuario = 5
episodios_agente = 2000

class Manager:
    def __init__(self, env):
        self.env: MazeEnv = env
        self.q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], env.maze_shape[0], env.maze_shape[1]))  # Q-table para el manager
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.end = env._end

    def select_goal(self, state):

        if np.random.rand() < self.epsilon:
            coins = np.argwhere(self.env._maze == 5).tolist()
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
        self.q_tables =  {} # Q-table para el worker
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.learned_paths = []

    def select_action(self, state, goal):
        self.epsilon = self.epsilon *0.95 if self.epsilon > 0.10 else 0.1
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        else:
            q_table = self.q_tables[tuple(goal)]
            return np.argmax(q_table[state[0], state[1]])

    def update_q_table(self, state, action, reward, next_state, goal):
        q_table = self.q_tables[tuple(goal)]
        best_next_q = np.max(q_table[next_state[0], next_state[1]])
        q_table[state[0], state[1], action] += self.alpha * (reward + self.gamma * best_next_q - q_table[state[0], state[1], action])
        self.q_tables[tuple(goal)] = q_table  # Guardar el q_table actualizado
    
    def transfer_learning(self, goal):
        # print(f"Transfer learning to goal: {goal}")
        if(self.learned_paths == []):
            self.learned_paths.append(goal)
            q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], 4))
            self.q_tables[tuple(goal)] = q_table
        else:
            if goal in self.learned_paths: #APRENDIDO
                self.q_table = self.q_tables[tuple(goal)]
                self.epsilon = 0.1
            else: #NO APRENDIDO
                # Calcular path mas parecido
                min_distance = -1
                min_goal = None
                for learned_goal in self.learned_paths:
                    distance = sum(abs(np.array(goal)-np.array(learned_goal)))
                    # distance = np.linalg.norm(np.array(goal) - np.array(learned_goal))
                    if distance < min_distance or min_distance == -1:
                        min_distance = distance
                        min_goal = learned_goal
                
                #Crear nueva política
                if(min_distance < 3): #Si tengo una política parecida
                    self.q_tables[tuple(goal)] = self.q_tables[tuple(min_goal)]
                    # Le doy más libertad a investigar
                    self.epsilon = 0.5
                    
                else: #Si no tengo una política parecida
                    q_table = np.zeros((env.maze_shape[0], env.maze_shape[1], 4))
                    self.q_tables[tuple(goal)] = q_table
                    self.epsilon = 0.1

                # Añadir nuevo objetivo
                self.learned_paths.append(goal)


        


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

print("mazed")

env = MazeEnv((5, 5), False, 3, 'human')
manager = Manager(env)
worker = Worker(env)

# ITERACION DEL USUARIO
print("AYUDANDO AL AGENTE...")
# Bucle principal para controlar el agente con el teclado
for i in range(episodios_usuario):
    _ = env.reset()
    state = env.get_current_position()
    done  = False
    goal  = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    worker.transfer_learning(goal)
    print(f"user episode: {i+1:02d}/{episodios_usuario:02d} -> goal: {goal}")
    
    env.render()
    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                done = True
            elif event.type == pygame.KEYDOWN:
                action = get_action_from_key(event.key)
                if action is not None:
                    _, reward, done = env.step(action,goal)
                    next_state = env.get_current_position()
                    reward = 1 if reward<1 else reward
                    worker.update_q_table(state, action, reward, next_state,goal)
                    manager.update_q_table(state, goal, reward, next_state)
                    state = next_state

                    # Si se alcanza el objetivo, el episodio termina
                    if np.array_equal(state, goal):
                        goal = manager.select_goal(state)
                        manager.update_q_table(state, goal, reward, next_state)
                        worker.transfer_learning(goal)
                        print(f"user episode: {i+1:02d}/{episodios_usuario:02d} -> goal: {goal}")
        
        env.render()
    print()
env.close()
                    

# ITERACION DEL AGENTE
print("EL AGENTE SOLO...")
_ = env.reset()
state = env.get_current_position()
done = False
goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
env.render()

# Iteración por episodios
for i in range(episodios_agente):
    _ = env.reset()
    state = env.get_current_position()
    done = False
    goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
    worker.transfer_learning(goal)
    movimientos = 0

    # Iteración por pasos
    while not done and movimientos < 150:
        action = worker.select_action(state, goal)
        _, reward, done = env.step(action,goal)
        next_state = env.get_current_position()
        
        worker.update_q_table(state, action, reward, next_state,goal)
        manager.update_q_table(state, goal, reward, next_state)
        state = next_state
        env.render()
        movimientos+=1
        # Si se alcanza el objetivo, selecciona un nuevo objetivo
        if np.array_equal(state, goal):
            goal = manager.select_goal(state)
            manager.update_q_table(state, goal, reward, next_state)
            worker.transfer_learning(goal)

env.close()

#Probar la eficiencia de la politica
_ = env.reset()
state = env.get_current_position()
done = False
goal = manager.select_goal(state)  # El manager selecciona el objetivo al inicio del episodio
print(f"Objetivo: {goal}")
while not done:
    action = worker.select_action(state, goal)
    _, reward, done = env.step(action,goal)
    next_state = env.get_current_position()
    
    state = next_state
    env.render()
    # print(f"Estado: {state}, Recompensa: {reward}, Objetivo: {goal}")

    # Si se alcanza el objetivo, selecciona un nuevo objetivo
    if np.array_equal(state, goal):
        print(f"Objetivo: {goal}")
        goal = manager.select_goal(state)
        manager.update_q_table(state, goal, reward, next_state)
    
    if(done):
        print("SE LOGRO ENCONTRAR EL CAMINO OPTIMO")


